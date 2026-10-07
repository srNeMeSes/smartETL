"""Otimizações da Fase 5: os caminhos rápidos dão os mesmos valores dos caminhos de referência.

- Estatísticas de ordem por contagem (Hodges-Lehmann com n grande) = ordenação de todos os pares;
- odds ratio condicional vetorizada (core/exatos.py) = `scipy.stats.contingency.odds_ratio`;
- IC exato da mediana com a cdf vetorizada = laço original sobre k;
- e o que antes falhava ou demorava (n = 100 000) agora roda.
"""

import math
import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.exatos import ic_odds_ratio_condicional, odds_ratio_condicional
from core.testes import nao_parametricos as np_testes
from core.testes.nao_parametricos import (
    TesteMannWhitney,
    TesteWilcoxon,
    deslocamento_hodges_lehmann,
    hodges_lehmann,
    ic_mediana_exato,
    kesimo_diferenca,
    kesimo_walsh,
)


@pytest.mark.parametrize("semente", [1, 2, 3])
def test_kesimo_walsh_igual_a_ordenar_todos_os_pares(semente):
    rng = np.random.default_rng(semente)
    x = np.sort(np.round(rng.normal(0, 3, 300), 1))  # arredondado: muitos empates
    i, j = np.triu_indices(len(x))
    walsh = np.sort((x[i] + x[j]) / 2)
    for k in (1, 2, 17, len(walsh) // 2, len(walsh) - 1, len(walsh)):
        assert kesimo_walsh(x, k) == pytest.approx(walsh[k - 1], abs=1e-12)  # 1 ulp


@pytest.mark.parametrize("semente", [4, 5])
def test_kesimo_diferenca_igual_a_ordenar_todos_os_pares(semente):
    rng = np.random.default_rng(semente)
    x1 = np.sort(np.round(rng.normal(10, 2, 220), 2))
    x2 = np.sort(np.round(rng.normal(11, 3, 180), 2))
    dif = np.sort((x1[:, None] - x2[None, :]).ravel())
    for k in (1, 3, 999, len(dif) // 2, len(dif)):
        assert kesimo_diferenca(x1, x2, k) == pytest.approx(dif[k - 1], abs=1e-12)


@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_hodges_lehmann_por_contagem_igual_ao_enumerado(monkeypatch, alternativa):
    rng = np.random.default_rng(9)
    d = np.round(rng.normal(1, 2, 400), 1)
    x1, x2 = np.round(rng.normal(5, 1, 150), 2), np.round(rng.normal(5.5, 1, 170), 2)
    enumerado = hodges_lehmann(d, 0.05, alternativa, False)
    desloc_enumerado = deslocamento_hodges_lehmann(x1, x2, 0.05, alternativa, False)
    monkeypatch.setattr(np_testes, "MAX_PARES_ENUMERADOS", 0)  # força o caminho por contagem
    assert hodges_lehmann(d, 0.05, alternativa, False) == pytest.approx(enumerado, rel=1e-12)
    assert deslocamento_hodges_lehmann(x1, x2, 0.05, alternativa, False) == pytest.approx(
        desloc_enumerado, rel=1e-12
    )


TABELAS = [
    (3, 1, 1, 3),
    (10, 2, 3, 15),
    (0, 5, 6, 4),
    (7, 0, 2, 9),
    (12, 5, 9, 11),
    (1, 9, 9, 1),
    (40, 22, 31, 55),
    (3, 0, 0, 3),
]


@pytest.mark.parametrize("tabela", TABELAS)
@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_odds_ratio_condicional_vetorizada_igual_ao_scipy(tabela, alternativa):
    a, b, c, d = tabela
    ref = stats.contingency.odds_ratio([[a, b], [c, d]], kind="conditional")
    estimativa = odds_ratio_condicional(a, b, c, d)
    if math.isfinite(ref.statistic) and ref.statistic > 0:
        assert estimativa == pytest.approx(ref.statistic, rel=1e-9)
    else:
        assert estimativa == ref.statistic
    ic = ref.confidence_interval(confidence_level=0.95, alternative=alternativa)
    assert ic_odds_ratio_condicional(a, b, c, d, 0.95, alternativa) == pytest.approx(
        (ic.low, ic.high), rel=1e-8
    )


def _ic_mediana_laco(valores, alfa, alternativa):
    """O laço original sobre k (referência)."""
    x = np.sort(valores)
    n = len(x)
    cauda = alfa / 2 if alternativa == "two-sided" else alfa
    k = 0
    for j in range(1, n // 2 + 2):
        if stats.binom.cdf(j - 1, n, 0.5) <= cauda:
            k = j
    return k


@pytest.mark.parametrize("n", [5, 6, 9, 40, 333])
@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_ic_mediana_vetorizado_igual_ao_laco(n, alternativa):
    x = np.random.default_rng(n).normal(size=n)
    k = _ic_mediana_laco(x, 0.05, alternativa)
    inferior, superior, _ = ic_mediana_exato(x, 0.05, alternativa)
    ordenado = np.sort(x)
    if k == 0:
        assert (inferior, superior) == (-math.inf, math.inf)
    elif alternativa == "two-sided":
        assert (inferior, superior) == (ordenado[k - 1], ordenado[n - k])


def test_wilcoxon_e_mann_whitney_com_n_grande_nao_estouram_a_memoria():
    rng = np.random.default_rng(0)
    n = 100_000
    df = pd.DataFrame({"y": np.round(rng.normal(41, 5, n), 3), "g": rng.choice(["A", "B"], n)})
    inicio = time.perf_counter()
    r = TesteWilcoxon().executar(
        df, {"modo": "Uma amostra", "coluna1": "y", "m0": 40, "alfa": 0.05}
    )
    assert r.estatisticas["pseudomediana"] == pytest.approx(41, abs=0.1)
    r = TesteMannWhitney().executar(df, {"coluna": "y", "grupo": "g", "alfa": 0.05})
    assert abs(r.estatisticas["deslocamento_hl"]) < 0.2
    assert time.perf_counter() - inicio < 30


def _curva_roc_ingenua(y, p):
    """A versão O(n²) original (referência)."""
    limiares = np.unique(p)[::-1]
    n1, n0 = y.sum(), len(y) - y.sum()
    tpr = [0.0] + [float(((p >= t) & (y == 1)).sum() / n1) for t in limiares]
    fpr = [0.0] + [float(((p >= t) & (y == 0)).sum() / n0) for t in limiares]
    return np.array(fpr), np.array(tpr)


@pytest.mark.parametrize("semente", [1, 2])
def test_curva_roc_vetorizada_igual_a_ingenua(semente):
    from core.diagnosticos import curva_roc

    rng = np.random.default_rng(semente)
    p = np.round(rng.uniform(size=500), 2)  # muitos empates
    y = (rng.uniform(size=500) < p).astype(float)
    fpr, tpr = curva_roc(y, p)
    fpr_ref, tpr_ref = _curva_roc_ingenua(y, p)
    assert fpr == pytest.approx(fpr_ref, abs=1e-15) and tpr == pytest.approx(tpr_ref, abs=1e-15)


def test_postos_por_linha_do_friedman_iguais_ao_laco():
    dados = np.round(np.random.default_rng(3).normal(size=(200, 4)), 1)
    assert np.array_equal(
        stats.rankdata(dados, axis=1), np.apply_along_axis(stats.rankdata, 1, dados)
    )


def _xlsx_misto(tmp_path):
    df = pd.DataFrame(
        {
            "inteiro": [1, 2, 3, 4],
            "decimal": [1.5, np.nan, 3.25, -2.0],
            "texto": ["a", "b", None, "d"],
            "misto": ["1,5", "x", "2", "3"],
        }
    )
    caminho = tmp_path / "misto.xlsx"
    df.to_excel(caminho, index=False)
    return caminho


def test_xlsx_calamine_igual_ao_openpyxl(tmp_path):
    from core.io import _ler_planilha

    conteudo = _xlsx_misto(tmp_path).read_bytes()
    _, rapido = _ler_planilha(conteudo, "calamine")
    _, referencia = _ler_planilha(conteudo, "openpyxl")
    pd.testing.assert_frame_equal(rapido, referencia)


def test_xlsx_usa_openpyxl_se_o_calamine_falhar(tmp_path, monkeypatch):
    from core import io as modulo

    original = modulo._ler_planilha
    usados = []

    def falha_no_calamine(conteudo, motor):
        usados.append(motor)
        if motor == "calamine":
            raise ImportError("sem calamine")
        return original(conteudo, motor)

    monkeypatch.setattr(modulo, "_ler_planilha", falha_no_calamine)
    dados = modulo.carregar_dados(_xlsx_misto(tmp_path))
    assert usados == ["calamine", "openpyxl"]
    assert list(dados.df.columns) == ["inteiro", "decimal", "texto", "misto"]
