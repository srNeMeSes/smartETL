"""Shapiro-Wilk como aviso de normalidade nos testes de médias e nas ANOVAs.

Fonte dos valores de referência: tests/referencias_r/referencias_normalidade.json, gerado por
tests/referencias_r/gerar_referencias_normalidade.R com R 4.6.1 (`shapiro.test`) sobre mtcars
(mpg; mpg por am), sleep (diferenças pareadas), PlantGrowth (resíduos da ANOVA de 1 fator) e
ToothGrowth (resíduos da ANOVA de 2 fatores, com e sem interação). Os resíduos da ANOVA de 1
fator também são conferidos à mão (valor menos a média do grupo).
"""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import RAIZ
from scipy import stats

from core.diagnosticos import (
    aviso_normalidade,
    normalidade_amostra,
    rotulo_normalidade,
)
from core.testes.anova import TesteAnova1Fator, TesteAnova2Fatores
from core.testes.medias import TesteT1Amostra, TesteT2Amostras, TesteTPareado

PASTA = RAIZ / "tests" / "referencias_r"
R = json.loads((PASTA / "referencias_normalidade.json").read_text(encoding="utf-8"))
REL = 1e-6


def _df(nome):
    return pd.read_csv(PASTA / f"normalidade_{nome}.csv")


def _resumo(resultado):
    return dict(resultado.tabelas["Resumo"].itertuples(index=False))


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------


def test_normalidade_amostra_casos_de_borda():
    assert normalidade_amostra(np.array([1.0, 2.0])) is None  # n < 3
    assert normalidade_amostra(np.array([3.0, 3.0, 3.0, 3.0])) is None  # constante
    r = normalidade_amostra(np.array([1.0, np.nan, 2.0, 4.0, 8.0]))
    assert r.n == 4 and r.teste == "Shapiro-Wilk"
    w, p = stats.shapiro([1.0, 2.0, 4.0, 8.0])
    assert (r.estatistica, r.p_valor) == pytest.approx((w, p), rel=1e-12)
    grande = normalidade_amostra(np.random.default_rng(1).normal(size=6000))
    assert grande.teste == "Kolmogorov-Smirnov (Lilliefors)" and grande.n == 6000


def test_rotulo_e_aviso():
    assert rotulo_normalidade(None) == "p-valor do Shapiro-Wilk"
    assert rotulo_normalidade(None, "resíduos") == "p-valor do Shapiro-Wilk (resíduos)"
    assert aviso_normalidade(None, "x", "o Wilcoxon", "o teste t") is None
    assimetrica = normalidade_amostra(np.exp(np.arange(10.0)))
    aviso = aviso_normalidade(assimetrica, "'x'", "o Wilcoxon (uma amostra)", "o teste t")
    assert aviso.startswith("O teste de Shapiro-Wilk indica que 'x' não segue distribuição")
    assert "não segue distribuição normal" in aviso
    assert "Considere o Wilcoxon (uma amostra)." in aviso and "costuma" not in aviso
    grande = normalidade_amostra(np.exp(np.linspace(0, 10, 40)))
    aviso = aviso_normalidade(grande, "os resíduos", "o Kruskal-Wallis", "a ANOVA", plural=True)
    assert "os resíduos não seguem distribuição normal (p < 0,001)" in aviso
    assert aviso.endswith("Com n ≥ 30, a falta de normalidade costuma afetar pouco a ANOVA.")
    normal = normalidade_amostra(np.random.default_rng(2).normal(size=50))
    assert not normal.rejeita and aviso_normalidade(normal, "x", "o W", "o t") is None


# ---------------------------------------------------------------------------
# Testes de médias contra o R
# ---------------------------------------------------------------------------


def test_t_uma_amostra():
    r = TesteT1Amostra().executar(_df("mtcars"), {"coluna": "mpg", "mu0": "20", "alfa": 0.05})
    assert r.estatisticas["p_normalidade"] == pytest.approx(R["t_1am"]["p"], rel=REL)
    assert "p-valor do Shapiro-Wilk" in _resumo(r)
    assert not any("Shapiro" in a for a in r.avisos)  # p = 0,123


def test_t_duas_amostras():
    r = TesteT2Amostras().executar(_df("mtcars"), {"coluna": "mpg", "grupo": "am", "alfa": 0.05})
    e = r.estatisticas
    assert e["p_normalidade1"] == pytest.approx(R["t_2am_automatico"]["p"], rel=REL)
    assert e["p_normalidade2"] == pytest.approx(R["t_2am_manual"]["p"], rel=REL)
    resumo = _resumo(r)
    assert "p-valor do Shapiro-Wilk (grupo 'automatico')" in resumo
    assert "p-valor do Shapiro-Wilk (grupo 'manual')" in resumo


def test_t_pareado_com_aviso():
    r = TesteTPareado().executar(
        _df("sleep"), {"coluna1": "antes", "coluna2": "depois", "alfa": 0.05}
    )
    assert r.estatisticas["p_normalidade"] == pytest.approx(R["t_pareado"]["p"], rel=REL)
    assert _resumo(r)["p-valor do Shapiro-Wilk (diferenças)"] == "0,033"
    (aviso,) = [a for a in r.avisos if "Shapiro" in a]
    assert "a diferença 'antes' − 'depois' não segue distribuição normal (p = 0,033)" in aviso
    assert "Considere o Wilcoxon (pareado)." in aviso and "costuma" not in aviso  # n = 10


def test_t_duas_amostras_com_aviso_e_grupo_pequeno():
    df = pd.DataFrame(
        {"v": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 50.0, 2.0, 3.0], "g": ["A"] * 8 + ["B"] * 2}
    )
    r = TesteT2Amostras().executar(df, {"coluna": "v", "grupo": "g", "alfa": 0.05})
    assert r.estatisticas["p_normalidade1"] < 0.05 and np.isnan(r.estatisticas["p_normalidade2"])
    assert _resumo(r)["p-valor do Shapiro-Wilk (grupo 'B')"] == "—"
    (aviso,) = [a for a in r.avisos if "Shapiro" in a]
    assert "'v' no grupo 'A'" in aviso and "Mann-Whitney" in aviso


# ---------------------------------------------------------------------------
# ANOVAs contra o R
# ---------------------------------------------------------------------------


def test_anova_1fator_residuos():
    df = _df("plantgrowth")
    r = TesteAnova1Fator().executar(df, {"coluna": "peso", "grupo": "grupo", "alfa": 0.05})
    assert r.estatisticas["p_normalidade"] == pytest.approx(R["anova_1fator"]["p"], rel=REL)
    residuos = df.peso - df.groupby("grupo").peso.transform("mean")
    assert r.estatisticas["p_normalidade"] == pytest.approx(stats.shapiro(residuos)[1], rel=1e-9)
    assert "p-valor do Shapiro-Wilk (resíduos)" in _resumo(r)
    assert not any("Shapiro" in a for a in r.avisos)


def test_anova_1fator_com_aviso():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"v": rng.exponential(size=60), "g": ["a", "b", "c"] * 20})
    r = TesteAnova1Fator().executar(df, {"coluna": "v", "grupo": "g", "alfa": 0.05})
    (aviso,) = [a for a in r.avisos if "Shapiro" in a]
    assert "os resíduos da ANOVA não seguem distribuição normal" in aviso
    assert "Considere o Kruskal-Wallis." in aviso
    assert aviso.endswith("a falta de normalidade costuma afetar pouco a ANOVA.")  # n = 60


@pytest.mark.parametrize(("interacao", "chave"), [(True, "interacao"), (False, "aditivo")])
def test_anova_2fatores_residuos(interacao, chave):
    params = {
        "coluna": "len",
        "fator_a": "supp",
        "fator_b": "dose",
        "interacao": interacao,
        "alfa": 0.05,
    }
    r = TesteAnova2Fatores().executar(_df("toothgrowth"), params)
    ref = R[f"anova_2fator_{chave}"]["p"]
    assert r.estatisticas["p_normalidade"] == pytest.approx(ref, rel=REL)
    assert "p-valor do Shapiro-Wilk (resíduos)" in _resumo(r)
    assert not any("Shapiro" in a for a in r.avisos)  # p = 0,669 / 0,057


def test_anova_2fatores_com_aviso():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {"v": rng.exponential(size=80), "a": ["x", "y"] * 40, "b": ["p", "p", "q", "q"] * 20}
    )
    params = {"coluna": "v", "fator_a": "a", "fator_b": "b", "alfa": 0.05}
    r = TesteAnova2Fatores().executar(df, params)
    (aviso,) = [a for a in r.avisos if "Shapiro" in a]
    assert "Considere uma transformação de 'v' (ex.: logaritmo)." in aviso
