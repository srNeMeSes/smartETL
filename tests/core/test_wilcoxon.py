"""Wilcoxon dos postos sinalizados: checklist da seção 9 do CLAUDE.md.

Fontes dos valores de referência (independentes do wrapper, que usa scipy):
- W⁺ e W⁻ com postos médios calculados à mão;
- p exato: enumeração das 2ⁿ atribuições de sinal (tests/referencias.py), dados sem empates;
- p aproximado: z = (W⁺ − n(n+1)/4)/√(n(n+1)(2n+1)/24 − Σ(t³ − t)/48), normal de
  `statistics.NormalDist` (sem correção de continuidade);
- pseudomediana de Hodges-Lehmann: mediana das médias de Walsh com numpy; IC exato pelos quantis
  da distribuição de W⁺ obtida por enumeração (método do `wilcox.test` do R);
- r = z/√n.
"""

import itertools
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from referencias import wilcoxon_exato

from core.base import ErroValidacao
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.nao_parametricos import (
    ALTERNATIVAS_SINAL,
    PAREADO,
    UMA_AMOSTRA,
    TesteWilcoxon,
    distribuicao_postos_sinalizados,
)

REL = 1e-9
# Diferenças em relação a M₀ = 10 com módulos distintos (sem empates nem zeros): p exato.
SEM_EMPATES = [12.1, 8.7, 15.3, 10.4, 21.9, 9.2, 18.6, 13.8, 7.5, 16.2, 11.6]
# Com empates e dois valores iguais a M₀ = 10: aproximação normal.
COM_EMPATES = [12, 8, 15, 10, 22, 9, 18, 10, 14, 25, 11, 16, 7, 19, 13]


@pytest.fixture
def teste():
    return TesteWilcoxon()


def _df(valores):
    return pd.DataFrame({"x": valores, "g": ["a"] * len(valores)})


def _params(**extra):
    return {
        "modo": UMA_AMOSTRA,
        "coluna1": "x",
        "coluna2": None,
        "m0": 10,
        "alternativa": "M ≠ M₀",
        "alfa": 0.05,
    } | extra


def _rotulo(alternativa: str) -> str:
    return next(r for r, a in ALTERNATIVAS_SINAL.items() if a == alternativa)


def _postos_medios(valores):
    ordem = sorted(range(len(valores)), key=lambda i: valores[i])
    postos = [0.0] * len(valores)
    i = 0
    while i < len(ordem):
        j = i
        while j + 1 < len(ordem) and valores[ordem[j + 1]] == valores[ordem[i]]:
            j += 1
        for k in range(i, j + 1):
            postos[ordem[k]] = (i + j) / 2 + 1
        i = j + 1
    return postos


def _manual(valores, m0):
    d = [v - m0 for v in valores if v != m0]
    postos = _postos_medios([abs(v) for v in d])
    w_mais = sum(p for p, v in zip(postos, d, strict=True) if v > 0)
    w_menos = sum(p for p, v in zip(postos, d, strict=True) if v < 0)
    n = len(d)
    empates = [postos.count(p) for p in set(postos)]
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t**3 - t for t in empates) / 48
    z = (w_mais - n * (n + 1) / 4) / math.sqrt(var)
    return d, w_mais, w_menos, z


def _p_normal(z, alternativa):
    phi = NormalDist().cdf
    return {"two-sided": 2 * (1 - phi(abs(z))), "greater": 1 - phi(z), "less": phi(z)}[alternativa]


def _qsignrank_enumerado(p, n):
    somas = sorted(
        sum(r for r, s in zip(range(1, n + 1), sinais, strict=True) if s)
        for sinais in itertools.product([0, 1], repeat=n)
    )
    total = len(somas)
    for w in range(n * (n + 1) // 2 + 1):
        if sum(1 for s in somas if s <= w) / total >= p - 1e-12:
            return w
    return n * (n + 1) // 2


# ---------------------------------------------------------------------------
# Valores de referência
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_p_exato_sem_empates(teste, alternativa):
    r = teste.executar(_df(SEM_EMPATES), _params(alternativa=_rotulo(alternativa)))
    d, w_mais, w_menos, _ = _manual(SEM_EMPATES, 10)
    assert r.estatisticas["usou_exato"] == 1.0
    assert (r.estatisticas["w_mais"], r.estatisticas["w_menos"]) == pytest.approx((w_mais, w_menos))
    bilateral, maior, menor = wilcoxon_exato(d)
    esperado = {"two-sided": bilateral, "greater": maior, "less": menor}[alternativa]
    assert r.p_valor == pytest.approx(esperado, rel=REL)


@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_p_aproximado_com_empates(teste, alternativa):
    r = teste.executar(_df(COM_EMPATES), _params(alternativa=_rotulo(alternativa)))
    _, w_mais, w_menos, z = _manual(COM_EMPATES, 10)
    assert r.estatisticas["usou_exato"] == 0.0
    assert (r.estatisticas["w_mais"], r.estatisticas["w_menos"]) == pytest.approx((w_mais, w_menos))
    assert r.estatisticas["z"] == pytest.approx(z, rel=REL)
    assert r.p_valor == pytest.approx(_p_normal(z, alternativa), rel=1e-7)
    assert r.estatisticas["r"] == pytest.approx(z / math.sqrt(13), rel=REL)


def test_amostra_grande_usa_aproximacao(teste):
    x = np.random.default_rng(1).normal(11, 2, 60)
    r = teste.executar(_df(x), _params())
    _, _, _, z = _manual(list(x), 10)
    assert r.estatisticas["usou_exato"] == 0.0
    assert r.p_valor == pytest.approx(_p_normal(z, "two-sided"), rel=1e-7)


def test_distribuicao_exata_contra_enumeracao():
    for n in (1, 4, 7):
        contagens = distribuicao_postos_sinalizados(n)
        somas = [
            sum(r for r, s in zip(range(1, n + 1), sinais, strict=True) if s)
            for sinais in itertools.product([0, 1], repeat=n)
        ]
        assert contagens.tolist() == [somas.count(w) for w in range(n * (n + 1) // 2 + 1)]


@pytest.mark.parametrize("alfa", [0.01, 0.05, 0.10])
@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_hodges_lehmann_e_ic_exato(teste, alfa, alternativa):
    r = teste.executar(_df(SEM_EMPATES), _params(alfa=alfa, alternativa=_rotulo(alternativa)))
    d = np.array(SEM_EMPATES) - 10
    walsh = sorted((d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d)))
    assert r.estatisticas["pseudomediana"] == pytest.approx(float(np.median(walsh)) + 10)
    n, m = len(d), len(walsh)
    cauda = alfa / 2 if alternativa == "two-sided" else alfa
    qu = max(_qsignrank_enumerado(cauda, n), 1)
    esperado = {
        "two-sided": (walsh[qu - 1] + 10, walsh[m - qu] + 10),
        "greater": (walsh[qu - 1] + 10, math.inf),
        "less": (-math.inf, walsh[m - qu] + 10),
    }[alternativa]
    assert (r.estatisticas["ic_inferior"], r.estatisticas["ic_superior"]) == pytest.approx(esperado)


def test_ic_aproximado_com_empates(teste):
    r = teste.executar(_df(COM_EMPATES), _params())
    d = np.array([v - 10 for v in COM_EMPATES if v != 10], dtype=float)
    walsh = sorted((d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d)))
    n = len(d)
    z = NormalDist().inv_cdf(0.975)
    k = math.floor(n * (n + 1) / 4 - z * math.sqrt(n * (n + 1) * (2 * n + 1) / 24))
    assert (r.estatisticas["ic_inferior"], r.estatisticas["ic_superior"]) == pytest.approx(
        (walsh[k - 1] + 10, walsh[len(walsh) - k] + 10)
    )


# ---------------------------------------------------------------------------
# Zeros, modo pareado, decisão e saída
# ---------------------------------------------------------------------------


def test_zeros_descartados_com_aviso(teste):
    r = teste.executar(_df(COM_EMPATES), _params())
    assert (r.estatisticas["zeros"], r.estatisticas["n"]) == (2, 13)
    assert any("2 diferença(s) nula(s)" in a and "n = 13" in a for a in r.avisos)
    assert any("empates nos valores absolutos" in a for a in r.avisos)
    assert any("simétrica" in a for a in r.avisos)


def test_pareado(teste):
    antes = [140, 152, 138, 160, 145, 149, 155, 139, 147, 158]
    depois = [132.5, 141, 139, 149.5, 145, 141, 147.3, 134, 146.2, 150.1]
    df = pd.DataFrame({"antes": antes, "depois": depois})
    r = teste.executar(df, _params(modo=PAREADO, coluna1="antes", coluna2="depois", m0=0))
    d = [a - b for a, b in zip(antes, depois, strict=True)]
    nao_nulas = [v for v in d if v != 0]
    assert r.estatisticas["n"] == len(nao_nulas) == 9
    bilateral, _, _ = wilcoxon_exato(nao_nulas)
    assert r.p_valor == pytest.approx(bilateral, rel=REL)
    assert "a mediana das diferenças 'antes' − 'depois'" in r.interpretacao
    assert r.figuras[0].titulo == "Diferenças 'antes' − 'depois'"


def test_decisao_e_interpretacao(teste):
    r = teste.executar(_df(COM_EMPATES), _params())
    assert r.decisao == REJEITA_H0
    assert "rejeita-se H₀ (M = 10) em favor de H₁ (M ≠ 10)" in r.interpretacao
    assert "Há evidência estatística de que a mediana de 'x' é diferente de 10." in r.interpretacao
    r = teste.executar(_df(COM_EMPATES), _params(alternativa="M < M₀"))
    assert r.decisao == NAO_REJEITA_H0
    assert "Não há evidência suficiente de que a mediana de 'x' seja menor que 10." in (
        r.interpretacao
    )


def test_resultado_completo(teste):
    r = teste.executar(_df(SEM_EMPATES), _params())
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Tipo de teste"] == "Uma amostra"
    assert medidas["Método do p-valor"] == "Exato"
    assert medidas["n usado no teste"] == "11"
    for chave in (
        "W⁺ (soma dos postos positivos)",
        "Pseudomediana (Hodges-Lehmann)",
        "Tamanho de efeito r = z/√n",
    ):
        assert chave in medidas
    assert "IC 95% para a pseudomediana" in medidas
    (figura,) = r.figuras
    assert figura.tipo == "histograma"
    refs = [ref["valor"] for ref in figura.dados["referencias"]]
    assert refs == pytest.approx([r.estatisticas["pseudomediana"], 10.0])
    assert r.comparacao is None and teste.comparacao_inicial() is None  # sem card


def test_mesma_validacao_e_formulario_do_sinal(teste):
    assert [s.nome for s in teste.parametros()] == [
        "modo",
        "coluna1",
        "coluna2",
        "m0",
        "alternativa",
        "alfa",
    ]
    assert "Todas as observações são iguais a M₀" in " ".join(
        teste.validar(_df([10, 10]), _params())
    )
    with pytest.raises(ErroValidacao):
        teste.executar(_df(SEM_EMPATES), _params(coluna1="g"))
    assert teste.validar(_df(SEM_EMPATES), _params()) == []
