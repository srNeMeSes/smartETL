"""Teste do sinal (uma amostra e pareado): checklist da seção 9 do CLAUDE.md.

Fontes dos valores de referência (independentes do wrapper, que usa scipy):
- p exato: nº de sinais + ~ Bin(n, 1/2) somado com `math.comb` (tests/referencias.py) e
  `statsmodels.stats.descriptivestats.sign_test` (bilateral);
- IC exato da mediana por estatísticas de ordem: k = maior j com P(B ≤ j − 1) ≤ cauda,
  B ~ Bin(N, 1/2), com a função de distribuição calculada por `math.comb`.
"""

import math

import numpy as np
import pandas as pd
import pytest
from referencias import binomial_exato
from statsmodels.stats.descriptivestats import sign_test

from core.base import ErroValidacao
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.nao_parametricos import (
    ALTERNATIVAS_SINAL,
    PAREADO,
    UMA_AMOSTRA,
    TesteSinal,
    ic_mediana_exato,
)

REL = 1e-9
# Tempos de atendimento (min): 10 acima de 10, 3 abaixo e 2 iguais a M₀ = 10.
TEMPOS = [12, 8, 15, 10, 22, 9, 18, 10, 14, 25, 11, 16, 7, 19, 13]


@pytest.fixture
def teste():
    return TesteSinal()


@pytest.fixture
def df():
    return pd.DataFrame({"tempo": TEMPOS, "setor": list("abcabcabcabcabc")})


def _params(**extra):
    return {
        "modo": UMA_AMOSTRA,
        "coluna1": "tempo",
        "coluna2": None,
        "m0": 10,
        "alternativa": "M ≠ M₀",
        "alfa": 0.05,
    } | extra


def _rotulo(alternativa: str) -> str:
    return next(r for r, a in ALTERNATIVAS_SINAL.items() if a == alternativa)


def _cdf(k, n):
    return sum(math.comb(n, i) for i in range(k + 1)) / 2**n


def _ic_manual(valores, alfa, alternativa):
    x = sorted(valores)
    n = len(x)
    cauda = alfa / 2 if alternativa == "two-sided" else alfa
    k = max((j for j in range(1, n + 1) if _cdf(j - 1, n) <= cauda), default=0)
    if k == 0:
        return -math.inf, math.inf
    return {
        "two-sided": (x[k - 1], x[n - k]),
        "greater": (x[k - 1], math.inf),
        "less": (-math.inf, x[n - k]),
    }[alternativa]


# ---------------------------------------------------------------------------
# Valores de referência
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("m0", [10, 12, 20])
def test_p_exato_contra_binomial_manual(teste, df, alternativa, m0):
    r = teste.executar(df, _params(m0=m0, alternativa=_rotulo(alternativa)))
    x = np.array(TEMPOS)
    pos, neg = int((x > m0).sum()), int((x < m0).sum())
    assert (r.estatisticas["positivos"], r.estatisticas["negativos"]) == (pos, neg)
    assert r.p_valor == pytest.approx(binomial_exato(pos, pos + neg, alternativa), rel=REL)


@pytest.mark.parametrize("m0", [10, 12, 20])
def test_bilateral_contra_statsmodels(teste, df, m0):
    r = teste.executar(df, _params(m0=m0))
    _, p_sm = sign_test(TEMPOS, m0)
    assert r.p_valor == pytest.approx(p_sm, rel=REL)


@pytest.mark.parametrize("alfa", [0.01, 0.05, 0.10])
@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_ic_mediana_por_estatisticas_de_ordem(teste, df, alfa, alternativa):
    r = teste.executar(df, _params(alfa=alfa, alternativa=_rotulo(alternativa)))
    esperado = _ic_manual(TEMPOS, alfa, alternativa)
    assert (r.estatisticas["ic_inferior"], r.estatisticas["ic_superior"]) == esperado
    assert r.estatisticas["confianca_obtida"] >= 1 - alfa
    assert r.estatisticas["mediana"] == 13


def test_ic_amostra_muito_pequena():
    baixo, alto, confianca = ic_mediana_exato(np.array([1.0, 2.0, 3.0]), 0.05, "two-sided")
    assert (baixo, alto, confianca) == (-math.inf, math.inf, 1.0)


# ---------------------------------------------------------------------------
# Empates, modo pareado, decisão e saída
# ---------------------------------------------------------------------------


def test_empates_com_m0_descartados(teste, df):
    r = teste.executar(df, _params())
    e = r.estatisticas
    assert (e["empates"], e["n"], e["n_validos"]) == (2, 13, 15)
    assert (
        "2 observação(ões) igual(is) a M₀ = 10 (sem sinal) foram descartadas; o teste usa n = 13."
        in (r.avisos)
    )


def test_pareado_testa_a_mediana_das_diferencas(teste):
    antes = [140, 152, 138, 160, 145, 149, 155, 139, 147, 158]
    depois = [132, 141, 139, 149, 145, 141, 147, 134, 146, 150]
    df = pd.DataFrame({"antes": antes, "depois": depois})
    r = teste.executar(df, _params(modo=PAREADO, coluna1="antes", coluna2="depois", m0=0))
    d = np.array(antes) - np.array(depois)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    assert (r.estatisticas["positivos"], r.estatisticas["negativos"]) == (pos, neg) == (8, 1)
    assert r.estatisticas["empates"] == 1  # 145 → 145
    assert r.p_valor == pytest.approx(binomial_exato(pos, pos + neg, "two-sided"), rel=REL)
    assert r.estatisticas["mediana"] == pytest.approx(float(np.median(d)))
    (figura,) = r.figuras
    assert figura.titulo == "Diferenças 'antes' − 'depois'"
    assert "a mediana das diferenças 'antes' − 'depois'" in r.interpretacao


def test_pareado_descarta_linhas_incompletas(teste):
    df = pd.DataFrame({"a": [5.0, 6.0, np.nan, 8.0, 9.0], "b": [4.0, np.nan, 3.0, 7.0, 7.5]})
    r = teste.executar(df, _params(modo=PAREADO, coluna1="a", coluna2="b", m0=0))
    assert r.estatisticas["n_validos"] == 3
    assert "2 linha(s) com valor ausente em 'a' ou 'b' foram descartadas." in r.avisos


def test_uma_amostra_ignora_ausentes(teste):
    df = pd.DataFrame({"tempo": [*TEMPOS, np.nan]})
    r = teste.executar(df, _params())
    assert r.estatisticas["n_validos"] == 15
    assert "1 linha(s) com valor ausente em 'tempo' foram descartadas." in r.avisos


def test_decisao_e_interpretacao(teste, df):
    r = teste.executar(df, _params())
    assert r.decisao == NAO_REJEITA_H0  # p ≈ 0,092
    assert "não se rejeita H₀ (M = 10)" in r.interpretacao
    assert "Não há evidência suficiente de que a mediana de 'tempo' seja diferente de 10." in (
        r.interpretacao
    )
    r = teste.executar(df, _params(alternativa="M > M₀"))
    assert r.decisao == REJEITA_H0  # p ≈ 0,046
    assert "rejeita-se H₀ (M = 10) em favor de H₁ (M > 10)" in r.interpretacao
    assert "Há evidência estatística de que a mediana de 'tempo' é maior que 10." in (
        r.interpretacao
    )
    assert teste.executar(df, _params(alternativa="M > M₀", alfa=0.01)).decisao == NAO_REJEITA_H0


def test_resultado_completo(teste, df):
    r = teste.executar(df, _params(m0="10,0"))
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Tipo de teste"] == "Uma amostra"
    assert medidas["Sinais + (acima de M₀)"] == "10"
    assert medidas["Sinais − (abaixo de M₀)"] == "3"
    assert medidas["n usado no teste (+ e −)"] == "13"
    assert medidas["Mediana amostral"] == "13,0000"
    assert medidas["IC 95% exato para a mediana"] == "[10,0000; 18,0000]"
    assert medidas["Confiança obtida do IC"] == "96,5%"
    (figura,) = r.figuras
    assert figura.tipo == "histograma"
    assert [ref["valor"] for ref in figura.dados["referencias"]] == [13.0, 10.0]
    assert r.comparacao is None and teste.comparacao_inicial() is None  # sem card


# ---------------------------------------------------------------------------
# Entradas inválidas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "mensagem"),
    [
        ({"modo": "x"}, "Escolha o tipo de teste."),
        ({"coluna1": None}, "Selecione a variável."),
        ({"coluna1": "setor"}, "A coluna 'setor' não é numérica."),
        ({"m0": "abc"}, "Informe um número válido para a mediana hipotética (M₀)."),
        ({"modo": PAREADO, "coluna2": None}, "Selecione a medida 2."),
        ({"modo": PAREADO, "coluna1": None, "coluna2": "tempo"}, "Selecione a medida 1."),
        ({"modo": PAREADO, "coluna2": "tempo"}, "As duas medidas devem ser colunas diferentes."),
        ({"alternativa": "μ ≠ μ₀"}, "Escolha uma hipótese alternativa válida."),
        ({"alfa": 5}, "O nível de significância (α) deve estar entre 0 e 1."),
    ],
)
def test_validacao(teste, df, params, mensagem):
    assert mensagem in teste.validar(df, _params(**params))
    with pytest.raises(ErroValidacao):
        teste.executar(df, _params(**params))


def test_todas_iguais_a_m0(teste):
    erros = teste.validar(pd.DataFrame({"tempo": [10, 10, 10]}), _params())
    assert any("Todas as observações são iguais a M₀" in e for e in erros)


def test_sem_observacoes(teste):
    erros = teste.validar(pd.DataFrame({"tempo": [np.nan, np.nan]}), _params())
    assert "Não há observações válidas para o teste." in erros


def test_parametros_validos_sem_erros(teste, df):
    assert teste.validar(df, _params()) == []


def test_formulario(teste):
    specs = teste.parametros()
    assert [(s.nome, s.tipo) for s in specs] == [
        ("modo", "opcao"),
        ("coluna1", "coluna_numerica"),
        ("coluna2", "coluna_numerica"),
        ("m0", "numero"),
        ("alternativa", "opcao"),
        ("alfa", "alfa"),
    ]
    assert specs[0].opcoes == [UMA_AMOSTRA, PAREADO] and specs[0].padrao == UMA_AMOSTRA
    assert specs[2].obrigatorio is False
    assert specs[3].padrao == 0
