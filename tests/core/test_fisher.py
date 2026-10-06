"""Teste exato de Fisher: checklist da seção 9 do CLAUDE.md.

Fontes dos valores de referência (independentes do wrapper, que usa scipy), em
tests/referencias.py, só com a biblioteca padrão:
- p-valores exatos pela hipergeométrica com `math.comb` (bilateral: probabilidades ≤ à
  observada, como o `fisher.test` do R);
- odds ratio condicional (EMV): ψ tal que E[a | ψ] = a, pela hipergeométrica não central e
  bisseção;
- IC exato condicional: inversão das caudas P(X ≥ a; ψ) e P(X ≤ a; ψ) (método do R);
- odds ratio amostral = ad/bc.
"""

import math

import numpy as np
import pandas as pd
import pytest
from referencias import fisher_exato, ic_exato_odds_ratio, odds_ratio_condicional

from core.base import ErroValidacao
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.categoricos import ALTERNATIVAS_OR, TesteFisher

REL = 1e-6


@pytest.fixture
def teste():
    return TesteFisher()


def _dados(a, b, c, d, nomes=("Fumante", "Não fumante"), resultado=("Doente", "Saudável")):
    linhas = (
        [(nomes[0], resultado[0])] * a
        + [(nomes[0], resultado[1])] * b
        + [(nomes[1], resultado[0])] * c
        + [(nomes[1], resultado[1])] * d
    )
    return pd.DataFrame(linhas[::-1], columns=["habito", "condicao"])


def _params(**extra):
    return {
        "coluna1": "habito",
        "evento1": "Fumante",
        "coluna2": "condicao",
        "evento2": "Doente",
        "alternativa": "OR ≠ 1",
        "alfa": 0.05,
    } | extra


def _rotulo(alternativa: str) -> str:
    return next(r for r, a in ALTERNATIVAS_OR.items() if a == alternativa)


TABELAS = [(12, 5, 4, 11), (3, 1, 1, 3), (8, 2, 1, 5), (20, 15, 18, 17)]


# ---------------------------------------------------------------------------
# Valores de referência
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tabela", TABELAS)
@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_p_exato_contra_hipergeometrica(teste, tabela, alternativa):
    r = teste.executar(_dados(*tabela), _params(alternativa=_rotulo(alternativa)))
    bilateral, maior, menor = fisher_exato(*tabela)
    esperado = {"two-sided": bilateral, "greater": maior, "less": menor}[alternativa]
    assert r.p_valor == pytest.approx(esperado, rel=REL)


@pytest.mark.parametrize("tabela", TABELAS)
def test_odds_ratio_amostral_e_condicional(teste, tabela):
    a, b, c, d = tabela
    e = teste.executar(_dados(*tabela), _params()).estatisticas
    assert e["odds_ratio_amostral"] == pytest.approx(a * d / (b * c), rel=REL)
    assert e["odds_ratio_condicional"] == pytest.approx(odds_ratio_condicional(*tabela), rel=REL)
    assert (e["a"], e["b"], e["c"], e["d"]) == tabela


@pytest.mark.parametrize("alfa", [0.01, 0.05, 0.10])
@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_ic_exato_condicional(teste, alfa, alternativa):
    tabela = (12, 5, 4, 11)
    r = teste.executar(_dados(*tabela), _params(alternativa=_rotulo(alternativa), alfa=alfa))
    baixo, alto = ic_exato_odds_ratio(*tabela, 1 - alfa, alternativa)
    for obtido, esperado in (
        (r.estatisticas["ic_inferior"], baixo),
        (r.estatisticas["ic_superior"], alto),
    ):
        if esperado in (0.0, math.inf):
            assert obtido == esperado
        else:
            assert obtido == pytest.approx(esperado, rel=1e-5)


# ---------------------------------------------------------------------------
# Orientação da tabela, decisão, interpretação e saída
# ---------------------------------------------------------------------------


def test_evento_define_a_orientacao_da_tabela(teste):
    df = _dados(12, 5, 4, 11)
    a = teste.executar(df, _params())
    b = teste.executar(df, _params(evento1="Não fumante"))  # troca as linhas
    assert b.estatisticas["odds_ratio_amostral"] == pytest.approx(
        1 / a.estatisticas["odds_ratio_amostral"], rel=REL
    )
    assert b.p_valor == pytest.approx(a.p_valor, rel=REL)  # bilateral não muda
    c = teste.executar(df, _params(alternativa="OR > 1"))
    d = teste.executar(df, _params(evento1="Não fumante", alternativa="OR < 1"))
    assert c.p_valor == pytest.approx(d.p_valor, rel=REL)


def test_tabela_2x2_com_totais(teste):
    r = teste.executar(_dados(12, 5, 4, 11), _params())
    tabela = r.tabelas["Tabela 2×2"]
    assert list(tabela.columns) == ["habito \\ condicao", "Doente", "Saudável", "Total"]
    assert tabela.values.tolist() == [
        ["Fumante", 12, 5, 17],
        ["Não fumante", 4, 11, 15],
        ["Total", 16, 16, 32],
    ]


def test_rejeita_bilateral_e_interpretacao(teste):
    r = teste.executar(_dados(12, 5, 4, 11), _params())
    assert r.p_valor < 0.05 and r.decisao == REJEITA_H0
    assert "rejeita-se H₀ (OR = 1, as variáveis são independentes) em favor de H₁ (OR ≠ 1)" in (
        r.interpretacao
    )
    assert (
        "Há evidência estatística de associação entre 'habito' = 'Fumante' e 'condicao' = 'Doente'."
    ) in r.interpretacao


@pytest.mark.parametrize(
    ("rotulo", "decisao", "trecho"),
    [
        ("OR > 1", REJEITA_H0, "'habito' = 'Fumante' aumenta a chance de 'condicao' = 'Doente'"),
        ("OR < 1", NAO_REJEITA_H0, "'habito' = 'Fumante' diminui a chance de"),
    ],
)
def test_alternativas_unilaterais(teste, rotulo, decisao, trecho):
    r = teste.executar(_dados(12, 5, 4, 11), _params(alternativa=rotulo))
    assert r.decisao == decisao
    assert trecho in r.interpretacao


def test_nao_rejeita_sem_associacao(teste):
    r = teste.executar(_dados(20, 15, 18, 17), _params())
    assert r.decisao == NAO_REJEITA_H0
    assert "Não há evidência suficiente de associação" in r.interpretacao


def test_resultado_completo(teste):
    r = teste.executar(_dados(12, 5, 4, 11), _params(alfa=0.10, alternativa="OR > 1"))
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Observações (n)"] == "32"
    assert medidas["Evento da variável 1 ('habito')"] == "'Fumante'"
    assert medidas["Odds ratio amostral (ad/bc)"] == "6,6000"
    assert "Odds ratio condicional (EMV)" in medidas
    assert medidas["IC 90% exato para a odds ratio (unilateral)"].endswith("+∞]")
    (figura,) = r.figuras
    assert figura.tipo == "barras_agrupadas"
    assert figura.dados["series"] == ["Doente", "Saudável"]
    assert [g["rotulo"] for g in figura.dados["grupos"]] == ["Fumante", "Não fumante"]
    assert figura.dados["grupos"][0]["valores"] == pytest.approx([12 / 17, 5 / 17])
    assert r.comparacao is None and teste.comparacao_inicial() is None  # sem card
    assert r.avisos == []


# ---------------------------------------------------------------------------
# Casos de borda e entradas inválidas
# ---------------------------------------------------------------------------


def test_celula_zero(teste):
    r = teste.executar(_dados(5, 0, 3, 4), _params())
    e = r.estatisticas
    assert math.isinf(e["odds_ratio_amostral"]) and math.isinf(e["odds_ratio_condicional"])
    assert math.isinf(e["ic_superior"])
    assert r.p_valor == pytest.approx(fisher_exato(5, 0, 3, 4)[0], rel=REL)
    assert any("célula com zero" in a for a in r.avisos)
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Odds ratio amostral (ad/bc)"] == "+∞"


def test_linhas_incompletas_descartadas(teste):
    df = pd.concat(
        [
            _dados(12, 5, 4, 11),
            pd.DataFrame({"habito": [None, "Fumante"], "condicao": ["Doente", np.nan]}),
        ]
    )
    r = teste.executar(df, _params())
    assert r.estatisticas["n"] == 32
    assert "2 linha(s) com valor ausente em 'habito' ou 'condicao' foram descartadas." in r.avisos


@pytest.mark.parametrize(
    ("params", "mensagem"),
    [
        ({"coluna1": None}, "Selecione a variável 1."),
        ({"coluna2": None}, "Selecione a variável 2."),
        ({"coluna2": "nao_existe"}, "A coluna 'nao_existe' não existe no arquivo."),
        ({"evento1": None}, "Escolha o evento da variável 1."),
        ({"evento2": "Talvez"}, "O valor 'Talvez' não aparece na coluna 'condicao'."),
        (
            {"coluna2": "habito", "evento2": "Fumante"},
            "As duas variáveis devem ser colunas diferentes.",
        ),
        ({"alternativa": "p ≠ p₀"}, "Escolha uma hipótese alternativa válida."),
        ({"alfa": 1}, "O nível de significância (α) deve estar entre 0 e 1."),
    ],
)
def test_validacao(teste, params, mensagem):
    df = _dados(12, 5, 4, 11)
    assert mensagem in teste.validar(df, _params(**params))
    with pytest.raises(ErroValidacao):
        teste.executar(df, _params(**params))


def test_coluna_sem_2_valores(teste):
    df = pd.DataFrame({"habito": ["a", "b", "c", "a"], "condicao": ["x", "y", "x", "y"]})
    assert "A coluna 'habito' deve ter exatamente 2 valores distintos (tem 3)." in teste.validar(
        df, _params(evento1="a", evento2="x")
    )


def test_linhas_completas_sem_os_2_valores(teste):
    df = pd.DataFrame(
        {"habito": ["Fumante", "Fumante", "Não fumante"], "condicao": ["Doente", "Saudável", None]}
    )
    erros = teste.validar(df, _params())
    assert "Nas linhas completas, a coluna 'habito' deve ter os 2 valores (tem 1)." in erros


def test_parametros_validos_sem_erros(teste):
    assert teste.validar(_dados(12, 5, 4, 11), _params()) == []


def test_formulario(teste):
    specs = teste.parametros()
    assert [(s.nome, s.tipo, s.depende_de) for s in specs] == [
        ("coluna1", "coluna_binaria", None),
        ("evento1", "nivel", "coluna1"),
        ("coluna2", "coluna_binaria", None),
        ("evento2", "nivel", "coluna2"),
        ("alternativa", "opcao", None),
        ("alfa", "alfa", None),
    ]
    assert specs[4].opcoes == ["OR ≠ 1", "OR > 1", "OR < 1"]
