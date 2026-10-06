"""McNemar: checklist da seção 9 do CLAUDE.md.

Fontes dos valores de referência (independentes do wrapper, que usa scipy):
- exato: b ~ Bin(b + c, 1/2) somado com `math.comb` (tests/referencias.py);
- assintótico com Edwards: χ² = (|b − c| − 1)²/(b + c) e p = erfc(√(χ²/2)) (1 gl); unilaterais
  pela raiz com sinal z = (b − c ∓ 1)/√(b + c) e a normal de `statistics.NormalDist`;
- statsmodels (`contingency_tables.mcnemar`, exact=True / exact=False com correction=True) para
  o bilateral;
- IC exato da odds ratio pareada: Clopper-Pearson por bisseção (tests/referencias.py);
- IC da diferença pareada: Wald com Var = [(b + c) − (b − c)²/n]/n².
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from referencias import binomial_exato, clopper_pearson
from statsmodels.stats.contingency_tables import mcnemar

from core.base import ErroValidacao
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.categoricos import ALTERNATIVAS_MCNEMAR, LIMITE_EXATO, TesteMcNemar

REL = 1e-6


@pytest.fixture
def teste():
    return TesteMcNemar()


def _dados(a, b, c, d, evento="Sim", outro="Não"):
    linhas = (
        [(evento, evento)] * a
        + [(evento, outro)] * b
        + [(outro, evento)] * c
        + [(outro, outro)] * d
    )
    return pd.DataFrame(linhas[::-1], columns=["antes", "depois"])


def _params(**extra):
    return {
        "coluna1": "antes",
        "coluna2": "depois",
        "evento": "Sim",
        "alternativa": "p₁ ≠ p₂",
        "alfa": 0.05,
    } | extra


def _rotulo(alternativa: str) -> str:
    return next(r for r, a in ALTERNATIVAS_MCNEMAR.items() if a == alternativa)


def _assintotico_manual(b, c, alternativa):
    m = b + c
    if alternativa == "two-sided":
        qui2 = max(abs(b - c) - 1, 0) ** 2 / m
        return math.erfc(math.sqrt(qui2 / 2))
    if alternativa == "greater":
        return 1 - NormalDist().cdf((b - c - 1) / math.sqrt(m))
    return NormalDist().cdf((b - c + 1) / math.sqrt(m))


POUCOS = (20, 12, 3, 15)  # b + c = 15 < 25: decisão pelo exato
MUITOS = (40, 30, 14, 36)  # b + c = 44 ≥ 25: decisão pelo assintótico


# ---------------------------------------------------------------------------
# Valores de referência
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tabela", [POUCOS, MUITOS, (5, 6, 6, 5), (3, 1, 7, 2)])
@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_card_exato_e_assintotico_contra_formulas(teste, tabela, alternativa):
    _, b, c, _ = tabela
    r = teste.executar(_dados(*tabela), _params(alternativa=_rotulo(alternativa)))
    linha = list(ALTERNATIVAS_MCNEMAR.values()).index(alternativa)
    p_exato, p_assintotico = r.comparacao.linhas[linha]
    assert p_exato == pytest.approx(binomial_exato(b, b + c, alternativa), rel=REL)
    assert p_assintotico == pytest.approx(_assintotico_manual(b, c, alternativa), rel=REL)
    assert r.estatisticas["p_exato"] == pytest.approx(p_exato, rel=REL)
    assert r.estatisticas["p_assintotico"] == pytest.approx(p_assintotico, rel=REL)


@pytest.mark.parametrize("tabela", [POUCOS, MUITOS])
def test_bilateral_contra_statsmodels(teste, tabela):
    a, b, c, d = tabela
    r = teste.executar(_dados(*tabela), _params())
    exato = mcnemar([[a, b], [c, d]], exact=True)
    assintotico = mcnemar([[a, b], [c, d]], exact=False, correction=True)
    assert r.comparacao.linhas[0] == pytest.approx((exato.pvalue, assintotico.pvalue), rel=REL)
    assert r.estatisticas["qui2"] == pytest.approx(assintotico.statistic, rel=REL)


@pytest.mark.parametrize(("tabela", "usa_exato"), [(POUCOS, True), (MUITOS, False)])
def test_regra_automatica_da_decisao(teste, tabela, usa_exato):
    r = teste.executar(_dados(*tabela), _params())
    _, b, c, _ = tabela
    assert (b + c < LIMITE_EXATO) is usa_exato
    assert r.estatisticas["usou_exato"] == float(usa_exato)
    esperado = r.estatisticas["p_exato"] if usa_exato else r.estatisticas["p_assintotico"]
    assert r.p_valor == esperado
    metodo = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))["Método da decisão"]
    assert metodo.startswith("Exato" if usa_exato else "Qui-quadrado com Edwards")


def test_limite_exato_em_25(teste):
    abaixo = teste.executar(_dados(10, 14, 10, 10), _params())  # b + c = 24
    no_limite = teste.executar(_dados(10, 15, 10, 10), _params())  # b + c = 25
    assert abaixo.estatisticas["usou_exato"] == 1.0
    assert no_limite.estatisticas["usou_exato"] == 0.0


@pytest.mark.parametrize("alfa", [0.01, 0.05, 0.10])
@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_ic_da_diferenca_e_da_odds_ratio(teste, alfa, alternativa):
    a, b, c, d = POUCOS
    n = a + b + c + d
    r = teste.executar(_dados(*POUCOS), _params(alfa=alfa, alternativa=_rotulo(alternativa)))
    e = r.estatisticas
    dif = (b - c) / n
    ep = math.sqrt((b + c) - (b - c) ** 2 / n) / n
    cauda = alfa / 2 if alternativa == "two-sided" else alfa
    z = NormalDist().inv_cdf(1 - cauda)
    esperado_dif = {
        "two-sided": (dif - z * ep, dif + z * ep),
        "greater": (dif - z * ep, 1.0),
        "less": (-1.0, dif + z * ep),
    }[alternativa]
    assert (e["ic_inferior"], e["ic_superior"]) == pytest.approx(esperado_dif, rel=REL)
    theta_l, theta_u = clopper_pearson(b, b + c, cauda, cauda)
    esperado_or = {
        "two-sided": (theta_l / (1 - theta_l), theta_u / (1 - theta_u)),
        "greater": (theta_l / (1 - theta_l), math.inf),
        "less": (0.0, theta_u / (1 - theta_u)),
    }[alternativa]
    for obtido, esperado in zip(
        (e["or_ic_inferior"], e["or_ic_superior"]), esperado_or, strict=True
    ):
        if esperado in (0.0, math.inf):
            assert obtido == esperado
        else:
            assert obtido == pytest.approx(esperado, rel=1e-6)
    assert e["odds_ratio_pareada"] == pytest.approx(b / c)
    assert (e["p1"], e["p2"]) == pytest.approx(((a + b) / n, (a + c) / n))


# ---------------------------------------------------------------------------
# Decisão, interpretação e saída
# ---------------------------------------------------------------------------


def test_rejeita_e_interpretacao(teste):
    r = teste.executar(_dados(*POUCOS), _params())
    assert r.decisao == REJEITA_H0  # exato bilateral ≈ 0,035
    assert "rejeita-se H₀ (p₁ = p₂) em favor de H₁ (p₁ ≠ p₂)" in r.interpretacao
    assert (
        "Há evidência estatística de que a proporção de 'Sim' em 'antes' é diferente da "
        "proporção em 'depois'."
    ) in r.interpretacao


@pytest.mark.parametrize(
    ("rotulo", "decisao", "trecho"),
    [
        ("p₁ > p₂", REJEITA_H0, "é maior que a proporção"),
        ("p₁ < p₂", NAO_REJEITA_H0, "menor que a"),
    ],
)
def test_alternativas_unilaterais(teste, rotulo, decisao, trecho):
    r = teste.executar(_dados(*POUCOS), _params(alternativa=rotulo))
    assert r.decisao == decisao
    assert trecho in r.interpretacao


def test_nao_rejeita_sem_mudanca(teste):
    r = teste.executar(_dados(5, 6, 6, 5), _params())
    assert r.decisao == NAO_REJEITA_H0
    assert "Não há evidência suficiente" in r.interpretacao


def test_evento_inverte_o_sentido(teste):
    a = teste.executar(_dados(*POUCOS), _params())
    b = teste.executar(_dados(*POUCOS), _params(evento="Não"))
    assert b.estatisticas["diferenca"] == pytest.approx(-a.estatisticas["diferenca"])
    assert b.p_valor == pytest.approx(a.p_valor, rel=REL)
    assert (b.estatisticas["b"], b.estatisticas["c"]) == (a.estatisticas["c"], a.estatisticas["b"])


def test_resultado_completo(teste):
    r = teste.executar(_dados(*POUCOS), _params())
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Pares completos (n)"] == "50"
    assert medidas["Pares discordantes b ('Sim' → 'Não')"] == "12"
    assert medidas["Pares discordantes c ('Não' → 'Sim')"] == "3"
    assert medidas["Odds ratio pareada (b/c)"] == "4,0000"
    pares = r.tabelas["Tabela de pares"]
    assert list(pares.columns) == ["antes \\ depois", "Sim", "Não", "Total"]
    assert pares.values.tolist() == [["Sim", 20, 12, 32], ["Não", 3, 15, 18], ["Total", 23, 27, 50]]
    comp = r.comparacao
    assert (comp.titulo_esquerda, comp.titulo_direita) == ("Exato (binomial)", "Qui-quadrado")
    assert comp.hipoteses == ["p₁ ≠ p₂", "p₁ > p₂", "p₁ < p₂"]
    (figura,) = r.figuras
    assert figura.tipo == "barras"
    assert [(c["rotulo"], c["valor"]) for c in figura.dados["categorias"]] == [
        ("antes", pytest.approx(0.64)),
        ("depois", pytest.approx(0.46)),
    ]
    assert any("decisão usa o teste exato" in a for a in r.avisos)
    sem_aviso = teste.executar(_dados(*MUITOS), _params())
    assert not any("decisão usa o teste exato" in a for a in sem_aviso.avisos)


# ---------------------------------------------------------------------------
# Casos de borda e entradas inválidas
# ---------------------------------------------------------------------------


def test_linhas_incompletas_descartadas(teste):
    df = pd.concat(
        [_dados(*POUCOS), pd.DataFrame({"antes": ["Sim", None], "depois": [np.nan, "Não"]})]
    )
    r = teste.executar(df, _params())
    assert r.estatisticas["n"] == 50
    assert "2 linha(s) com valor ausente em 'antes' ou 'depois' foram descartadas." in r.avisos


def test_c_zero(teste):
    r = teste.executar(_dados(10, 6, 0, 10), _params())
    assert math.isinf(r.estatisticas["odds_ratio_pareada"])
    assert math.isinf(r.estatisticas["or_ic_superior"])
    assert any("vai a +∞" in a for a in r.avisos)


@pytest.mark.parametrize(
    ("params", "mensagem"),
    [
        ({"coluna1": None}, "Selecione a medida 1."),
        ({"coluna2": "nao_existe"}, "A coluna 'nao_existe' não existe no arquivo."),
        ({"coluna2": "antes"}, "As duas medidas devem ser colunas diferentes."),
        ({"evento": None}, "Escolha o evento."),
        ({"evento": "Talvez"}, "O valor 'Talvez' não aparece na coluna 'antes'."),
        ({"alternativa": "OR ≠ 1"}, "Escolha uma hipótese alternativa válida."),
        ({"alfa": 0}, "O nível de significância (α) deve estar entre 0 e 1."),
    ],
)
def test_validacao(teste, params, mensagem):
    df = _dados(*POUCOS)
    assert mensagem in teste.validar(df, _params(**params))
    with pytest.raises(ErroValidacao):
        teste.executar(df, _params(**params))


def test_valores_diferentes_entre_medidas(teste):
    df = pd.DataFrame({"antes": ["Sim", "Não", "Sim"], "depois": ["S", "N", "S"]})
    erros = teste.validar(df, _params())
    assert any("devem usar os mesmos 2 valores" in e for e in erros)


def test_sem_pares_discordantes(teste):
    erros = teste.validar(_dados(10, 0, 0, 10), _params())
    assert any("Não há pares discordantes (b + c = 0)" in e for e in erros)


def test_parametros_validos_sem_erros(teste):
    assert teste.validar(_dados(*POUCOS), _params()) == []


def test_formulario(teste):
    specs = teste.parametros()
    assert [(s.nome, s.tipo, s.depende_de) for s in specs] == [
        ("coluna1", "coluna_binaria", None),
        ("coluna2", "coluna_binaria", None),
        ("evento", "nivel", "coluna1"),
        ("alternativa", "opcao", None),
        ("alfa", "alfa", None),
    ]
    assert teste.comparacao_inicial().titulo_direita == "Qui-quadrado"
