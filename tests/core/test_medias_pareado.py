"""Teste t (pareado): checklist da seção 9 do CLAUDE.md.

Fontes dos valores de referência (independentes do wrapper em core/testes/medias.py):
- estatística t, gl e d_z: fórmulas manuais com numpy sobre as diferenças;
- p-valores e IC: statsmodels (`DescrStatsW(d).ttest_mean(0)` / `tconfint_mean`), que é o
  teste t pareado expresso como t de uma amostra das diferenças;
- correlação entre as medidas: fórmula de Pearson com numpy;
- p-valores exatos do Wilcoxon: enumeração das 2^10 atribuições de sinal (tests/referencias.py).
"""

import math

import numpy as np
import pandas as pd
import pytest
from referencias import wilcoxon_exato
from statsmodels.stats.weightstats import DescrStatsW

from core.base import ErroValidacao
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.medias import ALTERNATIVAS_2, TesteTPareado

ANTES = np.array([72.1, 68.4, 75.3, 80.2, 66.9, 71.5, 77.8, 69.2, 74.6, 70.3])
# Diferenças antes − depois com módulos distintos (sem empates nem zeros).
DIFERENCAS = np.array([2.3, -0.8, 3.1, 1.7, 0.4, 2.9, -1.2, 1.1, 3.6, 0.6])
DEPOIS = ANTES - DIFERENCAS
STATSMODELS = {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}
REL = 1e-6


@pytest.fixture
def teste():
    return TesteTPareado()


@pytest.fixture
def df():
    return pd.DataFrame({"antes": ANTES, "depois": DEPOIS, "nome": list("abcdefghij")})


def _params(**extra):
    return {"coluna1": "antes", "coluna2": "depois", "alternativa": "μ₁ ≠ μ₂", "alfa": 0.05} | extra


def _rotulo(alternativa_scipy: str) -> str:
    return next(r for r, a in ALTERNATIVAS_2.items() if a == alternativa_scipy)


def _d():
    return ANTES - DEPOIS


# ---------------------------------------------------------------------------
# Valores de referência
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_t_contra_formula_manual_e_statsmodels(teste, df, alternativa):
    r = teste.executar(df, _params(alternativa=_rotulo(alternativa)))
    d = _d()
    n, media, s = len(d), d.mean(), d.std(ddof=1)
    t_sm, p_sm, gl_sm = DescrStatsW(d).ttest_mean(0, alternative=STATSMODELS[alternativa])
    e = r.estatisticas
    assert e["t"] == pytest.approx(media / (s / math.sqrt(n)), rel=REL)
    assert e["t"] == pytest.approx(t_sm, rel=REL)
    assert e["gl"] == gl_sm == n - 1
    assert r.p_valor == pytest.approx(p_sm, rel=REL)
    assert e["media_diferencas"] == pytest.approx(media, rel=REL)
    assert e["dp_diferencas"] == pytest.approx(s, rel=REL)
    assert e["erro_padrao"] == pytest.approx(s / math.sqrt(n), rel=REL)
    assert e["d_cohen"] == pytest.approx(media / s, rel=REL)
    assert e["media1"] == pytest.approx(ANTES.mean(), rel=REL)
    assert e["media2"] == pytest.approx(DEPOIS.mean(), rel=REL)


@pytest.mark.parametrize("alfa", [0.01, 0.05, 0.10])
@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_ic_contra_statsmodels(teste, df, alfa, alternativa):
    r = teste.executar(df, _params(alternativa=_rotulo(alternativa), alfa=alfa))
    baixo, alto = DescrStatsW(_d()).tconfint_mean(alpha=alfa, alternative=STATSMODELS[alternativa])
    for obtido, esperado in (
        (r.estatisticas["ic_inferior"], baixo),
        (r.estatisticas["ic_superior"], alto),
    ):
        if math.isinf(esperado):
            assert obtido == esperado
        else:
            assert obtido == pytest.approx(esperado, rel=REL)


def test_correlacao_de_pearson(teste, df):
    x, y = ANTES - ANTES.mean(), DEPOIS - DEPOIS.mean()
    r_manual = (x * y).sum() / math.sqrt((x**2).sum() * (y**2).sum())
    assert teste.executar(df, _params()).estatisticas["correlacao"] == pytest.approx(
        r_manual, rel=REL
    )


def test_wilcoxon_exato_por_enumeracao(teste, df):
    r = teste.executar(df, _params())
    esperado = wilcoxon_exato(_d())
    assert [p for _, p in r.comparacao.linhas] == pytest.approx(list(esperado), rel=REL)
    assert r.estatisticas["p_wilcoxon"] == pytest.approx(esperado[0], rel=REL)


def test_comparacao_tres_alternativas(teste, df):
    r = teste.executar(df, _params(alternativa="μ₁ > μ₂"))
    comp = r.comparacao
    assert (comp.titulo_esquerda, comp.titulo_direita) == ("t Student", "Wilcoxon")
    assert comp.hipoteses == ["μ₁ ≠ μ₂", "μ₁ > μ₂", "μ₁ < μ₂"]
    for (p_t, _), alt in zip(comp.linhas, ["two-sided", "larger", "smaller"], strict=True):
        assert p_t == pytest.approx(DescrStatsW(_d()).ttest_mean(0, alternative=alt)[1], rel=REL)
    assert comp.linhas[1][0] == pytest.approx(r.p_valor, rel=REL)


def test_amostra_maior_contra_statsmodels(teste):
    rng = np.random.default_rng(3)
    antes = rng.normal(50, 5, 40)
    depois = antes + rng.normal(1, 2, 40)
    df = pd.DataFrame({"antes": antes, "depois": depois})
    r = teste.executar(df, _params(alternativa="μ₁ < μ₂"))
    _, p_sm, _ = DescrStatsW(antes - depois).ttest_mean(0, alternative="smaller")
    assert r.p_valor == pytest.approx(p_sm, rel=REL)
    assert not any("Amostra pequena" in a for a in r.avisos)


# ---------------------------------------------------------------------------
# Decisão, interpretação e saída
# ---------------------------------------------------------------------------


def test_rejeita_h0_e_interpretacao(teste, df):
    r = teste.executar(df, _params())
    assert r.p_valor < 0.05 and r.decisao == REJEITA_H0
    assert "rejeita-se H₀ (μ₁ = μ₂) em favor de H₁ (μ₁ ≠ μ₂)" in r.interpretacao
    assert (
        "Há evidência estatística de que a média de 'antes' é diferente da média de 'depois' "
        "(diferença média = 1,37)."
    ) in r.interpretacao


@pytest.mark.parametrize(
    ("rotulo", "decisao", "texto"),
    [("μ₁ > μ₂", REJEITA_H0, "é maior que a média"), ("μ₁ < μ₂", NAO_REJEITA_H0, "menor que a")],
)
def test_alternativas_unilaterais(teste, df, rotulo, decisao, texto):
    r = teste.executar(df, _params(alternativa=rotulo))
    assert r.decisao == decisao
    assert texto in r.interpretacao


def test_inverter_colunas_inverte_sinal(teste, df):
    a = teste.executar(df, _params())
    b = teste.executar(df, _params(coluna1="depois", coluna2="antes"))
    assert b.estatisticas["t"] == pytest.approx(-a.estatisticas["t"], rel=REL)
    assert b.p_valor == pytest.approx(a.p_valor, rel=REL)


def test_resultado_completo(teste, df):
    r = teste.executar(df, _params(alfa=0.01))
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Pares completos (n)"] == "10"
    assert medidas["Graus de liberdade"] == "9"
    assert medidas["Média das diferenças (d̄ = x₁ − x₂)"] == "1,3700"
    assert "IC 99% para μ₁ − μ₂" in medidas
    assert "d de Cohen (d_z)" in medidas and "p-valor (Wilcoxon)" in medidas
    (figura,) = r.figuras
    assert figura.tipo == "histograma"
    assert figura.titulo == "Diferenças 'antes' − 'depois'"
    assert sum(figura.dados["contagens"]) == 10
    assert [ref["valor"] for ref in figura.dados["referencias"]] == pytest.approx([1.37, 0.0])
    assert any("Amostra pequena (n = 10 pares)" in a for a in r.avisos)


def test_ic_unilateral_rotulado(teste, df):
    r = teste.executar(df, _params(alternativa="μ₁ < μ₂"))
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["IC 95% para μ₁ − μ₂ (unilateral)"].startswith("[-∞;")


# ---------------------------------------------------------------------------
# Casos de borda e entradas inválidas
# ---------------------------------------------------------------------------


def test_linhas_incompletas_descartadas_com_aviso(teste):
    df = pd.DataFrame(
        {"antes": [*ANTES, np.nan, 70.0, np.nan], "depois": [*DEPOIS, 71.0, np.nan, np.nan]}
    )
    r = teste.executar(df, _params())
    assert r.estatisticas["n"] == 10 and r.estatisticas["n_descartadas"] == 3
    assert "3 linha(s) com valor ausente em 'antes' ou 'depois' foram descartadas." in r.avisos
    _, p_sm, _ = DescrStatsW(_d()).ttest_mean(0)
    assert r.p_valor == pytest.approx(p_sm, rel=REL)


def test_par_com_diferenca_nula_no_wilcoxon(teste):
    df = pd.DataFrame({"antes": [*ANTES, 60.0], "depois": [*DEPOIS, 60.0]})
    r = teste.executar(df, _params())
    assert "No Wilcoxon, 1 par(es) com diferença nula foram descartados." in r.avisos
    assert r.comparacao.linhas[0][1] is not None


def test_correlacao_indefinida_com_medida_constante(teste):
    df = pd.DataFrame({"antes": [5.0, 5.0, 5.0, 5.0], "depois": [1.0, 2.0, 4.0, 3.5]})
    r = teste.executar(df, _params())
    assert math.isnan(r.estatisticas["correlacao"])
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Correlação entre as medidas (r)"] == "—"


@pytest.mark.parametrize(
    ("params", "mensagem"),
    [
        ({"coluna1": None}, "Selecione a medida 1."),
        ({"coluna2": None}, "Selecione a medida 2."),
        ({"coluna2": "nao_existe"}, "A coluna 'nao_existe' não existe no arquivo."),
        ({"coluna1": "nome"}, "A coluna 'nome' não é numérica."),
        ({"coluna2": "antes"}, "As duas medidas devem ser colunas diferentes."),
        ({"alternativa": "μ ≠ μ₀"}, "Escolha uma hipótese alternativa válida."),
        ({"alfa": None}, "O nível de significância (α) deve estar entre 0 e 1."),
    ],
)
def test_validacao_parametros(teste, df, params, mensagem):
    assert mensagem in teste.validar(df, _params(**params))
    with pytest.raises(ErroValidacao):
        teste.executar(df, _params(**params))


@pytest.mark.parametrize(
    ("antes", "depois", "trecho"),
    [
        ([1.0], [2.0], "ao menos 2 pares completos de 'antes' e 'depois' (há 1)"),
        ([1.0, np.nan, 3.0], [2.0, 2.0, np.nan], "(há 1)"),
        ([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], "variância zero"),
        ([1.0, np.inf], [2.0, 3.0], "valores infinitos"),
    ],
)
def test_validacao_dados(teste, antes, depois, trecho):
    erros = teste.validar(pd.DataFrame({"antes": antes, "depois": depois}), _params())
    assert any(trecho in e for e in erros), erros


def test_parametros_validos_sem_erros(teste, df):
    assert teste.validar(df, _params()) == []


def test_formulario(teste):
    specs = teste.parametros()
    assert [(s.nome, s.rotulo, s.tipo) for s in specs] == [
        ("coluna1", "Medida 1 (ex.: antes)", "coluna_numerica"),
        ("coluna2", "Medida 2 (ex.: depois)", "coluna_numerica"),
        ("alternativa", "Hipótese alternativa (H₁)", "opcao"),
        ("alfa", "Nível de significância (α)", "alfa"),
    ]
    assert teste.comparacao_inicial().titulo_direita == "Wilcoxon"
