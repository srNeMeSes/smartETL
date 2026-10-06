"""Teste t (uma amostra): checklist da seção 9 do CLAUDE.md.

Fontes dos valores de referência (independentes do wrapper em core/testes/medias.py):
- estatística t, gl, d de Cohen: fórmulas manuais com numpy;
- p-valores e IC do t: statsmodels (`DescrStatsW.ttest_mean` / `tconfint_mean`);
- p-valores exatos do Wilcoxon: enumeração das 2^n atribuições de sinal aos postos
  (amostra sem empates nem zeros, n = 10 → 1024 combinações).
"""

import math

import numpy as np
import pandas as pd
import pytest
from referencias import wilcoxon_exato
from statsmodels.stats.weightstats import DescrStatsW

from core.base import ErroValidacao, ResultadoTeste
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.medias import ALTERNATIVAS, TesteT1Amostra

# Diferenças em relação a μ₀ = 5 com módulos todos distintos (sem empates nem zeros).
AMOSTRA = [5.12, 4.87, 6.23, 5.81, 6.04, 5.55, 5.36, 6.18, 4.79, 5.93]
MU0 = 5.0
STATSMODELS = {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}
REL = 1e-6


@pytest.fixture
def teste():
    return TesteT1Amostra()


@pytest.fixture
def df():
    return pd.DataFrame({"x": AMOSTRA, "grupo": list("ab" * 5)})


def _params(**extra):
    return {"coluna": "x", "mu0": MU0, "alternativa": "μ ≠ μ₀", "alfa": 0.05} | extra


def _rotulo(alternativa_scipy: str) -> str:
    return next(r for r, a in ALTERNATIVAS.items() if a == alternativa_scipy)


# ---------------------------------------------------------------------------
# Valores de referência
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_t_contra_formula_manual_e_statsmodels(teste, df, alternativa):
    r = teste.executar(df, _params(alternativa=_rotulo(alternativa)))
    x = np.array(AMOSTRA)
    n, media, s = len(x), x.mean(), x.std(ddof=1)
    t_manual = (media - MU0) / (s / math.sqrt(n))
    t_sm, p_sm, gl_sm = DescrStatsW(x).ttest_mean(MU0, alternative=STATSMODELS[alternativa])

    e = r.estatisticas
    assert e["t"] == pytest.approx(t_manual, rel=REL)
    assert e["t"] == pytest.approx(t_sm, rel=REL)
    assert e["gl"] == gl_sm == n - 1
    assert r.p_valor == pytest.approx(p_sm, rel=REL)
    assert e["media"] == pytest.approx(media, rel=REL)
    assert e["desvio_padrao"] == pytest.approx(s, rel=REL)
    assert e["erro_padrao"] == pytest.approx(s / math.sqrt(n), rel=REL)
    assert e["d_cohen"] == pytest.approx((media - MU0) / s, rel=REL)
    assert e["diferenca"] == pytest.approx(media - MU0, rel=REL)


@pytest.mark.parametrize("alfa", [0.01, 0.05, 0.10])
@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_ic_contra_statsmodels(teste, df, alfa, alternativa):
    r = teste.executar(df, _params(alternativa=_rotulo(alternativa), alfa=alfa))
    baixo, alto = DescrStatsW(np.array(AMOSTRA)).tconfint_mean(
        alpha=alfa, alternative=STATSMODELS[alternativa]
    )
    for obtido, esperado in (
        (r.estatisticas["ic_inferior"], baixo),
        (r.estatisticas["ic_superior"], alto),
    ):
        if math.isinf(esperado):
            assert obtido == esperado
        else:
            assert obtido == pytest.approx(esperado, rel=REL)


def test_wilcoxon_exato_por_enumeracao(teste, df):
    r = teste.executar(df, _params())
    esperado = wilcoxon_exato(np.array(AMOSTRA) - MU0)
    obtido = [p_w for _, p_w in r.comparacao.linhas]
    assert obtido == pytest.approx(list(esperado), rel=REL)
    assert r.estatisticas["p_wilcoxon"] == pytest.approx(esperado[0], rel=REL)


def test_comparacao_tem_as_tres_alternativas_na_ordem_do_card(teste, df):
    r = teste.executar(df, _params(alternativa="μ < μ₀"))
    comp = r.comparacao
    assert (comp.titulo_esquerda, comp.titulo_direita) == ("t Student", "Wilcoxon")
    assert comp.hipoteses == ["μ ≠ μ₀", "μ > μ₀", "μ < μ₀"]
    x = np.array(AMOSTRA)
    for (p_t, _), alt in zip(comp.linhas, ["two-sided", "larger", "smaller"], strict=True):
        assert p_t == pytest.approx(DescrStatsW(x).ttest_mean(MU0, alternative=alt)[1], rel=REL)
    # A linha da alternativa escolhida é o p-valor do resultado.
    assert comp.linhas[2][0] == pytest.approx(r.p_valor, rel=REL)


def test_amostra_maior_contra_statsmodels(teste):
    x = np.random.default_rng(7).normal(10, 2, 40)
    df = pd.DataFrame({"x": x})
    for rotulo, alt in zip(ALTERNATIVAS, ["two-sided", "larger", "smaller"], strict=True):
        r = teste.executar(df, _params(mu0=9.5, alternativa=rotulo))
        t_sm, p_sm, _ = DescrStatsW(x).ttest_mean(9.5, alternative=alt)
        assert r.estatisticas["t"] == pytest.approx(t_sm, rel=REL)
        assert r.p_valor == pytest.approx(p_sm, rel=REL)
    assert not any("Amostra pequena" in a for a in r.avisos)  # n = 40 ≥ 30


# ---------------------------------------------------------------------------
# Decisão, interpretação, saída
# ---------------------------------------------------------------------------


def test_rejeita_h0(teste, df):
    r = teste.executar(df, _params())
    assert r.p_valor < 0.01
    assert r.decisao == REJEITA_H0
    assert "rejeita-se H₀ (μ = 5) em favor de H₁ (μ ≠ 5)" in r.interpretacao
    assert "Há evidência estatística de que a média de 'x' é diferente de 5." in r.interpretacao


def test_nao_rejeita_h0(teste, df):
    r = teste.executar(df, _params(mu0=5.6))
    assert r.p_valor > 0.05
    assert r.decisao == NAO_REJEITA_H0
    assert "não se rejeita H₀ (μ = 5,6)" in r.interpretacao
    assert "Não há evidência suficiente de que a média de 'x' seja diferente de 5,6." in (
        r.interpretacao
    )


@pytest.mark.parametrize(
    ("rotulo", "decisao", "texto"),
    [("μ > μ₀", REJEITA_H0, "maior que"), ("μ < μ₀", NAO_REJEITA_H0, "menor que")],
)
def test_alternativas_unilaterais(teste, df, rotulo, decisao, texto):
    r = teste.executar(df, _params(alternativa=rotulo))
    assert r.decisao == decisao
    assert texto in r.interpretacao


def test_alfa_muda_decisao(teste, df):
    # p bilateral ≈ 0,076 com μ₀ = 5,25: rejeita com α = 0,10, não rejeita com 0,05 e 0,01.
    p = teste.executar(df, _params(mu0=5.25)).p_valor
    assert 0.01 < p < 0.10
    assert teste.executar(df, _params(mu0=5.25, alfa=0.10)).decisao == REJEITA_H0
    assert teste.executar(df, _params(mu0=5.25, alfa=0.01)).decisao == NAO_REJEITA_H0
    assert "Com α = 0,10" in teste.executar(df, _params(mu0=5.25, alfa=0.10)).interpretacao


def test_resultado_completo(teste, df):
    r = teste.executar(df, _params())
    assert isinstance(r, ResultadoTeste)
    assert r.teste_id == "teste_t_1am" and r.alfa == 0.05
    resumo = r.tabelas["Resumo"]
    assert list(resumo.columns) == ["Medida", "Valor"]
    medidas = dict(resumo.itertuples(index=False, name=None))
    assert medidas["Observações (n)"] == "10"
    assert medidas["Graus de liberdade"] == "9"
    assert medidas["Média hipotética (μ₀)"] == "5,0000"
    assert medidas["IC 95% para μ"].startswith("[") and ";" in medidas["IC 95% para μ"]
    assert "p-valor (Wilcoxon)" in medidas
    (figura,) = r.figuras
    assert figura.tipo == "histograma"
    assert figura.titulo == "Distribuição de 'x'"
    assert sum(figura.dados["contagens"]) == 10
    assert [ref["valor"] for ref in figura.dados["referencias"]] == pytest.approx(
        [np.mean(AMOSTRA), MU0]
    )
    assert any("Amostra pequena (n = 10)" in a for a in r.avisos)


def test_ic_unilateral_rotulado(teste, df):
    r = teste.executar(df, _params(alternativa="μ > μ₀", alfa=0.01))
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["IC 99% para μ (unilateral)"].endswith("+∞]")


# ---------------------------------------------------------------------------
# Casos de borda e entradas inválidas
# ---------------------------------------------------------------------------


def test_nan_ignorados_com_aviso(teste):
    df = pd.DataFrame({"x": [*AMOSTRA, np.nan, np.nan]})
    r = teste.executar(df, _params())
    assert r.estatisticas["n"] == 10 and r.estatisticas["n_ausentes"] == 2
    assert "2 valor(es) ausente(s) em 'x' foram ignorados." in r.avisos
    t_sm = DescrStatsW(np.array(AMOSTRA)).ttest_mean(MU0)[0]
    assert r.estatisticas["t"] == pytest.approx(t_sm, rel=REL)


def test_zero_no_wilcoxon_gera_aviso(teste):
    df = pd.DataFrame({"x": [*AMOSTRA, MU0]})
    r = teste.executar(df, _params())
    assert any("1 observação(ões) igual(is) a μ₀ foram descartadas" in a for a in r.avisos)
    assert r.comparacao.linhas[0][1] is not None


def test_wilcoxon_sem_diferencas():
    p, avisos = TesteT1Amostra._wilcoxon(np.array([5.0, 5.0]), 5.0)
    assert p == {"two-sided": None, "greater": None, "less": None}
    assert "Wilcoxon não calculado" in avisos[0]


def test_mu0_como_texto_com_virgula(teste, df):
    a = teste.executar(df, _params(mu0="5,3"))
    b = teste.executar(df, _params(mu0=5.3))
    assert a.p_valor == pytest.approx(b.p_valor, rel=REL)


@pytest.mark.parametrize(
    ("params", "mensagem"),
    [
        ({"coluna": None}, "Selecione a variável."),
        ({"coluna": "nao_existe"}, "A coluna 'nao_existe' não existe no arquivo."),
        ({"coluna": "grupo"}, "A coluna 'grupo' não é numérica."),
        ({"mu0": None}, "Informe um número válido para a média hipotética (μ₀)."),
        ({"mu0": "abc"}, "Informe um número válido para a média hipotética (μ₀)."),
        ({"alternativa": "x"}, "Escolha uma hipótese alternativa válida."),
        ({"alfa": 0}, "O nível de significância (α) deve estar entre 0 e 1."),
        ({"alfa": "x"}, "O nível de significância (α) deve estar entre 0 e 1."),
    ],
)
def test_validacao_parametros(teste, df, params, mensagem):
    assert mensagem in teste.validar(df, _params(**params))
    with pytest.raises(ErroValidacao):
        teste.executar(df, _params(**params))


@pytest.mark.parametrize(
    ("valores", "trecho"),
    [
        ([1.0], "ao menos 2 observações válidas em 'x' (há 1)"),
        ([1.0, np.nan, np.nan], "ao menos 2 observações válidas em 'x' (há 1)"),
        ([np.nan, np.nan], "(há 0)"),
        ([3.0, 3.0, 3.0], "variância zero"),
        ([1.0, np.inf, 2.0], "valores infinitos"),
    ],
)
def test_validacao_dados(teste, valores, trecho):
    erros = teste.validar(pd.DataFrame({"x": valores}), _params())
    assert any(trecho in e for e in erros)


def test_validacao_coluna_booleana(teste):
    erros = teste.validar(pd.DataFrame({"x": [True, False, True]}), _params())
    assert "A coluna 'x' não é numérica." in erros


def test_parametros_validos_sem_erros(teste, df):
    assert teste.validar(df, _params()) == []
