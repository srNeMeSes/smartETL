"""Regressão linear — cálculo e pressupostos contra o R (docs/regressao_linear.md, seção 5).

Fonte dos valores de referência: tests/referencias_r/referencias.json, gerado por
tests/referencias_r/gerar_referencias.R com R 4.6.1 (`lm`, `confint`, `predict`), lmtest 0.9.40
(`bptest`, `gqtest`, `hmctest`, `dwtest`, `bgtest`), car 3.1.5 (`vif`) e nortest 1.0-4
(`lillie.test`), sobre datasets públicos (mtcars, Prestige, longley, cars) e grande.csv
(n = 6000, numpy com semente 2026). Os CSV da pasta são exatamente os dados usados no R.
Estatísticas determinísticas: rel = 1e-6 (exigência da seção 9). O p-valor do Harrison-McCabe é
simulado nos dois lados (R com 100 000 réplicas): tolerância de 4 erros padrão de Monte Carlo.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from conftest import RAIZ
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from core import diagnosticos as dg
from core.testes.regressao import (
    INTERCEPTO,
    nivel_referencia_padrao,
    preparar_dados,
    variaveis_colineares,
)

PASTA = RAIZ / "tests" / "referencias_r"
R = json.loads((PASTA / "referencias.json").read_text(encoding="utf-8"))
REL = 1e-6

MODELOS = {
    "mtcars": ("mpg", ["wt", "hp", "cyl", "am"]),
    "prestige": ("prestige", ["education", "income", "type"]),
    "longley": (
        "Employed",
        ["GNP_deflator", "GNP", "Unemployed", "Armed_Forces", "Population", "Year"],
    ),
    "cars": ("dist", ["speed"]),
    "grande": ("y", ["x1", "x2", "g"]),
}


def _dados(nome, referencias=None):
    y, preditores = MODELOS[nome]
    df = pd.read_csv(PASTA / f"{nome}.csv", encoding="utf-8")
    return preparar_dados(df, y, preditores, referencias)


def _ajuste(dados):
    return sm.OLS(dados.y, dados.x).fit()


def _termo_r(coluna):
    """Nome do coeficiente no R: "(Intercept)", "wt", "cyl4 cilindros"."""
    if coluna == INTERCEPTO:
        return "(Intercept)"
    return coluna.replace("[", "").replace("]", "")


def _grupos(dados):
    return [[i - 1 for i in v.colunas] for v in dados.variaveis]


@pytest.fixture(scope="module", params=list(MODELOS))
def caso(request):
    dados = _dados(request.param)
    return request.param, dados, _ajuste(dados), R[request.param]


# ---------------------------------------------------------------------------
# Faixas de classificação (uso interno)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dw", "esperado"),
    [
        (0.0, dg.DW_FORTE_POSITIVA),
        (0.99, dg.DW_FORTE_POSITIVA),
        (1.0, dg.DW_POSITIVA),
        (1.49, dg.DW_POSITIVA),
        (1.50, dg.DW_SEM),
        (2.0, dg.DW_SEM),
        (2.50, dg.DW_SEM),
        (2.51, dg.DW_NEGATIVA),
        (3.0, dg.DW_NEGATIVA),
        (3.01, dg.DW_FORTE_NEGATIVA),
        (4.0, dg.DW_FORTE_NEGATIVA),
    ],
)
def test_faixas_durbin_watson(dw, esperado):
    assert dg.classificar_dw(dw) == esperado


@pytest.mark.parametrize(
    ("vif", "esperado"),
    [
        (1.0, dg.VIF_SEM),
        (1.99, dg.VIF_SEM),
        (2.0, dg.VIF_MODERADA),
        (4.99, dg.VIF_MODERADA),
        (5.00, dg.VIF_PROBLEMATICA),
        (9.99, dg.VIF_PROBLEMATICA),
        (10.0, dg.VIF_GRAVE),
        (19.99, dg.VIF_GRAVE),
        (20.0, dg.VIF_MUITO_GRAVE),
        (1e6, dg.VIF_MUITO_GRAVE),
    ],
)
def test_faixas_vif(vif, esperado):
    assert dg.classificar_vif(vif) == esperado


# ---------------------------------------------------------------------------
# Ajuste (lm) contra o R
# ---------------------------------------------------------------------------


def test_coeficientes_ep_t_p_e_ic(caso):
    _, dados, ajuste, r = caso
    assert [_termo_r(c) for c in dados.colunas] == r["termos"]
    assert dados.y.size == r["n"]
    for nosso, chave in (
        (ajuste.params, "estimativa"),
        (ajuste.bse, "ep"),
        (ajuste.tvalues, "t"),
        (ajuste.pvalues, "p"),
    ):
        assert list(nosso) == pytest.approx(r[chave], rel=REL), chave
    for nivel, sufixo in ((0.05, "95"), (0.10, "90")):
        ic = ajuste.conf_int(alpha=nivel)
        assert list(ic[:, 0]) == pytest.approx(r[f"ic{sufixo}_li"], rel=REL)
        assert list(ic[:, 1]) == pytest.approx(r[f"ic{sufixo}_ls"], rel=REL)


def test_r2_rmse_e_f(caso):
    _, _, ajuste, r = caso
    assert ajuste.rsquared == pytest.approx(r["r2"], rel=REL)
    assert ajuste.rsquared_adj == pytest.approx(r["r2_ajustado"], rel=REL)
    assert math.sqrt(ajuste.mse_resid) == pytest.approx(r["rmse"], rel=REL)
    assert ajuste.fvalue == pytest.approx(r["f"], rel=REL)
    assert (ajuste.df_model, ajuste.df_resid) == (r["gl1"], r["gl2"])
    assert ajuste.f_pvalue == pytest.approx(r["p_f"], rel=REL)


# ---------------------------------------------------------------------------
# Pressupostos contra o R
# ---------------------------------------------------------------------------


def test_breusch_pagan_studentizado(caso):
    _, dados, ajuste, r = caso
    lm, p, _, _ = het_breuschpagan(ajuste.resid, dados.x, robust=True)
    assert (lm, p) == pytest.approx((r["bp"]["estatistica"], r["bp"]["p"]), rel=REL)


def test_goldfeld_quandt_pelos_ajustados_e_por_variavel(caso):
    _, dados, ajuste, r = caso
    gq = dg.goldfeld_quandt(dados.y, dados.x, dg.ordem_estavel(ajuste.fittedvalues))
    ref = r["gq_ajustados"]
    assert (gq.estatistica, gq.p_valor) == pytest.approx((ref["estatistica"], ref["p"]), rel=REL)
    assert (gq.gl1, gq.gl2) == (ref["gl1"], ref["gl2"])
    if "ordenar_por" in r:
        ordem = dg.ordem_estavel(dados.linhas[r["ordenar_por"]])
        gq2 = dg.goldfeld_quandt(dados.y, dados.x, ordem)
        assert (gq2.estatistica, gq2.p_valor) == pytest.approx(
            (r["gq_variavel"]["estatistica"], r["gq_variavel"]["p"]), rel=REL
        )


def test_goldfeld_quandt_metade_degenerada():
    # mtcars ordenado pelos ajustados: na metade inferior todos os carros são de 8 cilindros.
    dados = _dados("mtcars")
    gq = dg.goldfeld_quandt(dados.y, dados.x, dg.ordem_estavel(_ajuste(dados).fittedvalues))
    assert gq.degenerado
    dados = _dados("cars")
    assert not dg.goldfeld_quandt(dados.y, dados.x, dg.ordem_estavel(dados.y)).degenerado


def test_harrison_mccabe(caso):
    _, dados, ajuste, r = caso
    ordem = dg.ordem_estavel(ajuste.fittedvalues)
    k = dados.x.shape[1]
    hmc = dg.harrison_mccabe(ajuste.resid, ordem, k, simulacoes=100_000, semente=7)
    assert hmc.estatistica == pytest.approx(r["hmc_ajustados"]["estatistica"], rel=REL)
    p_r = r["hmc_ajustados"]["p_100000"]
    erro = math.sqrt(2 * max(p_r * (1 - p_r), 1e-4) / 100_000)
    assert abs(hmc.p_valor - p_r) <= 4 * erro
    if "ordenar_por" in r:
        ordem = dg.ordem_estavel(dados.linhas[r["ordenar_por"]])
        assert dg.harrison_mccabe(ajuste.resid, ordem, k, simulacoes=10).estatistica == (
            pytest.approx(r["hmc_variavel"]["estatistica"], rel=REL)
        )


def test_harrison_mccabe_reprodutivel_e_limites():
    dados = _dados("cars")
    ajuste = _ajuste(dados)
    ordem = dg.ordem_estavel(ajuste.fittedvalues)
    a = dg.harrison_mccabe(ajuste.resid, ordem, 2)
    assert a == dg.harrison_mccabe(ajuste.resid, ordem, 2)  # semente fixa
    assert a.simulacoes == 1000 and 0 <= a.p_valor <= 1
    assert dg.harrison_mccabe(np.ones(5), np.arange(5), 3) is None  # ⌊n/2⌋ < k


def test_durbin_watson_e_breusch_godfrey(caso):
    _, _, ajuste, r = caso
    assert durbin_watson(ajuste.resid) == pytest.approx(r["dw"], rel=REL)
    bg = acorr_breusch_godfrey(ajuste, nlags=1, result_object=True)
    lm, p = bg.lm, bg.lmpval
    assert (lm, p) == pytest.approx((r["bg"]["estatistica"], r["bg"]["p"]), rel=REL)


def test_gvif_como_car_vif(caso):
    _, dados, ajuste, r = caso
    if r["vif"] == "nenhum":  # um único preditor: car::vif recusa
        assert len(dados.variaveis) == 1
        return
    valores = dg.gvif(np.asarray(ajuste.cov_params())[1:, 1:], _grupos(dados))
    assert [v.nome for v in dados.variaveis] == r["vif"]["variaveis"]
    assert valores == pytest.approx(r["vif"]["gvif"], rel=REL)
    gl = [len(v.colunas) for v in dados.variaveis]
    assert gl == r["vif"]["gl"]
    ajustado = [g ** (1 / (2 * d)) for g, d in zip(valores, gl, strict=True)]
    assert ajustado == pytest.approx(r["vif"]["gvif_ajustado"], rel=REL)


def test_normalidade_dos_residuos(caso):
    nome, _, ajuste, r = caso
    teste, estatistica, p = dg.normalidade(np.asarray(ajuste.resid))
    esperado = "Kolmogorov-Smirnov (Lilliefors)" if nome == "grande" else "Shapiro-Wilk"
    assert teste == esperado and r["normalidade"]["teste"] in esperado.lower()
    assert (estatistica, p) == pytest.approx(
        (r["normalidade"]["estatistica"], r["normalidade"]["p"]), rel=REL
    )


@pytest.mark.parametrize("n", [20, 150])
def test_lilliefors_estatistica_igual_ao_statsmodels(n):
    from statsmodels.stats.diagnostic import lilliefors as lf_sm

    x = np.random.default_rng(n).gamma(2.0, size=n)
    assert dg.lilliefors(x)[0] == pytest.approx(lf_sm(x, pvalmethod="approx")[0], rel=1e-12)


# ---------------------------------------------------------------------------
# Montagem do modelo
# ---------------------------------------------------------------------------


def test_referencia_padrao_e_a_mais_frequente():
    df = pd.read_csv(PASTA / "mtcars.csv", encoding="utf-8")
    assert nivel_referencia_padrao(df["cyl"]) == "8 cilindros"  # 14 de 32
    assert nivel_referencia_padrao(pd.Series(["b", "a", "b", "a", "c"])) == "a"  # empate: ordem
    dados = _dados("mtcars")
    cyl = dados.variaveis[2]
    assert (cyl.categorica, cyl.referencia, cyl.niveis) == (
        True,
        "8 cilindros",
        ["4 cilindros", "6 cilindros", "8 cilindros"],
    )
    assert cyl.contagens == {"4 cilindros": 11, "6 cilindros": 7, "8 cilindros": 14}


def test_troca_do_nivel_de_referencia():
    padrao = _ajuste(_dados("prestige"))
    dados = _dados("prestige", {"type": "prof"})
    trocado = _ajuste(dados)
    r = R["prestige_ref_prof"]
    assert [_termo_r(c) for c in dados.colunas] == r["termos"]
    assert list(trocado.params) == pytest.approx(r["estimativa"], rel=REL)
    assert trocado.rsquared == pytest.approx(padrao.rsquared, rel=1e-12)
    # consistência: bc − prof no novo modelo = −(prof − bc) no padrão (ref. bc)
    assert trocado.params[3] == pytest.approx(-padrao.params[3], rel=1e-9)
    assert trocado.params[4] == pytest.approx(padrao.params[4] - padrao.params[3], rel=1e-9)


def test_linhas_incompletas_descartadas():
    dados = _dados("prestige")
    assert (dados.descartadas, len(dados.y)) == (4, 98)


def test_colinearidade_perfeita_identifica_as_variaveis():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=30), rng.normal(size=30)
    df = pd.DataFrame(
        {"y": rng.normal(size=30), "a": a, "b": b, "c": 2 * a - b, "d": rng.normal(size=30)}
    )
    assert variaveis_colineares(preparar_dados(df, "y", ["a", "b", "d"])) == []
    assert variaveis_colineares(preparar_dados(df, "y", ["a", "b", "c", "d"])) == [["a", "b", "c"]]
    df["k"] = 3.0
    assert variaveis_colineares(preparar_dados(df, "y", ["a", "k"])) == [[INTERCEPTO, "k"]]
