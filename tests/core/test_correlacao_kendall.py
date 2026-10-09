"""Correlação de Kendall (τ-b): checklist da seção 9 do CLAUDE.md.

Fonte dos valores de referência: tests/referencias_r/referencias_kendall.json, gerado por
tests/referencias_r/gerar_referencias_kendall.R com R 4.6.1 (`cor.test(method = "kendall")`)
sobre "distintos" (n = 20 sem empates: p exato; o R devolve T = pares concordantes), mtcars e
cars (com empates: z com correção de empates). Também: pares concordantes e discordantes por
enumeração de todos os pares; IC pela z de Fisher com var = 0,437/(n − 4), à mão.
"""

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
from conftest import RAIZ
from scipy import stats

from core.base import ErroValidacao
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.correlacao import ALTERNATIVAS_COR
from core.testes.correlacao_kendall import AVISO_IC_N4, TesteCorrelacaoKendall

PASTA = RAIZ / "tests" / "referencias_r"
R = json.loads((PASTA / "referencias_kendall.json").read_text(encoding="utf-8"))
ALT_R = {"two-sided": "two.sided", "greater": "greater", "less": "less"}
BASES = ["distintos", "mtcars", "cars"]


def _df(nome):
    return pd.read_csv(PASTA / f"kendall_{nome}.csv")


def _params(alternativa="ρ ≠ 0", **extra):
    return {"x": "x", "y": "y", "alternativa": alternativa, "alfa": 0.05} | extra


@pytest.mark.parametrize("nome", BASES)
@pytest.mark.parametrize("rotulo", list(ALTERNATIVAS_COR))
def test_contra_cor_test(nome, rotulo):
    ref = R[nome][ALT_R[ALTERNATIVAS_COR[rotulo]]]
    r = TesteCorrelacaoKendall().executar(_df(nome), _params(rotulo))
    assert r.estatisticas["tau"] == pytest.approx(ref["tau"], rel=1e-12)
    assert r.p_valor == pytest.approx(ref["p"], rel=1e-9)
    if ref["nome"] == "z":  # com empates: a mesma estatística z do R
        assert r.estatisticas["z"] == pytest.approx(ref["estatistica"], rel=1e-10)
        assert r.estatisticas["usou_exato"] == 0.0
    else:  # exato: o R devolve T = pares concordantes
        assert r.estatisticas["concordantes"] == ref["estatistica"]
        assert r.estatisticas["usou_exato"] == 1.0


@pytest.mark.parametrize("nome", BASES)
def test_pares_concordantes_e_discordantes_por_enumeracao(nome):
    df = _df(nome)
    x, y = df.x.to_numpy(), df.y.to_numpy()
    produtos = [(x[i] - x[j]) * (y[i] - y[j]) for i, j in itertools.combinations(range(len(x)), 2)]
    r = TesteCorrelacaoKendall().executar(df, _params())
    assert r.estatisticas["concordantes"] == sum(p > 0 for p in produtos)
    assert r.estatisticas["discordantes"] == sum(p < 0 for p in produtos)


@pytest.mark.parametrize("rotulo", list(ALTERNATIVAS_COR))
def test_ic_fieller(rotulo):
    df = _df("cars")
    alternativa = ALTERNATIVAS_COR[rotulo]
    r = TesteCorrelacaoKendall().executar(df, _params(rotulo, alfa=0.10))
    tau, ep = r.estatisticas["tau"], math.sqrt(0.437 / (50 - 4))
    z = math.atanh(tau)
    if alternativa == "two-sided":
        c = stats.norm.ppf(0.95)
        esperado = (math.tanh(z - c * ep), math.tanh(z + c * ep))
    elif alternativa == "greater":
        esperado = (math.tanh(z - stats.norm.ppf(0.90) * ep), 1.0)
    else:
        esperado = (-1.0, math.tanh(z + stats.norm.ppf(0.90) * ep))
    assert (r.estatisticas["ic_inferior"], r.estatisticas["ic_superior"]) == pytest.approx(esperado)
    assert any(m.startswith("IC 90% para τ") for m in r.tabelas["Resumo"]["Medida"])


def test_card_kendall_x_spearman():
    df = _df("mtcars")
    teste = TesteCorrelacaoKendall()
    inicial = teste.comparacao_inicial()
    assert (inicial.titulo_esquerda, inicial.titulo_direita) == ("Kendall", "Spearman")
    r = teste.executar(df, _params())
    for (p_kendall, p_spearman), alternativa in zip(
        r.comparacao.linhas, ALTERNATIVAS_COR.values(), strict=True
    ):
        assert p_kendall == pytest.approx(R["mtcars"][ALT_R[alternativa]]["p"], rel=1e-9)
        esperado = stats.spearmanr(df.x, df.y, alternative=alternativa).pvalue
        assert p_spearman == pytest.approx(esperado)


def test_decisao_interpretacao_e_avisos():
    r = TesteCorrelacaoKendall().executar(_df("mtcars"), _params())
    assert r.decisao == REJEITA_H0
    assert "(τ = -0,728: correlação negativa forte)" in r.interpretacao
    assert "rejeita-se H₀ (τ = 0: não há correlação entre 'x' e 'y')" in r.interpretacao
    assert any("Há empates nos valores" in a for a in r.avisos)
    resumo = dict(r.tabelas["Resumo"].itertuples(index=False))
    assert resumo["Método do p-valor"].startswith("Aproximação normal")
    r = TesteCorrelacaoKendall().executar(_df("distintos"), _params("ρ < 0"))
    assert r.decisao == NAO_REJEITA_H0 and not r.avisos
    resumo = dict(r.tabelas["Resumo"].itertuples(index=False))
    assert resumo["Método do p-valor"] == "Exato (n = 20 < 50, sem empates)"
    (fig,) = r.figuras
    assert fig.tipo == "dispersao" and fig.dados["n_total"] == 20


def test_n_pequeno_sem_ic_e_linhas_descartadas():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan], "y": [2.0, 1.0, 4.0, 3.0, 5.0]})
    r = TesteCorrelacaoKendall().executar(df, _params())
    assert r.estatisticas["n"] == 4 and math.isnan(r.estatisticas["ic_inferior"])
    assert AVISO_IC_N4 in r.avisos
    assert "1 linha(s) com X ou Y ausente foram descartadas." in r.avisos


@pytest.mark.parametrize(
    ("dados", "params", "trecho"),
    [
        (None, {"x": None}, "Selecione a variável X."),
        (None, {"y": "x"}, "As variáveis X e Y devem ser colunas diferentes."),
        (None, {"alternativa": "μ ≠ μ₀"}, "Escolha uma hipótese alternativa válida."),
        ({"x": [1.0, 2.0], "y": [3.0, 1.0]}, {}, "ao menos 3 linhas com X e Y preenchidos"),
        ({"x": [1.0, 1.0, 1.0], "y": [3.0, 1.0, 2.0]}, {}, "A coluna 'x' não varia"),
    ],
)
def test_validacao(dados, params, trecho):
    df = _df("cars") if dados is None else pd.DataFrame(dados)
    erros = TesteCorrelacaoKendall().validar(df, _params(**params))
    assert any(trecho in e for e in erros), erros
    with pytest.raises(ErroValidacao):
        TesteCorrelacaoKendall().executar(df, _params(**params))


def test_n_grande_rapido():
    # kendalltau e as contagens são O(n log n): 100 000 linhas sem enumerar pares.
    import time

    rng = np.random.default_rng(1)
    x = rng.normal(size=100_000)
    df = pd.DataFrame({"x": x, "y": x + rng.normal(size=100_000)})
    inicio = time.perf_counter()
    r = TesteCorrelacaoKendall().executar(df, _params())
    assert time.perf_counter() - inicio < 10 and r.estatisticas["usou_exato"] == 0.0
    total = r.estatisticas["concordantes"] + r.estatisticas["discordantes"]
    assert total == 100_000 * 99_999 // 2  # sem empates
