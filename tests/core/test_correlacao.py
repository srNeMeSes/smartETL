"""Correlação de Pearson e de Spearman: checklist da seção 9 do CLAUDE.md.

Fonte dos valores de referência: tests/referencias_r/referencias_correlacao.json, gerado por
tests/referencias_r/gerar_referencias_correlacao.R com R 4.6.1 (`cor.test`; Spearman com
`exact = FALSE`), sobre mtcars (wt × mpg), cars (speed × dist, com empates) e uma base com um
outlier (semente 42). Também: t = r·√(n − 2)/√(1 − r²) à mão; Spearman = Pearson dos postos
médios (numpy); IC da Spearman pela z de Fisher com a variância de Bonett & Wright, à mão.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
from conftest import RAIZ
from scipy.stats import norm

from core.base import ErroValidacao, GrupoFiguras
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.correlacao import (
    ALTERNATIVAS_COR,
    POSTOS,
    VALORES,
    TesteCorrelacaoPearson,
    TesteCorrelacaoSpearman,
    classificar_forca,
    descrever,
)

PASTA = RAIZ / "tests" / "referencias_r"
R = json.loads((PASTA / "referencias_correlacao.json").read_text(encoding="utf-8"))
REL = 1e-9
DATASETS = ["mtcars", "cars", "outlier"]
ALT_R = {"two-sided": "two.sided", "greater": "greater", "less": "less"}


def _df(nome):
    return pd.read_csv(PASTA / f"correlacao_{nome}.csv")


def _params(alternativa="ρ ≠ 0", **extra):
    return {"x": "x", "y": "y", "alternativa": alternativa, "alfa": 0.05} | extra


def _postos_medios(v):
    ordem = np.argsort(v, kind="stable")
    postos = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[ordem[j + 1]] == v[ordem[i]]:
            j += 1
        postos[ordem[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return postos


# ---------------------------------------------------------------------------
# Pearson contra o R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nome", DATASETS)
@pytest.mark.parametrize("rotulo", list(ALTERNATIVAS_COR))
def test_pearson_contra_cor_test(nome, rotulo):
    alternativa = ALTERNATIVAS_COR[rotulo]
    ref = R[nome][f"pearson_{ALT_R[alternativa]}"]
    r = TesteCorrelacaoPearson().executar(_df(nome), _params(rotulo))
    e = r.estatisticas
    assert e["r"] == pytest.approx(ref["r"], rel=REL)
    assert e["t"] == pytest.approx(ref["t"], rel=REL)
    assert e["gl"] == ref["gl"]
    assert r.p_valor == pytest.approx(ref["p"], rel=1e-9)
    assert (e["ic_inferior"], e["ic_superior"]) == pytest.approx(
        (ref["ic_li"], ref["ic_ls"]), rel=1e-9
    )


@pytest.mark.parametrize("nome", DATASETS)
def test_pearson_t_a_mao_e_ic_de_90(nome):
    df = _df(nome)
    r = TesteCorrelacaoPearson().executar(df, _params(alfa=0.10))
    coef, n = r.estatisticas["r"], len(df)
    assert r.estatisticas["t"] == pytest.approx(
        coef * math.sqrt(n - 2) / math.sqrt(1 - coef**2), rel=REL
    )
    assert np.corrcoef(df.x, df.y)[0, 1] == pytest.approx(coef, rel=1e-12)
    ref = R[nome]["pearson_ic90"]
    assert (r.estatisticas["ic_inferior"], r.estatisticas["ic_superior"]) == pytest.approx(
        (ref["ic_li"], ref["ic_ls"]), rel=1e-9
    )
    assert "IC 90% para ρ" in dict(r.tabelas["Resumo"].itertuples(index=False))


# ---------------------------------------------------------------------------
# Spearman contra o R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nome", DATASETS)
@pytest.mark.parametrize("rotulo", list(ALTERNATIVAS_COR))
def test_spearman_contra_cor_test(nome, rotulo):
    alternativa = ALTERNATIVAS_COR[rotulo]
    ref = R[nome][f"spearman_{ALT_R[alternativa]}"]
    r = TesteCorrelacaoSpearman().executar(_df(nome), _params(rotulo))
    assert r.estatisticas["rho"] == pytest.approx(ref["rho"], rel=REL)
    assert r.p_valor == pytest.approx(ref["p"], rel=1e-9)


@pytest.mark.parametrize("nome", DATASETS)
@pytest.mark.parametrize("rotulo", list(ALTERNATIVAS_COR))
def test_spearman_postos_medios_e_ic_bonett_wright(nome, rotulo):
    df = _df(nome)
    r = TesteCorrelacaoSpearman().executar(df, _params(rotulo))
    rho, n = r.estatisticas["rho"], len(df)
    postos_x, postos_y = _postos_medios(df.x.to_numpy()), _postos_medios(df.y.to_numpy())
    assert np.corrcoef(postos_x, postos_y)[0, 1] == pytest.approx(rho, rel=1e-12)
    ep = math.sqrt((1 + rho**2 / 2) / (n - 3))
    z = math.atanh(rho)
    alternativa = ALTERNATIVAS_COR[rotulo]
    if alternativa == "two-sided":
        c = norm.ppf(0.975)
        esperado = (math.tanh(z - c * ep), math.tanh(z + c * ep))
    elif alternativa == "greater":
        esperado = (math.tanh(z - norm.ppf(0.95) * ep), 1.0)
    else:
        esperado = (-1.0, math.tanh(z + norm.ppf(0.95) * ep))
    assert (r.estatisticas["ic_inferior"], r.estatisticas["ic_superior"]) == pytest.approx(
        esperado, rel=1e-12
    )


# ---------------------------------------------------------------------------
# Força, decisão, avisos e card
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("r", "forca", "texto"),
    [
        (0.0, "desprezível", "correlação desprezível"),
        (0.099, "desprezível", "correlação desprezível"),
        (0.1, "fraca", "correlação positiva fraca"),
        (-0.299, "fraca", "correlação negativa fraca"),
        (0.3, "moderada", "correlação positiva moderada"),
        (-0.5, "forte", "correlação negativa forte"),
        (1.0, "forte", "correlação positiva forte"),
    ],
)
def test_faixas_de_forca(r, forca, texto):
    assert classificar_forca(r) == forca and descrever(r) == texto


def test_decisao_interpretacao_e_unilaterais():
    df = _df("mtcars")  # r = −0,868
    r = TesteCorrelacaoPearson().executar(df, _params())
    assert r.decisao == REJEITA_H0
    assert "rejeita-se H₀ (ρ = 0: não há correlação entre 'x' e 'y')" in r.interpretacao
    assert "(r = -0,868: correlação negativa forte)" in r.interpretacao
    r = TesteCorrelacaoPearson().executar(df, _params("ρ > 0"))
    assert r.decisao == NAO_REJEITA_H0
    assert "Não há evidência suficiente de correlação positiva" in r.interpretacao
    r = TesteCorrelacaoSpearman().executar(df, _params("ρ < 0"))
    assert r.decisao == REJEITA_H0 and "correlação negativa entre 'x' e 'y'" in r.interpretacao
    assert "(ρₛ = " in r.interpretacao


def test_card_pearson_x_spearman():
    df = _df("cars")
    teste = TesteCorrelacaoPearson()
    inicial = teste.comparacao_inicial()
    assert (inicial.titulo_esquerda, inicial.titulo_direita) == ("Pearson", "Spearman")
    assert inicial.hipoteses == ["ρ ≠ 0", "ρ > 0", "ρ < 0"]
    r = teste.executar(df, _params())
    for (p_pearson, p_spearman), alternativa in zip(
        r.comparacao.linhas, ALTERNATIVAS_COR.values(), strict=True
    ):
        assert p_pearson == pytest.approx(R["cars"][f"pearson_{ALT_R[alternativa]}"]["p"], rel=1e-9)
        assert p_spearman == pytest.approx(
            R["cars"][f"spearman_{ALT_R[alternativa]}"]["p"], rel=1e-9
        )
    assert TesteCorrelacaoSpearman().comparacao_inicial() is None


def test_avisos():
    r = TesteCorrelacaoPearson().executar(_df("outlier"), _params())
    assert any("diferem bastante" in a for a in r.avisos)  # o outlier derruba a Pearson
    assert not TesteCorrelacaoPearson().executar(_df("mtcars"), _params()).avisos
    df = pd.concat([_df("cars"), pd.DataFrame({"x": [np.nan, 5.0], "y": [3.0, np.nan]})])
    r = TesteCorrelacaoSpearman().executar(df, _params())
    assert r.estatisticas["n"] == 50
    assert "2 linha(s) com X ou Y ausente foram descartadas." in r.avisos
    assert any("empates" in a for a in r.avisos)


def test_figuras():
    df = _df("mtcars")
    (fig,) = TesteCorrelacaoPearson().executar(df, _params()).figuras
    assert fig.tipo == "dispersao" and fig.dados["n_total"] == 32
    (reta,) = fig.dados["linhas"]
    inclinacao, intercepto = np.polyfit(df.x, df.y, 1)
    assert reta["y1"] == pytest.approx(intercepto + inclinacao * df.x.min())
    (grupo,) = TesteCorrelacaoSpearman().executar(df, _params()).figuras
    assert isinstance(grupo, GrupoFiguras) and list(grupo.opcoes) == [VALORES, POSTOS]
    assert max(grupo.opcoes[POSTOS].dados["x"]) == 32


# ---------------------------------------------------------------------------
# Entradas inválidas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("classe", [TesteCorrelacaoPearson, TesteCorrelacaoSpearman])
@pytest.mark.parametrize(
    ("df", "params", "trecho"),
    [
        (None, {"x": None}, "Selecione a variável X."),
        (None, {"y": "nao_existe"}, "A coluna 'nao_existe' não existe no arquivo."),
        (None, {"y": "x"}, "As variáveis X e Y devem ser colunas diferentes."),
        (None, {"alternativa": "μ ≠ μ₀"}, "Escolha uma hipótese alternativa válida."),
        (None, {"alfa": 0}, "O nível de significância (α) deve estar entre 0 e 1."),
        ({"x": [1.0, 2.0], "y": [3.0, 1.0]}, {}, "ao menos 3 linhas com X e Y preenchidos (há 2)"),
        ({"x": [1.0, 1.0, 1.0], "y": [3.0, 1.0, 2.0]}, {}, "A coluna 'x' não varia"),
        ({"x": ["a", "b", "c"], "y": [3.0, 1.0, 2.0]}, {}, "A coluna 'x' não é numérica."),
    ],
)
def test_validacao(classe, df, params, trecho):
    dados = _df("cars") if df is None else pd.DataFrame(df)
    erros = classe().validar(dados, _params(**params))
    assert any(trecho in e for e in erros), erros
    with pytest.raises(ErroValidacao):
        classe().executar(dados, _params(**params))


@pytest.mark.parametrize("classe", [TesteCorrelacaoPearson, TesteCorrelacaoSpearman])
def test_formulario(classe):
    specs = classe().parametros()
    assert [(s.nome, s.tipo) for s in specs] == [
        ("x", "coluna_numerica"),
        ("y", "coluna_numerica"),
        ("alternativa", "opcao"),
        ("alfa", "alfa"),
    ]
    assert specs[2].padrao == "ρ ≠ 0" and classe().validar(_df("cars"), _params()) == []


@pytest.mark.parametrize("classe", [TesteCorrelacaoPearson, TesteCorrelacaoSpearman])
def test_n3_sem_ic_nas_duas(classe):
    # Com n = 3, var(z de Fisher) = 1/(n − 3) é infinita: nenhum IC, nas duas correlações.
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 1.0, 5.0]})
    r = classe().executar(df, _params())
    assert math.isnan(r.estatisticas["ic_inferior"]) and math.isnan(r.estatisticas["ic_superior"])
    assert "Com n = 3 o intervalo de confiança não é calculado." in r.avisos
    assert any(v == "[—; —]" for v in r.tabelas["Resumo"]["Valor"])
