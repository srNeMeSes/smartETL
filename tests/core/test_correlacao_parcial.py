"""Correlação parcial: checklist da seção 9 do CLAUDE.md.

Fonte dos valores de referência: tests/referencias_r/referencias_parcial.json, gerado por
tests/referencias_r/gerar_referencias_parcial.R com R 4.6.1 e ppcor 1.1 (`pcor.test`, Pearson
e Spearman) sobre mtcars (wt × mpg controlando hp, e hp + disp) e iris (Sepal.Length ×
Petal.Length controlando Petal.Width, com empates). Também: a fórmula fechada da parcial de 1ª
ordem, r_xy·z = (r_xy − r_xz·r_yz)/√((1 − r_xz²)(1 − r_yz²)), à mão com numpy; p unilaterais
pela mesma estatística t; IC pela z de Fisher com var = 1/(n − 3 − k), à mão.
"""

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
from core.testes.correlacao_parcial import PEARSON, SPEARMAN, TesteCorrelacaoParcial

PASTA = RAIZ / "tests" / "referencias_r"
R = json.loads((PASTA / "referencias_parcial.json").read_text(encoding="utf-8"))
CASOS = {
    "mtcars_1": ("mtcars", ["z1"]),
    "mtcars_2": ("mtcars", ["z1", "z2"]),
    "iris_1": ("iris", ["z1"]),
}
REL = 1e-10


def _df(base):
    return pd.read_csv(PASTA / f"parcial_{base}.csv")


def _params(controles=("z1",), **extra):
    return {"x": "x", "y": "y", "controles": list(controles), "alfa": 0.05} | extra


# ---------------------------------------------------------------------------
# Contra o R (ppcor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("caso", list(CASOS))
@pytest.mark.parametrize("metodo", [PEARSON, SPEARMAN])
def test_contra_pcor_test(caso, metodo):
    base, controles = CASOS[caso]
    ref = R[caso][metodo.lower()]
    r = TesteCorrelacaoParcial().executar(_df(base), _params(controles, metodo=metodo))
    e = r.estatisticas
    assert e["r"] == pytest.approx(ref["r"], rel=REL)
    assert e["t"] == pytest.approx(ref["t"], rel=REL)
    assert r.p_valor == pytest.approx(ref["p"], rel=1e-9)
    assert (e["n"], e["k"], e["gl"]) == (ref["n"], ref["k"], ref["n"] - 2 - ref["k"])
    assert e["r_simples"] == pytest.approx(ref["simples"], rel=REL)


@pytest.mark.parametrize("rotulo", list(ALTERNATIVAS_COR))
def test_unilaterais_pela_estatistica_t_e_ic(rotulo):
    df = _df("mtcars")
    alternativa = ALTERNATIVAS_COR[rotulo]
    r = TesteCorrelacaoParcial().executar(df, _params(["z1", "z2"], alternativa=rotulo))
    t, gl, coef = r.estatisticas["t"], 32 - 2 - 2, r.estatisticas["r"]
    esperado = {
        "two-sided": 2 * stats.t.sf(abs(t), gl),
        "greater": stats.t.sf(t, gl),
        "less": stats.t.cdf(t, gl),
    }[alternativa]
    assert r.p_valor == pytest.approx(esperado, rel=1e-12)
    ep, z = 1 / math.sqrt(32 - 3 - 2), math.atanh(coef)
    if alternativa == "two-sided":
        c = stats.norm.ppf(0.975)
        ic = (math.tanh(z - c * ep), math.tanh(z + c * ep))
    elif alternativa == "greater":
        ic = (math.tanh(z - stats.norm.ppf(0.95) * ep), 1.0)
    else:
        ic = (-1.0, math.tanh(z + stats.norm.ppf(0.95) * ep))
    assert (r.estatisticas["ic_inferior"], r.estatisticas["ic_superior"]) == pytest.approx(ic)


def test_formula_fechada_de_primeira_ordem():
    df = _df("iris")
    rxy, rxz, ryz = (
        np.corrcoef(df[a], df[b])[0, 1] for a, b in (("x", "y"), ("x", "z1"), ("y", "z1"))
    )
    fechada = (rxy - rxz * ryz) / math.sqrt((1 - rxz**2) * (1 - ryz**2))
    r = TesteCorrelacaoParcial().executar(df, _params())
    assert r.estatisticas["r"] == pytest.approx(fechada, rel=1e-12)


# ---------------------------------------------------------------------------
# Card, decisão, avisos e figura
# ---------------------------------------------------------------------------


def test_card_parcial_x_simples():
    df = _df("mtcars")
    teste = TesteCorrelacaoParcial()
    inicial = teste.comparacao_inicial()
    assert (inicial.titulo_esquerda, inicial.titulo_direita) == ("Parcial", "Simples")
    assert inicial.hipoteses == ["ρ ≠ 0", "ρ > 0", "ρ < 0"]
    r = teste.executar(df, _params())
    for (_, p_simples), alternativa in zip(
        r.comparacao.linhas, ALTERNATIVAS_COR.values(), strict=True
    ):
        assert p_simples == pytest.approx(
            stats.pearsonr(df.x, df.y, alternative=alternativa).pvalue
        )
    assert r.comparacao.linhas[0][0] == pytest.approx(R["mtcars_1"]["pearson"]["p"], rel=1e-9)


def test_decisao_interpretacao_e_aviso_de_divergencia():
    r = TesteCorrelacaoParcial().executar(_df("iris"), _params())
    assert r.decisao == REJEITA_H0
    assert (
        "rejeita-se H₀ (ρ parcial = 0: não há correlação entre 'x' e 'y', controlando por 'z1')"
        in (r.interpretacao)
    )
    assert "(r parcial = 0,542: correlação positiva forte)" in r.interpretacao
    # Simples 0,872 → parcial 0,542: parte da relação passa pelo controle.
    assert any("Controlar por 'z1' muda bastante a correlação" in a for a in r.avisos)
    r = TesteCorrelacaoParcial().executar(_df("iris"), _params(alternativa="ρ < 0"))
    assert r.decisao == NAO_REJEITA_H0 and "correlação negativa" in r.interpretacao


def test_resumo_e_figura_dos_residuos():
    df = _df("mtcars")
    r = TesteCorrelacaoParcial().executar(df, _params(["z1", "z2"]))
    resumo = dict(r.tabelas["Resumo"].itertuples(index=False))
    assert resumo["Variáveis de controle"] == "'z1', 'z2' (k = 2)"
    assert resumo["Graus de liberdade (n − 2 − k)"] == "28"
    assert resumo["Método"] == PEARSON
    (fig,) = r.figuras
    assert fig.tipo == "dispersao" and "controlando por 'z1', 'z2'" in fig.titulo
    assert abs(np.mean(fig.dados["x"])) < 1e-9  # resíduos têm média zero


def test_linhas_incompletas_descartadas():
    df = pd.concat([_df("mtcars"), pd.DataFrame({"x": [1.0], "y": [2.0], "z1": [np.nan]})])
    r = TesteCorrelacaoParcial().executar(df, _params())
    assert r.estatisticas["n"] == 32
    assert "1 linha(s) com X, Y ou algum controle ausente foram descartadas." in r.avisos


# ---------------------------------------------------------------------------
# Entradas inválidas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dados", "params", "trecho"),
    [
        (None, {"x": None}, "Selecione a variável X."),
        (None, {"y": "x"}, "As variáveis X e Y devem ser colunas diferentes."),
        (None, {"controles": []}, "Selecione ao menos uma variável de controle."),
        (None, {"controles": ["x"]}, "X e Y não podem ser também variáveis de controle."),
        (
            None,
            {"controles": ["z1", "z1"]},
            "As variáveis de controle devem ser colunas diferentes.",
        ),
        (None, {"controles": ["nao_existe"]}, "A coluna 'nao_existe' não existe no arquivo."),
        (None, {"metodo": "Kendall"}, "Escolha o método."),
        (None, {"alternativa": "μ ≠ μ₀"}, "Escolha uma hipótese alternativa válida."),
        ({"x": [1.0, 2, 3, 4], "y": [2.0, 1, 4, 3], "z1": [1.0, 3, 2, 5]}, {}, "ao menos 5 linhas"),
        (
            {"x": [1.0, 2, 3, 4, 5], "y": [2.0, 1, 4, 3, 6], "z1": [7.0] * 5},
            {},
            "constantes ou combinação exata",
        ),
        (
            {"x": [1.0, 2, 3, 4, 5], "y": [2.0, 1, 4, 3, 6], "z1": [2.0, 4, 6, 8, 10]},
            {},
            "A variável 'x' é explicada totalmente pelos controles",
        ),
    ],
)
def test_validacao(dados, params, trecho):
    df = _df("mtcars") if dados is None else pd.DataFrame(dados)
    erros = TesteCorrelacaoParcial().validar(df, _params(**params))
    assert any(trecho in e for e in erros), erros
    with pytest.raises(ErroValidacao):
        TesteCorrelacaoParcial().executar(df, _params(**params))


def test_formulario():
    specs = TesteCorrelacaoParcial().parametros()
    assert [(s.nome, s.tipo) for s in specs] == [
        ("x", "coluna_numerica"),
        ("y", "coluna_numerica"),
        ("controles", "multi_coluna"),
        ("metodo", "opcao"),
        ("alternativa", "opcao"),
        ("alfa", "alfa"),
    ]
    assert specs[3].padrao == PEARSON and specs[3].opcoes == [PEARSON, SPEARMAN]
