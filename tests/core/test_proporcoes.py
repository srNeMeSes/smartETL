"""Teste Z (uma proporção): checklist da seção 9 do CLAUDE.md.

Fontes dos valores de referência (independentes do wrapper em core/testes/proporcoes.py, que usa
statsmodels e scipy):
- z = (p̂ − p₀)/√(p₀(1 − p₀)/n) e p-valores pela normal via `math.erfc` (biblioteca padrão);
- IC de Wilson pela fórmula fechada, com o quantil de `statistics.NormalDist` (biblioteca padrão);
- binomial exato somando a função de probabilidade com `math.comb` (bilateral: soma das
  probabilidades ≤ à do valor observado);
- h de Cohen pela fórmula 2·arcsen(√p̂) − 2·arcsen(√p₀).
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from core.base import ErroValidacao
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.proporcoes import ALTERNATIVAS_P, TesteZ1Prop, ic_wilson

N, K = 50, 31
REL = 1e-6


@pytest.fixture
def teste():
    return TesteZ1Prop()


@pytest.fixture
def df():
    return pd.DataFrame({"resposta": ["Sim"] * K + ["Não"] * (N - K), "idade": range(N)})


def _params(**extra):
    return {
        "coluna": "resposta",
        "sucesso": "Sim",
        "p0": 0.5,
        "alternativa": "p ≠ p₀",
        "alfa": 0.05,
    } | extra


def _rotulo(alternativa: str) -> str:
    return next(r for r, a in ALTERNATIVAS_P.items() if a == alternativa)


def _z_manual(k, n, p0):
    return (k / n - p0) / math.sqrt(p0 * (1 - p0) / n)


def _p_normal(z, alternativa):
    return {
        "two-sided": math.erfc(abs(z) / math.sqrt(2)),
        "greater": 0.5 * math.erfc(z / math.sqrt(2)),
        "less": 0.5 * math.erfc(-z / math.sqrt(2)),
    }[alternativa]


def _wilson_manual(k, n, confianca):
    z = NormalDist().inv_cdf(1 - (1 - confianca) / 2)
    p = k / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    meia = z / (1 + z**2 / n) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return centro - meia, centro + meia


def _binomial_manual(k, n, p0, alternativa):
    pmf = [math.comb(n, i) * p0**i * (1 - p0) ** (n - i) for i in range(n + 1)]
    if alternativa == "greater":
        return sum(pmf[k:])
    if alternativa == "less":
        return sum(pmf[: k + 1])
    return min(1.0, sum(q for q in pmf if q <= pmf[k] * (1 + 1e-7)))


# ---------------------------------------------------------------------------
# Valores de referência
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p0", [0.5, 0.4, 0.7])
@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_z_e_p_contra_formula_manual(teste, df, p0, alternativa):
    r = teste.executar(df, _params(p0=p0, alternativa=_rotulo(alternativa)))
    z = _z_manual(K, N, p0)
    assert r.estatisticas["z"] == pytest.approx(z, rel=REL)
    assert r.p_valor == pytest.approx(_p_normal(z, alternativa), rel=REL)
    assert r.estatisticas["p_hat"] == pytest.approx(K / N, rel=REL)
    assert r.estatisticas["erro_padrao"] == pytest.approx(math.sqrt(p0 * (1 - p0) / N), rel=REL)
    h = 2 * math.asin(math.sqrt(K / N)) - 2 * math.asin(math.sqrt(p0))
    assert r.estatisticas["h_cohen"] == pytest.approx(h, rel=REL)


@pytest.mark.parametrize("alfa", [0.01, 0.05, 0.10])
def test_ic_wilson_bilateral_contra_formula(teste, df, alfa):
    r = teste.executar(df, _params(alfa=alfa))
    baixo, alto = _wilson_manual(K, N, 1 - alfa)
    assert r.estatisticas["ic_inferior"] == pytest.approx(baixo, rel=REL)
    assert r.estatisticas["ic_superior"] == pytest.approx(alto, rel=REL)


@pytest.mark.parametrize("alfa", [0.01, 0.05, 0.10])
def test_ic_wilson_unilateral(alfa):
    baixo_bilateral, alto_bilateral = _wilson_manual(K, N, 1 - 2 * alfa)
    assert ic_wilson(K, N, alfa, "greater") == pytest.approx((baixo_bilateral, 1.0), rel=REL)
    assert ic_wilson(K, N, alfa, "less") == pytest.approx((0.0, alto_bilateral), rel=REL)


@pytest.mark.parametrize("p0", [0.5, 0.3])
def test_binomial_exato_contra_soma_manual(teste, df, p0):
    r = teste.executar(df, _params(p0=p0))
    esperado = [_binomial_manual(K, N, p0, a) for a in ("two-sided", "greater", "less")]
    assert [p for _, p in r.comparacao.linhas] == pytest.approx(esperado, rel=REL)
    assert r.estatisticas["p_binomial"] == pytest.approx(esperado[0], rel=REL)


def test_comparacao_tres_alternativas(teste, df):
    r = teste.executar(df, _params(alternativa="p < p₀"))
    comp = r.comparacao
    assert (comp.titulo_esquerda, comp.titulo_direita) == ("Teste Z", "Binomial exato")
    assert comp.hipoteses == ["p ≠ p₀", "p > p₀", "p < p₀"]
    z = _z_manual(K, N, 0.5)
    for (p_z, _), alt in zip(comp.linhas, ["two-sided", "greater", "less"], strict=True):
        assert p_z == pytest.approx(_p_normal(z, alt), rel=REL)
    assert comp.linhas[2][0] == pytest.approx(r.p_valor, rel=REL)


# ---------------------------------------------------------------------------
# Sucesso, decisão, interpretação e saída
# ---------------------------------------------------------------------------


def test_trocar_o_sucesso_inverte_a_proporcao(teste, df):
    a = teste.executar(df, _params())
    b = teste.executar(df, _params(sucesso="Não"))
    assert b.estatisticas["p_hat"] == pytest.approx(1 - a.estatisticas["p_hat"], rel=REL)
    assert b.estatisticas["z"] == pytest.approx(-a.estatisticas["z"], rel=REL)


def test_coluna_numerica_0_1(teste):
    df = pd.DataFrame({"comprou": [1.0] * 12 + [0.0] * 8})
    r = teste.executar(df, _params(coluna="comprou", sucesso="1"))
    assert r.estatisticas["sucessos"] == 12 and r.estatisticas["p_hat"] == 0.6


def test_rejeita_h0_e_interpretacao(teste, df):
    r = teste.executar(df, _params(p0=0.4))
    assert r.p_valor < 0.01 and r.decisao == REJEITA_H0
    assert "rejeita-se H₀ (p = 0,4) em favor de H₁ (p ≠ 0,4)" in r.interpretacao
    assert (
        "Há evidência estatística de que a proporção de 'Sim' em 'resposta' é diferente de 0,4."
    ) in r.interpretacao


def test_nao_rejeita_h0(teste, df):
    r = teste.executar(df, _params())
    assert r.decisao == NAO_REJEITA_H0  # p ≈ 0,090
    assert "Não há evidência suficiente de que a proporção de 'Sim'" in r.interpretacao


def test_alfa_e_alternativa_mudam_decisao(teste, df):
    assert teste.executar(df, _params(alfa=0.10)).decisao == REJEITA_H0
    assert teste.executar(df, _params(alternativa="p > p₀")).decisao == REJEITA_H0
    assert teste.executar(df, _params(alternativa="p < p₀")).decisao == NAO_REJEITA_H0


def test_resultado_completo(teste, df):
    r = teste.executar(df, _params(p0="0,5"))
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Observações (n)"] == "50"
    assert medidas["Sucessos ('Sim')"] == "31"
    assert medidas["Proporção amostral (p̂)"] == "0,6200 (62,0%)"
    assert "IC 95% de Wilson para p" in medidas
    assert "p-valor (binomial exato)" in medidas
    (figura,) = r.figuras
    assert figura.tipo == "barras"
    assert [(c["rotulo"], c["valor"]) for c in figura.dados["categorias"]] == [
        ("Sim (sucesso)", pytest.approx(0.62)),
        ("Não", pytest.approx(0.38)),
    ]
    assert figura.dados["maximo"] == 1.0 and figura.dados["percentual"] is True
    assert figura.dados["referencias"][0]["valor"] == 0.5
    assert r.avisos == []


def test_ic_unilateral_rotulado(teste, df):
    medidas = dict(
        teste.executar(df, _params(alternativa="p > p₀")).tabelas["Resumo"].itertuples(index=False)
    )
    assert medidas["IC 95% de Wilson para p (unilateral)"].endswith("1,0000]")


# ---------------------------------------------------------------------------
# Casos de borda e entradas inválidas
# ---------------------------------------------------------------------------


def test_ausentes_ignorados(teste):
    df = pd.DataFrame({"resposta": ["Sim"] * K + ["Não"] * (N - K) + [None, np.nan]})
    r = teste.executar(df, _params())
    assert r.estatisticas["n"] == N
    assert "2 valor(es) ausente(s) em 'resposta' foram ignorados." in r.avisos


def test_aviso_amostra_pequena_para_normal(teste):
    df = pd.DataFrame({"resposta": ["Sim"] * 6 + ["Não"] * 2})
    r = teste.executar(df, _params())
    assert any("aproximação normal é fraca" in a for a in r.avisos)
    assert any("n·p₀ = 4,0" in a for a in r.avisos)


@pytest.mark.parametrize(
    ("params", "mensagem"),
    [
        ({"coluna": None}, "Selecione a variável."),
        ({"coluna": "nao_existe"}, "A coluna 'nao_existe' não existe no arquivo."),
        ({"coluna": "idade"}, "A coluna 'idade' deve ter exatamente 2 valores distintos (tem 50)."),
        ({"sucesso": None}, "Escolha o valor que conta como sucesso."),
        ({"sucesso": "Talvez"}, "O valor 'Talvez' não aparece na coluna 'resposta'."),
        ({"p0": 0}, "Informe a proporção hipotética (p₀) entre 0 e 1 (ex.: 0,5)."),
        ({"p0": 1.2}, "Informe a proporção hipotética (p₀) entre 0 e 1 (ex.: 0,5)."),
        ({"p0": "abc"}, "Informe a proporção hipotética (p₀) entre 0 e 1 (ex.: 0,5)."),
        ({"alternativa": "μ ≠ μ₀"}, "Escolha uma hipótese alternativa válida."),
        ({"alfa": 2}, "O nível de significância (α) deve estar entre 0 e 1."),
    ],
)
def test_validacao(teste, df, params, mensagem):
    assert mensagem in teste.validar(df, _params(**params))
    with pytest.raises(ErroValidacao):
        teste.executar(df, _params(**params))


def test_coluna_com_um_so_valor(teste):
    erros = teste.validar(pd.DataFrame({"resposta": ["Sim"] * 5}), _params())
    assert "A coluna 'resposta' deve ter exatamente 2 valores distintos (tem 1)." in erros


def test_parametros_validos_sem_erros(teste, df):
    assert teste.validar(df, _params()) == []


def test_formulario(teste):
    specs = teste.parametros()
    assert [(s.nome, s.tipo) for s in specs] == [
        ("coluna", "coluna_binaria"),
        ("sucesso", "nivel"),
        ("p0", "numero"),
        ("alternativa", "opcao"),
        ("alfa", "alfa"),
    ]
    assert specs[1].depende_de == "coluna"
    assert specs[2].padrao == 0.5
    assert teste.comparacao_inicial().titulo_direita == "Binomial exato"
