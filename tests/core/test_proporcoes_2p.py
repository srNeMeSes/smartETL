"""Teste Z (duas proporções): checklist da seção 9 do CLAUDE.md.

Fontes dos valores de referência (independentes do wrapper, que usa statsmodels e scipy), todas
com a biblioteca padrão:
- z com proporção combinada e p-valores pela normal (`math.erfc`);
- IC de Wald (não combinado) para p₁ − p₂ com o quantil de `statistics.NormalDist`;
- h de Cohen, odds ratio e IC de Woolf, exp(ln OR ± z·√(1/a + 1/b + 1/c + 1/d)), pelas fórmulas;
- Fisher exato pela distribuição hipergeométrica com `math.comb` (bilateral: soma das
  probabilidades ≤ à da tabela observada).
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from core.base import ErroValidacao
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.proporcoes import ALTERNATIVAS_2P, TesteZ2Prop, ic_diferenca_wald

K1, N1, K2, N2 = 30, 50, 18, 45  # grupo A: 60%; grupo B: 40%
REL = 1e-6


@pytest.fixture
def teste():
    return TesteZ2Prop()


def _dados(k1=K1, n1=N1, k2=K2, n2=N2, g=("B", "A")):
    # O grupo "B" vem primeiro nos dados; o grupo 1 continua sendo "A" (ordem crescente).
    linhas = (
        [("Sim", g[1])] * k1
        + [("Não", g[1])] * (n1 - k1)
        + [("Sim", g[0])] * k2
        + [("Não", g[0])] * (n2 - k2)
    )
    return pd.DataFrame(linhas[::-1], columns=["comprou", "loja"])


@pytest.fixture
def df():
    return _dados()


def _params(**extra):
    return {
        "coluna": "comprou",
        "sucesso": "Sim",
        "grupo": "loja",
        "alternativa": "p₁ ≠ p₂",
        "alfa": 0.05,
    } | extra


def _rotulo(alternativa: str) -> str:
    return next(r for r, a in ALTERNATIVAS_2P.items() if a == alternativa)


def _z_manual(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    return (k1 / n1 - k2 / n2) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))


def _p_normal(z, alternativa):
    return {
        "two-sided": math.erfc(abs(z) / math.sqrt(2)),
        "greater": 0.5 * math.erfc(z / math.sqrt(2)),
        "less": 0.5 * math.erfc(-z / math.sqrt(2)),
    }[alternativa]


def _fisher_manual(a, b, c, d, alternativa):
    n1, n2, m = a + b, c + d, a + c
    total = math.comb(n1 + n2, m)
    pmf = {
        x: math.comb(n1, x) * math.comb(n2, m - x) / total
        for x in range(max(0, m - n2), min(n1, m) + 1)
    }
    if alternativa == "greater":
        return sum(p for x, p in pmf.items() if x >= a)
    if alternativa == "less":
        return sum(p for x, p in pmf.items() if x <= a)
    return min(1.0, sum(p for p in pmf.values() if p <= pmf[a] * (1 + 1e-7)))


# ---------------------------------------------------------------------------
# Valores de referência
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_z_combinado_contra_formula_manual(teste, df, alternativa):
    r = teste.executar(df, _params(alternativa=_rotulo(alternativa)))
    z = _z_manual(K1, N1, K2, N2)
    e = r.estatisticas
    assert e["z"] == pytest.approx(z, rel=REL)
    assert r.p_valor == pytest.approx(_p_normal(z, alternativa), rel=REL)
    assert (e["p1"], e["p2"]) == pytest.approx((K1 / N1, K2 / N2), rel=REL)
    assert e["p_combinada"] == pytest.approx((K1 + K2) / (N1 + N2), rel=REL)
    assert e["diferenca"] == pytest.approx(K1 / N1 - K2 / N2, rel=REL)


@pytest.mark.parametrize("alfa", [0.01, 0.05, 0.10])
def test_ic_wald_da_diferenca(teste, df, alfa):
    r = teste.executar(df, _params(alfa=alfa))
    p1, p2 = K1 / N1, K2 / N2
    z = NormalDist().inv_cdf(1 - alfa / 2)
    meia = z * math.sqrt(p1 * (1 - p1) / N1 + p2 * (1 - p2) / N2)
    assert r.estatisticas["ic_inferior"] == pytest.approx(p1 - p2 - meia, rel=REL)
    assert r.estatisticas["ic_superior"] == pytest.approx(p1 - p2 + meia, rel=REL)
    baixo_bi, alto_bi = ic_diferenca_wald(K1, N1, K2, N2, 2 * alfa, "two-sided")
    assert ic_diferenca_wald(K1, N1, K2, N2, alfa, "greater") == pytest.approx((baixo_bi, 1.0))
    assert ic_diferenca_wald(K1, N1, K2, N2, alfa, "less") == pytest.approx((-1.0, alto_bi))


@pytest.mark.parametrize("alfa", [0.01, 0.05, 0.10])
def test_h_de_cohen_e_odds_ratio_com_ic_de_woolf(teste, df, alfa):
    e = teste.executar(df, _params(alfa=alfa)).estatisticas
    a, b, c, d = K1, N1 - K1, K2, N2 - K2
    h = 2 * math.asin(math.sqrt(K1 / N1)) - 2 * math.asin(math.sqrt(K2 / N2))
    assert e["h_cohen"] == pytest.approx(h, rel=REL)
    odds = (a * d) / (b * c)
    assert e["odds_ratio"] == pytest.approx(odds, rel=REL)
    z = NormalDist().inv_cdf(1 - alfa / 2)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    assert e["or_ic_inferior"] == pytest.approx(math.exp(math.log(odds) - z * se), rel=REL)
    assert e["or_ic_superior"] == pytest.approx(math.exp(math.log(odds) + z * se), rel=REL)


@pytest.mark.parametrize(
    ("k1", "n1", "k2", "n2"), [(K1, N1, K2, N2), (7, 9, 2, 10), (3, 12, 9, 11)]
)
def test_fisher_exato_contra_hipergeometrica(teste, k1, n1, k2, n2):
    r = teste.executar(_dados(k1, n1, k2, n2), _params())
    esperado = [
        _fisher_manual(k1, n1 - k1, k2, n2 - k2, a) for a in ("two-sided", "greater", "less")
    ]
    assert [p for _, p in r.comparacao.linhas] == pytest.approx(esperado, rel=REL)


def test_comparacao_tres_alternativas(teste, df):
    r = teste.executar(df, _params(alternativa="p₁ > p₂"))
    comp = r.comparacao
    assert (comp.titulo_esquerda, comp.titulo_direita) == ("Teste Z", "Fisher exato")
    assert comp.hipoteses == ["p₁ ≠ p₂", "p₁ > p₂", "p₁ < p₂"]
    z = _z_manual(K1, N1, K2, N2)
    for (p_z, _), alt in zip(comp.linhas, ["two-sided", "greater", "less"], strict=True):
        assert p_z == pytest.approx(_p_normal(z, alt), rel=REL)
    assert comp.linhas[1][0] == pytest.approx(r.p_valor, rel=REL)


# ---------------------------------------------------------------------------
# Grupos, decisão, interpretação e saída
# ---------------------------------------------------------------------------


def test_grupo_1_em_ordem_crescente_e_tabela_2x2(teste, df):
    r = teste.executar(df, _params())
    tabela = r.tabelas["Tabela 2×2"]
    assert list(tabela.columns) == ["Grupo", "Sucesso ('Sim')", "Fracasso", "Total"]
    assert tabela.values.tolist() == [["A", 30, 20, 50], ["B", 18, 27, 45]]
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Grupo 1"] == "'A' (n = 50)"
    assert medidas["Proporção de 'Sim' no grupo 1 (p̂₁)"] == "0,6000 (60,0%)"
    assert medidas["Proporção de 'Sim' no grupo 2 (p̂₂)"] == "0,4000 (40,0%)"


def test_trocar_sucesso_inverte_a_diferenca(teste, df):
    a = teste.executar(df, _params())
    b = teste.executar(df, _params(sucesso="Não"))
    assert b.estatisticas["diferenca"] == pytest.approx(-a.estatisticas["diferenca"], rel=REL)
    assert b.estatisticas["odds_ratio"] == pytest.approx(1 / a.estatisticas["odds_ratio"], rel=REL)


def test_interpretacao(teste, df):
    r = teste.executar(df, _params(alternativa="p₁ > p₂"))
    assert r.decisao == REJEITA_H0  # p ≈ 0,026
    assert "rejeita-se H₀ (p₁ = p₂) em favor de H₁ (p₁ > p₂)" in r.interpretacao
    assert (
        "Há evidência estatística de que a proporção de 'Sim' em 'comprou' no grupo 'A' é "
        "maior que a proporção no grupo 'B'."
    ) in r.interpretacao
    r = teste.executar(df, _params())
    assert r.decisao == NAO_REJEITA_H0  # bilateral p ≈ 0,052
    assert "Não há evidência suficiente" in r.interpretacao
    assert teste.executar(df, _params(alfa=0.10)).decisao == REJEITA_H0


def test_resultado_completo(teste, df):
    r = teste.executar(df, _params(alfa=0.10))
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    for chave in (
        "IC 90% para p₁ − p₂",
        "h de Cohen",
        "Razão de chances (odds ratio)",
        "IC 90% para a odds ratio (Woolf)",
        "p-valor (Fisher exato)",
        "Erro padrão (combinado)",
    ):
        assert chave in medidas
    assert medidas["Razão de chances (odds ratio)"] == "2,2500"
    (figura,) = r.figuras
    assert figura.tipo == "barras"
    assert [c["rotulo"] for c in figura.dados["categorias"]] == ["A (n = 50)", "B (n = 45)"]
    assert figura.dados["referencias"][0]["valor"] == pytest.approx(48 / 95)
    assert r.avisos == []


def test_ic_unilateral_rotulado(teste, df):
    medidas = dict(
        teste.executar(df, _params(alternativa="p₁ < p₂")).tabelas["Resumo"].itertuples(index=False)
    )
    assert medidas["IC 95% para p₁ − p₂ (unilateral)"].startswith("[-1,0000;")


# ---------------------------------------------------------------------------
# Casos de borda e entradas inválidas
# ---------------------------------------------------------------------------


def test_linhas_incompletas_descartadas(teste, df):
    extra = pd.DataFrame({"comprou": [None, "Sim", np.nan], "loja": ["A", None, None]})
    r = teste.executar(pd.concat([df, extra], ignore_index=True), _params())
    assert (r.estatisticas["n1"], r.estatisticas["n2"]) == (50, 45)
    assert "3 linha(s) com resposta ou grupo ausente foram descartadas." in r.avisos


def test_celula_zero_odds_ratio_indefinida(teste):
    r = teste.executar(_dados(10, 10, 4, 10), _params())
    assert math.isnan(r.estatisticas["odds_ratio"])
    assert any("razão de chances (odds ratio) não é definida" in a for a in r.avisos)
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Razão de chances (odds ratio)"] == "— (célula zero)"


def test_aviso_frequencia_esperada_baixa(teste):
    r = teste.executar(_dados(3, 6, 1, 7), _params())
    assert any("frequência esperada menor que 5" in a for a in r.avisos)


@pytest.mark.parametrize(
    ("params", "mensagem"),
    [
        ({"coluna": None}, "Selecione a resposta."),
        ({"grupo": None}, "Selecione o grupo."),
        ({"grupo": "nao_existe"}, "A coluna 'nao_existe' não existe no arquivo."),
        ({"grupo": "comprou"}, "A resposta e o grupo devem ser colunas diferentes."),
        ({"sucesso": None}, "Escolha o valor que conta como sucesso."),
        ({"sucesso": "Talvez"}, "O valor 'Talvez' não aparece na coluna 'comprou'."),
        ({"alternativa": "p ≠ p₀"}, "Escolha uma hipótese alternativa válida."),
        ({"alfa": -1}, "O nível de significância (α) deve estar entre 0 e 1."),
    ],
)
def test_validacao_parametros(teste, df, params, mensagem):
    assert mensagem in teste.validar(df, _params(**params))
    with pytest.raises(ErroValidacao):
        teste.executar(df, _params(**params))


@pytest.mark.parametrize(
    ("comprou", "loja", "trecho"),
    [
        (
            ["Sim", "Não", "Sim", "Não"],
            ["a", "b", "c", "a"],
            "exatamente 2 níveis com dados válidos (tem 3)",
        ),
        (
            ["Sim", "Não", "Talvez", "Sim"],
            ["a", "b", "a", "b"],
            "exatamente 2 valores distintos (tem 3)",
        ),
        (
            ["Sim", "Sim", "Não", "Sim"],
            ["a", "b", None, "a"],
            "Todas as observações válidas são sucesso",
        ),
    ],
)
def test_validacao_dados(teste, comprou, loja, trecho):
    erros = teste.validar(pd.DataFrame({"comprou": comprou, "loja": loja}), _params())
    assert any(trecho in e for e in erros), erros


def test_parametros_validos_sem_erros(teste, df):
    assert teste.validar(df, _params()) == []


def test_formulario(teste):
    specs = teste.parametros()
    assert [(s.nome, s.tipo) for s in specs] == [
        ("coluna", "coluna_binaria"),
        ("sucesso", "nivel"),
        ("grupo", "coluna_binaria"),
        ("alternativa", "opcao"),
        ("alfa", "alfa"),
    ]
    assert specs[1].depende_de == "coluna"
    assert teste.comparacao_inicial().titulo_direita == "Fisher exato"
