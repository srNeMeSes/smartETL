"""Teste t (duas amostras): checklist da seção 9 do CLAUDE.md.

Fontes dos valores de referência (independentes do wrapper em core/testes/medias.py):
- estatística t e gl (Welch–Satterthwaite e pooled), d de Cohen: fórmulas manuais com numpy;
- p-valores e IC da diferença: statsmodels (`ttest_ind`, `CompareMeans.tconfint_diff`);
- p-valores exatos do Mann-Whitney: enumeração das C(11, 5) = 462 divisões dos postos entre os
  grupos (amostras sem empates).
"""

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW, ttest_ind

from core.base import ErroValidacao
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.medias import ALTERNATIVAS_2, VARIANCIAS, TesteT2Amostras, rotulo_nivel

A = [12.1, 14.3, 11.8, 15.2, 13.7]
B = [16.4, 15.9, 17.2, 14.8, 18.1, 16.7]
STATSMODELS = {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}
USEVAR = {"Diferentes (Welch)": "unequal", "Iguais (pooled)": "pooled"}
REL = 1e-6


@pytest.fixture
def teste():
    return TesteT2Amostras()


@pytest.fixture
def df():
    # Grupo "B" aparece primeiro nos dados: o grupo 1 continua sendo "A" (ordem crescente).
    return pd.DataFrame({"y": B + A, "turma": ["B"] * len(B) + ["A"] * len(A)})


def _params(**extra):
    return {
        "coluna": "y",
        "grupo": "turma",
        "variancias": "Diferentes (Welch)",
        "alternativa": "μ₁ ≠ μ₂",
        "alfa": 0.05,
    } | extra


def _rotulo(alternativa_scipy: str) -> str:
    return next(r for r, a in ALTERNATIVAS_2.items() if a == alternativa_scipy)


def _mann_whitney_exato(x, y):
    """p-valores exatos de U₁ por enumeração: (bilateral, maior, menor)."""
    postos = rankdata(np.concatenate([x, y]))
    n1 = len(x)
    u_obs = postos[:n1].sum() - n1 * (n1 + 1) / 2
    distribuicao = np.array(
        [
            sum(postos[list(idx)]) - n1 * (n1 + 1) / 2
            for idx in itertools.combinations(range(len(postos)), n1)
        ]
    )
    maior = np.mean(distribuicao >= u_obs - 1e-9)
    menor = np.mean(distribuicao <= u_obs + 1e-9)
    return min(1.0, 2 * min(maior, menor)), maior, menor


# ---------------------------------------------------------------------------
# Valores de referência
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variancias", list(VARIANCIAS))
@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_t_contra_formula_manual_e_statsmodels(teste, df, variancias, alternativa):
    r = teste.executar(df, _params(variancias=variancias, alternativa=_rotulo(alternativa)))
    x, y = np.array(A), np.array(B)
    n1, n2 = len(x), len(y)
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    if variancias == "Iguais (pooled)":
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        t_manual = (x.mean() - y.mean()) / math.sqrt(sp2 * (1 / n1 + 1 / n2))
        gl_manual = n1 + n2 - 2
    else:
        se2 = v1 / n1 + v2 / n2
        t_manual = (x.mean() - y.mean()) / math.sqrt(se2)
        gl_manual = se2**2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    t_sm, p_sm, gl_sm = ttest_ind(
        x, y, alternative=STATSMODELS[alternativa], usevar=USEVAR[variancias]
    )
    e = r.estatisticas
    assert e["t"] == pytest.approx(t_manual, rel=REL)
    assert e["t"] == pytest.approx(t_sm, rel=REL)
    assert e["gl"] == pytest.approx(gl_manual, rel=REL)
    assert e["gl"] == pytest.approx(gl_sm, rel=REL)
    assert r.p_valor == pytest.approx(p_sm, rel=REL)
    assert e["diferenca"] == pytest.approx(x.mean() - y.mean(), rel=REL)


@pytest.mark.parametrize("variancias", list(VARIANCIAS))
@pytest.mark.parametrize("alfa", [0.01, 0.05, 0.10])
@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_ic_da_diferenca_contra_statsmodels(teste, df, variancias, alfa, alternativa):
    r = teste.executar(
        df, _params(variancias=variancias, alternativa=_rotulo(alternativa), alfa=alfa)
    )
    baixo, alto = CompareMeans(DescrStatsW(np.array(A)), DescrStatsW(np.array(B))).tconfint_diff(
        alpha=alfa, alternative=STATSMODELS[alternativa], usevar=USEVAR[variancias]
    )
    for obtido, esperado in (
        (r.estatisticas["ic_inferior"], baixo),
        (r.estatisticas["ic_superior"], alto),
    ):
        if math.isinf(esperado):
            assert obtido == esperado
        else:
            assert obtido == pytest.approx(esperado, rel=REL)


def test_d_de_cohen_com_desvio_combinado(teste, df):
    r = teste.executar(df, _params())
    x, y = np.array(A), np.array(B)
    sp = math.sqrt(
        ((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2)
    )
    assert r.estatisticas["d_cohen"] == pytest.approx((x.mean() - y.mean()) / sp, rel=REL)


def test_mann_whitney_exato_por_enumeracao(teste, df):
    r = teste.executar(df, _params())
    esperado = _mann_whitney_exato(np.array(A), np.array(B))
    assert [p for _, p in r.comparacao.linhas] == pytest.approx(list(esperado), rel=REL)
    assert r.estatisticas["u"] == 1.0  # só 14,8 (B) fica abaixo de 15,2 (A)


def test_comparacao_tres_alternativas(teste, df):
    r = teste.executar(df, _params(alternativa="μ₁ < μ₂"))
    comp = r.comparacao
    assert (comp.titulo_esquerda, comp.titulo_direita) == ("t Student", "Mann-Whitney")
    assert comp.hipoteses == ["μ₁ ≠ μ₂", "μ₁ > μ₂", "μ₁ < μ₂"]
    x, y = np.array(A), np.array(B)
    for (p_t, _), alt in zip(comp.linhas, ["two-sided", "larger", "smaller"], strict=True):
        assert p_t == pytest.approx(ttest_ind(x, y, alternative=alt, usevar="unequal")[1], rel=REL)
    assert comp.linhas[2][0] == pytest.approx(r.p_valor, rel=REL)


def test_amostras_maiores_contra_statsmodels(teste):
    rng = np.random.default_rng(11)
    x, y = rng.normal(10, 2, 45), rng.normal(11, 3, 38)
    df = pd.DataFrame({"y": np.concatenate([x, y]), "g": [0] * 45 + [1] * 38})
    for variancias in VARIANCIAS:
        r = teste.executar(df, _params(grupo="g", variancias=variancias))
        _, p_sm, _ = ttest_ind(x, y, usevar=USEVAR[variancias])
        assert r.p_valor == pytest.approx(p_sm, rel=REL)
    assert not any("Amostra pequena" in a for a in r.avisos)


# ---------------------------------------------------------------------------
# Grupos, decisão, interpretação e saída
# ---------------------------------------------------------------------------


def test_grupo_1_e_o_primeiro_nivel_em_ordem_crescente(teste, df):
    r = teste.executar(df, _params())
    assert r.estatisticas["n1"] == len(A) and r.estatisticas["n2"] == len(B)
    assert r.estatisticas["media1"] == pytest.approx(np.mean(A), rel=REL)
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Grupo 1"] == "'A' (n = 5)"
    assert medidas["Grupo 2"] == "'B' (n = 6)"


def test_niveis_numericos(teste):
    df = pd.DataFrame({"y": A + B, "g": [2.0] * len(A) + [1.0] * len(B)})
    r = teste.executar(df, _params(grupo="g"))
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Grupo 1"] == "'1' (n = 6)"  # 1 < 2; rótulo sem ".0"
    assert [rotulo_nivel(v) for v in (1.0, 2.5, "x")] == ["1", "2.5", "x"]


def test_rejeita_h0_e_interpretacao(teste, df):
    r = teste.executar(df, _params())
    assert r.p_valor < 0.01 and r.decisao == REJEITA_H0
    assert "rejeita-se H₀ (μ₁ = μ₂) em favor de H₁ (μ₁ ≠ μ₂)" in r.interpretacao
    assert (
        "Há evidência estatística de que a média de 'y' no grupo 'A' é diferente da média "
        "no grupo 'B'."
    ) in r.interpretacao


@pytest.mark.parametrize(
    ("rotulo", "decisao", "texto"),
    [("μ₁ < μ₂", REJEITA_H0, "é menor que a média"), ("μ₁ > μ₂", NAO_REJEITA_H0, "maior que a")],
)
def test_alternativas_unilaterais(teste, df, rotulo, decisao, texto):
    r = teste.executar(df, _params(alternativa=rotulo))
    assert r.decisao == decisao
    assert texto in r.interpretacao


def test_nao_rejeita_com_grupos_parecidos(teste):
    df = pd.DataFrame({"y": A + [v + 0.1 for v in A], "g": ["a"] * 5 + ["b"] * 5})
    r = teste.executar(df, _params(grupo="g"))
    assert r.decisao == NAO_REJEITA_H0
    assert "Não há evidência suficiente" in r.interpretacao


def test_resultado_completo(teste, df):
    r = teste.executar(df, _params(variancias="Iguais (pooled)", alfa=0.10))
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Variâncias"] == "Iguais (pooled)"
    assert medidas["Graus de liberdade"] == "9"
    assert "IC 90% para μ₁ − μ₂" in medidas
    assert "U de Mann-Whitney" in medidas and "p-valor (Mann-Whitney)" in medidas
    (figura,) = r.figuras
    assert figura.tipo == "boxplot"
    assert figura.titulo == "'y' por 'turma'"
    assert [(g["rotulo"], g["n"]) for g in figura.dados["grupos"]] == [("A", 5), ("B", 6)]
    assert any("Amostra pequena em 'A' (n = 5) e 'B' (n = 6)" in a for a in r.avisos)


def test_welch_gl_fracionario_na_tabela(teste, df):
    medidas = dict(teste.executar(df, _params()).tabelas["Resumo"].itertuples(index=False))
    assert "," in medidas["Graus de liberdade"]


def test_ic_unilateral_rotulado(teste, df):
    r = teste.executar(df, _params(alternativa="μ₁ > μ₂"))
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["IC 95% para μ₁ − μ₂ (unilateral)"].endswith("+∞]")


# ---------------------------------------------------------------------------
# Casos de borda e entradas inválidas
# ---------------------------------------------------------------------------


def test_linhas_com_ausentes_descartadas(teste):
    df = pd.DataFrame({"y": [*A, *B, np.nan, 3.0], "turma": ["A"] * 5 + ["B"] * 6 + ["A", None]})
    r = teste.executar(df, _params())
    assert (r.estatisticas["n1"], r.estatisticas["n2"]) == (5, 6)
    assert "2 linha(s) com valor ou grupo ausente foram descartadas." in r.avisos


def test_aviso_pooled_com_variancias_muito_diferentes(teste):
    df = pd.DataFrame({"y": [10, 11, 9, 10.5, 1, 30, 15, 22], "g": list("aaaabbbb")})
    r = teste.executar(df, _params(grupo="g", variancias="Iguais (pooled)"))
    assert any("suposição de variâncias iguais é duvidosa" in a for a in r.avisos)
    r = teste.executar(df, _params(grupo="g"))
    assert not any("duvidosa" in a for a in r.avisos)


def test_um_grupo_com_variancia_zero_funciona_no_welch(teste):
    df = pd.DataFrame({"y": [5, 5, 5, 6, 8, 9], "g": list("aaabbb")})
    assert teste.validar(df, _params(grupo="g")) == []
    r = teste.executar(df, _params(grupo="g"))
    assert math.isfinite(r.estatisticas["t"])
    assert any("Todos os valores do grupo 'a' são iguais" in a for a in r.avisos)


@pytest.mark.parametrize(
    ("params", "mensagem"),
    [
        ({"coluna": None}, "Selecione a variável."),
        ({"grupo": None}, "Selecione o grupo."),
        ({"grupo": "nao_existe"}, "A coluna 'nao_existe' não existe no arquivo."),
        ({"coluna": "turma"}, "A coluna 'turma' não é numérica."),
        ({"grupo": "y"}, "A variável e o grupo devem ser colunas diferentes."),
        ({"variancias": "x"}, "Escolha a opção de variâncias."),
        ({"alternativa": "μ ≠ μ₀"}, "Escolha uma hipótese alternativa válida."),
        ({"alfa": 1.5}, "O nível de significância (α) deve estar entre 0 e 1."),
    ],
)
def test_validacao_parametros(teste, df, params, mensagem):
    assert mensagem in teste.validar(df, _params(**params))
    with pytest.raises(ErroValidacao):
        teste.executar(df, _params(**params))


@pytest.mark.parametrize(
    ("y", "g", "trecho"),
    [
        ([1, 2, 3, 4, 5, 6], list("aabbcc"), "exatamente 2 níveis com dados válidos (tem 3)"),
        ([1, 2, 3], list("aaa"), "(tem 1)"),
        ([1, 2, 3, 4], ["a", "b", "b", "b"], "O grupo 'a' tem 1 observação(ões) válida(s)"),
        ([1, 1, 2, 2], list("aabb"), "variância zero"),
        ([1, np.inf, 2, 3], list("aabb"), "valores infinitos"),
    ],
)
def test_validacao_dados(teste, y, g, trecho):
    erros = teste.validar(pd.DataFrame({"y": y, "turma": g}), _params())
    assert any(trecho in e for e in erros), erros


def test_parametros_validos_sem_erros(teste, df):
    assert teste.validar(df, _params()) == []


def test_formulario(teste):
    specs = teste.parametros()
    assert [(s.nome, s.tipo) for s in specs] == [
        ("coluna", "coluna_numerica"),
        ("grupo", "coluna_binaria"),
        ("variancias", "opcao"),
        ("alternativa", "opcao"),
        ("alfa", "alfa"),
    ]
    assert specs[2].padrao == "Diferentes (Welch)"
    assert specs[3].padrao == "μ₁ ≠ μ₂"
    assert teste.comparacao_inicial().titulo_direita == "Mann-Whitney"
