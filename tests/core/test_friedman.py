"""Friedman (com comparações múltiplas): checklist da seção 9 do CLAUDE.md.

Fontes dos valores de referência (independentes do wrapper, que usa scipy):
- Q = [12/(n·k(k+1))·Σ Rⱼ² − 3n(k+1)] / [1 − Σ(t³ − t)/(n(k³ − k))], com postos médios dentro de
  cada linha calculados à mão;
- p pela cauda fechada da qui-quadrado com 2 gl: exp(−Q/2) (k = 3);
- W de Kendall = Q/(n(k − 1));
- comparações: Wilcoxon exato por enumeração (tests/referencias.py) e Holm do statsmodels.
"""

import math

import numpy as np
import pandas as pd
import pytest
from referencias import wilcoxon_exato
from statsmodels.stats.multitest import multipletests

from core.base import ErroValidacao
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.nao_parametricos import TesteFriedman

REL = 1e-9
# 10 alunos × 3 provas; algumas linhas com empate e ordem variada.
P1 = [6.5, 7.0, 5.5, 8.0, 6.0, 7.5, 5.0, 6.8, 7.2, 6.1]
P2 = [7.2, 7.0, 6.4, 7.6, 6.9, 8.1, 5.9, 7.0, 6.9, 6.6]
P3 = [8.1, 8.0, 7.3, 9.2, 6.9, 8.8, 6.6, 6.5, 8.4, 7.5]


@pytest.fixture
def teste():
    return TesteFriedman()


@pytest.fixture
def df():
    return pd.DataFrame({"p1": P1, "p2": P2, "p3": P3, "turma": list("ab" * 5)})


def _params(**extra):
    return {"colunas": ["p1", "p2", "p3"], "comparacoes": False, "alfa": 0.05} | extra


def _postos_linha(linha):
    ordenados = sorted(linha)
    return [(ordenados.index(v) + 1 + len(ordenados) - ordenados[::-1].index(v)) / 2 for v in linha]


def _manual(colunas):
    linhas = list(zip(*colunas, strict=True))
    n, k = len(linhas), len(colunas)
    postos = [_postos_linha(list(linha)) for linha in linhas]
    somas = [sum(p[j] for p in postos) for j in range(k)]
    q = 12 / (n * k * (k + 1)) * sum(r**2 for r in somas) - 3 * n * (k + 1)
    empates = sum(t**3 - t for linha in linhas for t in (list(linha).count(v) for v in set(linha)))
    q /= 1 - empates / (n * (k**3 - k))
    return q, [r / n for r in somas]


# ---------------------------------------------------------------------------
# Valores de referência
# ---------------------------------------------------------------------------


def test_q_p_e_w_contra_formula(teste, df):
    r = teste.executar(df, _params())
    q, medios = _manual([P1, P2, P3])
    e = r.estatisticas
    assert e["q"] == pytest.approx(q, rel=REL)
    assert e["gl"] == 2 and (e["n"], e["k"]) == (10, 3)
    assert r.p_valor == pytest.approx(math.exp(-q / 2), rel=REL)
    assert e["w_kendall"] == pytest.approx(q / (10 * 2), rel=REL)
    medidas = r.tabelas["Medidas"]
    assert medidas["Coluna"].tolist() == ["p1", "p2", "p3"]
    assert medidas["Posto médio"].tolist() == [f"{m:.2f}".replace(".", ",") for m in medios]


def test_quatro_colunas_contra_formula(teste):
    rng = np.random.default_rng(8)
    base = rng.normal(10, 2, 15)
    colunas = [np.round(base + rng.normal(d, 1, 15), 1) for d in (0, 0.5, 1, 1.5)]
    df = pd.DataFrame({f"c{i}": c for i, c in enumerate(colunas)})
    r = teste.executar(df, _params(colunas=["c0", "c1", "c2", "c3"]))
    q, _ = _manual([list(c) for c in colunas])
    assert r.estatisticas["q"] == pytest.approx(q, rel=REL)
    assert r.estatisticas["gl"] == 3


def test_comparacoes_wilcoxon_holm(teste):
    p1 = [6.5, 7.0, 5.5, 8.0, 6.0, 7.5, 5.0, 6.8, 7.2, 6.1]
    p2 = [7.21, 7.83, 6.42, 8.55, 6.97, 8.14, 5.88, 7.06, 7.93, 6.69]
    p3 = [8.12, 8.04, 7.31, 9.23, 7.41, 8.86, 6.63, 7.95, 8.47, 7.58]
    df = pd.DataFrame({"p1": p1, "p2": p2, "p3": p3})
    r = teste.executar(df, _params(comparacoes=True))
    tabela = r.tabelas["Comparações múltiplas (Wilcoxon, Holm)"]
    assert tabela["Comparação"].tolist() == ["'p1' × 'p2'", "'p1' × 'p3'", "'p2' × 'p3'"]
    ps = [wilcoxon_exato(np.array(a) - np.array(b))[0] for a, b in ((p1, p2), (p1, p3), (p2, p3))]
    holm = multipletests(ps, method="holm")[1]
    assert tabela["p ajustado (Holm)"].tolist() == [f"{p:.3f}".replace(".", ",") for p in holm]
    assert r.estatisticas["comparacoes"] == 3


# ---------------------------------------------------------------------------
# Decisão, saída e avisos
# ---------------------------------------------------------------------------


def test_decisao_e_interpretacao(teste, df):
    r = teste.executar(df, _params())
    assert r.decisao == REJEITA_H0
    assert "rejeita-se H₀ (as 3 medidas têm a mesma distribuição)" in r.interpretacao
    assert "Há evidência estatística de diferença entre as medidas 'p1', 'p2', 'p3'." in (
        r.interpretacao
    )
    parecidos = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6], "b": [2, 1, 4, 3, 6, 5], "c": [1.5, 2.5, 3.5, 3.6, 5.5, 5.4]}
    )
    r = teste.executar(parecidos, _params(colunas=["a", "b", "c"]))
    assert r.decisao == NAO_REJEITA_H0
    assert "Não há evidência suficiente" in r.interpretacao


def test_saida_figura_e_sem_card(teste, df):
    r = teste.executar(df, _params())
    (figura,) = r.figuras
    assert figura.tipo == "boxplot"
    assert [g["rotulo"] for g in figura.dados["grupos"]] == ["p1", "p2", "p3"]
    assert "Comparações múltiplas (Wilcoxon, Holm)" not in r.tabelas  # desligado por padrão
    assert r.comparacao is None and teste.comparacao_inicial() is None


def test_avisos(teste, df):
    incompleto = pd.concat([df, pd.DataFrame({"p1": [1.0], "p2": [np.nan], "p3": [2.0]})])
    r = teste.executar(incompleto, _params())
    textos = " ".join(r.avisos)
    assert "1 linha(s) com valor ausente em alguma das colunas foram descartadas." in textos
    assert "empates dentro de linhas" in textos
    pequeno = teste.executar(df.head(5), _params())
    assert any("Poucas linhas completas (n = 5 < 10)" in a for a in pequeno.avisos)


def test_comparacoes_nao_exibidas_sem_rejeicao(teste):
    parecidos = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6], "b": [2, 1, 4, 3, 6, 5], "c": [1.5, 2.5, 3.5, 3.6, 5.5, 5.4]}
    )
    r = teste.executar(parecidos, _params(colunas=["a", "b", "c"], comparacoes=True))
    assert "Comparações múltiplas (Wilcoxon, Holm)" not in r.tabelas
    assert any("Comparações múltiplas não exibidas" in a for a in r.avisos)


# ---------------------------------------------------------------------------
# Entradas inválidas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("colunas", "trecho"),
    [
        ([], "Selecione ao menos 3 colunas (há 0)."),
        (["p1", "p2"], "Selecione ao menos 3 colunas (há 2)."),
        (["p1", "p1", "p2"], "As colunas escolhidas devem ser diferentes."),
        (["p1", "p2", "nao_existe"], "A coluna 'nao_existe' não existe no arquivo."),
        (["p1", "p2", "turma"], "A coluna 'turma' não é numérica."),
    ],
)
def test_validacao_colunas(teste, df, colunas, trecho):
    assert trecho in teste.validar(df, _params(colunas=colunas))
    with pytest.raises(ErroValidacao):
        teste.executar(df, _params(colunas=colunas))


def test_validacao_dados(teste):
    iguais = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0], "c": [1.0, 2.0]})
    assert any(
        "medidas são iguais entre si" in e
        for e in teste.validar(iguais, _params(colunas=["a", "b", "c"]))
    )
    uma = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 1.0], "c": [3.0, 1.0]})
    assert any(
        "ao menos 2 linhas completas" in e
        for e in teste.validar(uma, _params(colunas=["a", "b", "c"]))
    )
    assert "O nível de significância (α) deve estar entre 0 e 1." in teste.validar(
        pd.DataFrame({"p1": P1, "p2": P2, "p3": P3}), _params(alfa=2)
    )


def test_formulario(teste, df):
    specs = teste.parametros()
    assert [(s.nome, s.tipo, s.padrao) for s in specs] == [
        ("colunas", "multi_coluna", None),
        ("comparacoes", "booleano", False),
        ("alfa", "alfa", 0.05),
    ]
    assert teste.validar(df, _params()) == []
