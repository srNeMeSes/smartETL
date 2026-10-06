"""core/interpretacao.py e core/figuras.py."""

import math

import numpy as np
import pytest

from core.figuras import MAX_CLASSES, boxplot, histograma, resumo_boxplot
from core.interpretacao import (
    NAO_REJEITA_H0,
    REJEITA_H0,
    decidir,
    formatar_alfa,
    formatar_numero,
    formatar_p_valor,
    interpretar,
)


@pytest.mark.parametrize(
    ("valor", "casas", "esperado"),
    [
        (1234.5, 2, "1.234,50"),
        (-0.123456, 4, "-0,1235"),
        (3, 0, "3"),
        (None, 2, "—"),
        (math.nan, 2, "—"),
        (math.inf, 2, "+∞"),
        (-math.inf, 2, "-∞"),
    ],
)
def test_formatar_numero(valor, casas, esperado):
    assert formatar_numero(valor, casas) == esperado


@pytest.mark.parametrize(
    ("p", "esperado"),
    [(None, "—"), (math.nan, "—"), (0.0004, "< 0,001"), (0.001, "0,001"), (0.0456, "0,046")],
)
def test_formatar_p_valor(p, esperado):
    assert formatar_p_valor(p) == esperado


def test_formatar_alfa():
    assert [formatar_alfa(a) for a in (0.01, 0.05, 0.1)] == ["0,01", "0,05", "0,10"]


@pytest.mark.parametrize(
    ("p", "alfa", "esperado"),
    [(0.01, 0.05, REJEITA_H0), (0.05, 0.05, REJEITA_H0), (0.0501, 0.05, NAO_REJEITA_H0)],
)
def test_decidir_p_menor_ou_igual_a_alfa(p, alfa, esperado):
    assert decidir(p, alfa) == esperado


def test_interpretar_rejeita():
    texto = interpretar(0.012, 0.05, "μ = 5", "μ ≠ 5", "Há evidência.", "Não há evidência.")
    assert texto == (
        "Com α = 0,05, o p-valor (0,012) é menor ou igual a α: rejeita-se H₀ (μ = 5) "
        "em favor de H₁ (μ ≠ 5). Há evidência."
    )


def test_interpretar_nao_rejeita():
    texto = interpretar(0.3, 0.01, "μ = 5", "μ > 5", "Há evidência.", "Não há evidência.")
    assert texto == (
        "Com α = 0,01, o p-valor (0,300) é maior que α: não se rejeita H₀ (μ = 5). "
        "Não há evidência."
    )


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------


def test_histograma_conta_todos_os_valores():
    x = np.random.default_rng(42).normal(10, 2, 200)
    fig = histograma(x, "Distribuição de 'x'", "x", [("x̄", float(x.mean()), "destaque")])
    assert fig.tipo == "histograma"
    assert fig.titulo == "Distribuição de 'x'"
    dados = fig.dados
    assert sum(dados["contagens"]) == 200
    assert len(dados["bordas"]) == len(dados["contagens"]) + 1
    assert dados["bordas"] == sorted(dados["bordas"])
    assert dados["referencias"] == [{"rotulo": "x̄", "valor": float(x.mean()), "estilo": "destaque"}]
    assert dados["rotulo_x"] == "x"


def test_histograma_inclui_referencia_fora_dos_dados():
    fig = histograma([1.0, 2.0, 3.0], "t", "x", [("μ₀", 10.0, "tracejado")])
    assert fig.dados["bordas"][0] == 1.0
    assert fig.dados["bordas"][-1] == 10.0
    assert sum(fig.dados["contagens"]) == 3


def test_histograma_limita_classes_e_ignora_nao_finitos():
    x = np.concatenate([np.arange(10_000, dtype=float), [np.nan, np.inf]])
    fig = histograma(x, "t", "x")
    assert len(fig.dados["contagens"]) <= MAX_CLASSES
    assert sum(fig.dados["contagens"]) == 10_000


def test_histograma_valor_unico_e_vazio():
    fig = histograma([5.0, 5.0], "t", "x")
    assert sum(fig.dados["contagens"]) == 2
    with pytest.raises(ValueError):
        histograma([np.nan], "t", "x")


def test_resumo_boxplot_tukey():
    # Quartis com interpolação linear (numpy): Q1 = 2, Q3 = 4, IQR = 2 → cercas em -1 e 7.
    r = resumo_boxplot([1, 2, 3, 4, 100])
    assert (r["q1"], r["mediana"], r["q3"]) == (2.0, 3.0, 4.0)
    assert (r["bigode_inf"], r["bigode_sup"]) == (1.0, 4.0)
    assert r["outliers"] == [100.0]
    assert r["media"] == 22.0 and r["n"] == 5


def test_resumo_boxplot_sem_outliers_e_vazio():
    r = resumo_boxplot([1.0, 2.0, np.nan, 3.0])
    assert r["n"] == 3 and r["outliers"] == []
    assert (r["bigode_inf"], r["bigode_sup"]) == (1.0, 3.0)
    with pytest.raises(ValueError):
        resumo_boxplot([np.nan])


def test_boxplot_figura():
    fig = boxplot([("A", [1, 2, 3]), ("B", [4, 5, 6, 7])], "'y' por 'g'", "y")
    assert fig.tipo == "boxplot" and fig.titulo == "'y' por 'g'"
    assert fig.dados["rotulo_y"] == "y"
    assert [(g["rotulo"], g["n"], g["mediana"]) for g in fig.dados["grupos"]] == [
        ("A", 3, 2.0),
        ("B", 4, 5.5),
    ]
