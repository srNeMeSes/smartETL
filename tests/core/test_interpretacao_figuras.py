"""core/interpretacao.py e core/figuras.py."""

import math

import numpy as np
import pytest

from core.figuras import MAX_CLASSES, histograma
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
