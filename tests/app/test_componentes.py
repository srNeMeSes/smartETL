"""Campos de formulário e card de comparação."""

import flet as ft
import pytest
from ajudantes_ui import textos

from app.ui import tema
from app.ui.componentes import campos
from app.ui.componentes.card_comparacao import (
    SEM_VALOR,
    CardComparacaoTestes,
    formatar_p_valor,
)
from core.base import ComparacaoPValores

# --------------------------------------------------------------------------
# Campos
# --------------------------------------------------------------------------


def _assert_estilo_unico(campo):
    assert campo.width == campos.LARGURA_CAMPO
    assert campo.height == campos.ALTURA_CAMPO
    assert campo.border_width == 2
    assert campo.border_radius == tema.RAIO_PEQUENO
    assert campo.focused_border_color == tema.LARANJA
    assert campo.label_style.weight == ft.FontWeight.BOLD


def test_dropdown_campo():
    dd = campos.dropdown_campo("Variável", ["a", "b"], valor="b")
    _assert_estilo_unico(dd)
    assert dd.label == "Variável"
    assert [o.key for o in dd.options] == ["a", "b"]
    assert dd.value == "b"
    assert dd.leading_icon == ft.Icons.TUNE


def test_campo_texto():
    campo = campos.campo_texto("Média Hipotética")
    _assert_estilo_unico(campo)
    assert campo.prefix_icon == ft.Icons.ANALYTICS
    assert campo.cursor_color == tema.LARANJA


def test_caixa_selecao():
    caixa = campos.caixa_selecao("Com interação", valor=True)
    assert caixa.value is True
    assert caixa.label == "Com interação"


# --------------------------------------------------------------------------
# Card
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "esperado"),
    [(None, "—"), (0.0004, "< 0,001"), (0.001, "0,001"), (0.0456, "0,046"), (1.0, "1,000")],
)
def test_formatar_p_valor(p, esperado):
    assert formatar_p_valor(p) == esperado


@pytest.fixture
def card():
    return CardComparacaoTestes()


def test_card_padrao_sem_p_valores_falsos(card):
    assert len(card._textos_p_esquerda) == 3
    assert [t.value for t in card._textos_p_esquerda] == [SEM_VALOR] * 3
    assert [t.value for t in card._textos_p_direita] == [SEM_VALOR] * 3
    assert {"t Student", "Wilcoxon"} <= set(textos(card))


def test_card_de_comparacao_usa_hipoteses_do_teste():
    comp = ComparacaoPValores("t Student", "Wilcoxon", ["μ ≠ μ₀", "μ > μ₀"], [(None, None)] * 2)
    card = CardComparacaoTestes.de_comparacao(comp)
    conteudo = textos(card)
    assert "Hₐ:  μ ≠ μ₀" in conteudo
    assert "Hₐ:  μ > μ₀" in conteudo
    assert len(card._textos_p_esquerda) == 2


def test_card_aplicar_formata_p_valores(card):
    comp = ComparacaoPValores(
        "t", "W", ["a", "b", "c"], [(0.012, 0.0001), (None, 0.5), (1.0, None)]
    )
    card.aplicar(comp)  # fora da página: não pode quebrar
    assert [t.value for t in card._textos_p_esquerda] == ["0,012", SEM_VALOR, "1,000"]
    assert [t.value for t in card._textos_p_direita] == ["< 0,001", "0,500", SEM_VALOR]


def test_atualizar_p_values_altera_textos(card):
    card.atualizar_p_values(
        [
            {"esquerda": "0.012", "direita": "0.034"},
            {"esquerda": "0.500", "direita": None},  # None preserva o valor atual
            {"direita": "0.900"},
            {"esquerda": "9", "direita": "9"},  # linha extra é ignorada
        ],
        atualizar_pagina=False,
    )
    assert [t.value for t in card._textos_p_esquerda] == ["0.012", "0.500", SEM_VALOR]
    assert [t.value for t in card._textos_p_direita] == ["0.034", SEM_VALOR, "0.900"]


def test_atualizar_p_values_sem_pagina_com_padrao(card):
    # Antes da Fase 1 isto lançava RuntimeError (Control.page fora da página).
    card.atualizar_p_values([{"esquerda": "0.1", "direita": "0.2"}])
    card.atualizar_p_value_linha(1, direita="0.3")
    assert card._textos_p_esquerda[0].value == "0.1"
    assert card._textos_p_direita[1].value == "0.3"


def test_atualiza_quando_montado(card, monkeypatch):
    chamadas = []
    monkeypatch.setattr(card, "_esta_na_pagina", lambda: True)
    monkeypatch.setattr(card, "update", lambda: chamadas.append(1))
    card.atualizar_p_value_linha(0, esquerda="x")
    card.atualizar_p_value_linha(0, esquerda="y", atualizar_pagina=False)
    assert chamadas == [1]


@pytest.mark.parametrize("indice", [-1, 3])
def test_atualizar_p_value_linha_indice_invalido(card, indice):
    with pytest.raises(IndexError):
        card.atualizar_p_value_linha(indice, esquerda="x")
