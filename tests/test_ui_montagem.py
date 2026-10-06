"""Montagem da UI sem janela: sidebar, card de comparação, formulário e página."""

from unittest.mock import MagicMock

import flet as ft
import pytest
from conftest import iterar_controles, textos

import conteiner_parametros as cp
import smartetl_app
from ConteinerTestes import CardComparacaoTestes
from utils import criar_sidebar_testes, testes_hipotese

# --------------------------------------------------------------------------
# Sidebar de testes
# --------------------------------------------------------------------------


@pytest.fixture
def linhas_sidebar():
    sidebar = criar_sidebar_testes(None)  # o parâmetro `page` não é usado
    return sidebar.content.controls


def test_sidebar_tem_todos_os_testes(linhas_sidebar):
    rotulos = [linha.content.controls[1].value for linha in linhas_sidebar]
    assert rotulos == [rotulo for _, rotulo in testes_hipotese]


def test_sidebar_primeiro_teste_marcado(linhas_sidebar):
    primeira, *demais = linhas_sidebar
    assert primeira.content.controls[0].icon == ft.Icons.RADIO_BUTTON_CHECKED
    assert primeira.bgcolor == "#FFF1E6"
    for linha in demais:
        assert linha.content.controls[0].icon == ft.Icons.RADIO_BUTTON_UNCHECKED
        assert linha.bgcolor == "transparent"


def test_sidebar_linhas_clicaveis(linhas_sidebar):
    assert all(linha.on_click is not None for linha in linhas_sidebar)


# --------------------------------------------------------------------------
# CardComparacaoTestes
# --------------------------------------------------------------------------


@pytest.fixture
def card():
    return CardComparacaoTestes()


def test_card_monta_com_padroes(card):
    assert isinstance(card, ft.Container)
    assert len(card._textos_p_esquerda) == 3
    assert len(card._textos_p_direita) == 3
    conteudo = textos(card)
    assert "t Student" in conteudo
    assert "Wilcoxon" in conteudo


def test_card_aceita_titulos_e_linhas(card):
    personalizado = CardComparacaoTestes(
        titulo_esquerda="A",
        titulo_direita="B",
        linhas=card._linhas_padrao()[:1],
    )
    assert len(personalizado._textos_p_esquerda) == 1
    assert {"A", "B"} <= set(textos(personalizado))


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
    assert [t.value for t in card._textos_p_esquerda] == ["0.012", "0.500", "0.001"]
    assert [t.value for t in card._textos_p_direita] == ["0.034", "0.001", "0.900"]
    assert "0.012" in textos(card)


def test_atualizar_p_value_linha(card):
    card.atualizar_p_value_linha(2, esquerda="0.2", atualizar_pagina=False)
    assert card._textos_p_esquerda[2].value == "0.2"
    assert card._textos_p_direita[2].value == "0.001"


@pytest.mark.parametrize("indice", [-1, 3])
def test_atualizar_p_value_linha_indice_invalido(card, indice):
    with pytest.raises(IndexError):
        card.atualizar_p_value_linha(indice, esquerda="x", atualizar_pagina=False)


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="N1: no Flet 0.86.2 `Control.page` lança RuntimeError fora da página; "
    "`self.page is not None` quebra quando atualizar_pagina=True (padrão).",
)
def test_atualizar_p_values_sem_pagina_com_padrao(card):
    card.atualizar_p_values([{"esquerda": "0.1", "direita": "0.2"}])
    assert card._textos_p_esquerda[0].value == "0.1"


# --------------------------------------------------------------------------
# Formulário do teste t (uma amostra) e despacho
# --------------------------------------------------------------------------

COLUNAS = ["id", "users", "qtd"]


def test_teste_t_1am_monta_tres_abas():
    parametro, analise, visual = cp.teste_t_1am(COLUNAS)
    assert all(isinstance(c, ft.Control) for c in (parametro, analise, visual))
    conteudo = textos(parametro)
    assert "Parâmetros do teste" in conteudo


def test_teste_t_1am_dropdown_lista_as_colunas():
    parametro, _, _ = cp.teste_t_1am(COLUNAS)
    dropdowns = [c for c in iterar_controles(parametro) if isinstance(c, ft.Dropdown)]
    variavel = next(d for d in dropdowns if d.label == "Variável")
    alfa = next(d for d in dropdowns if d.label.startswith("Nível de significância"))
    assert [o.key for o in variavel.options] == COLUNAS
    assert [o.key for o in alfa.options] == ["0.01", "0.05", "0.10"]
    assert alfa.value == "0.05"


def test_teste_selecionado_id_desconhecido():
    with pytest.raises(ValueError, match="não está cadastrado"):
        cp.teste_selecionado("nao_existe", COLUNAS)


def _tem_campo_media_hipotetica(controle) -> bool:
    return any(
        isinstance(c, ft.TextField) and c.label == "Média Hipotética"
        for c in iterar_controles(controle)
    )


@pytest.mark.xfail(
    strict=True,
    reason="Problema 3: `globals().get` cai no formulário do teste_t_1am para qualquer teste.",
)
def test_teste_selecionado_nao_reusa_formulario_do_t_1am():
    parametro, _, _ = cp.teste_selecionado("wilcoxon", COLUNAS)
    assert not _tem_campo_media_hipotetica(parametro)


@pytest.mark.xfail(
    strict=True,
    reason="Problema 4: a mesma instância de CardComparacaoTestes está em Parâmetros e Análise.",
)
def test_card_nao_e_compartilhado_entre_abas():
    parametro, analise, _ = cp.teste_t_1am(COLUNAS)
    cards_p = [
        c for c in iterar_controles(parametro) if isinstance(c, CardComparacaoTestes)
    ]
    cards_a = [
        c for c in iterar_controles(analise) if isinstance(c, CardComparacaoTestes)
    ]
    assert not {id(c) for c in cards_p} & {id(c) for c in cards_a}


# --------------------------------------------------------------------------
# Página completa (main) com uma página falsa
# --------------------------------------------------------------------------


@pytest.fixture
def raiz_app():
    page = MagicMock()
    smartetl_app.main(page)
    assert page.add.call_count == 1
    return page, page.add.call_args.args[0]


def test_main_monta_pagina(raiz_app):
    page, raiz = raiz_app
    assert page.title == "smartETL — Processamento de dados"
    assert (page.window.width, page.window.height) == (1440, 900)
    assert (page.window.min_width, page.window.min_height) == (1150, 720)
    (servico,), _ = page.services.append.call_args
    assert isinstance(servico, ft.FilePicker)
    sidebar, _area_principal = raiz.controls
    assert sidebar.width == 260
    conteudo = textos(raiz)
    for esperado in ("Processamento de dados", "Arquivo", "Executar teste"):
        assert esperado in conteudo


def test_main_tem_tres_abas_e_tabela_vazia(raiz_app):
    _, raiz = raiz_app
    rotulos = [
        t.label.value
        for c in iterar_controles(raiz)
        if isinstance(c, ft.TabBar)
        for t in c.tabs
    ]
    assert rotulos == ["Parâmetros", "Análise", "Visualização"]
    tabela = next(c for c in iterar_controles(raiz) if isinstance(c, ft.DataTable))
    assert [col.label.value for col in tabela.columns] == [
        f"column{i}" for i in range(1, 21)
    ]
    assert len(tabela.rows) == 12


def _botao_executar(raiz):
    return next(
        c
        for c in iterar_controles(raiz)
        if isinstance(c, ft.Button) and "Executar teste" in textos(c)
    )


@pytest.mark.xfail(
    strict=True, reason="Problema 2: botão 'Executar teste' sem on_click."
)
def test_botao_executar_tem_on_click(raiz_app):
    _, raiz = raiz_app
    assert _botao_executar(raiz).on_click is not None
