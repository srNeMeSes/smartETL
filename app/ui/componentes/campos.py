"""Campos de formulário com o estilo único do smartETL."""

from collections.abc import Sequence

import flet as ft

from app.ui import tema

LARGURA_CAMPO = 300
ALTURA_CAMPO = 58


def _estilo_rotulo() -> ft.TextStyle:
    return ft.TextStyle(color=tema.CAMPO_ROTULO, weight=ft.FontWeight.BOLD, size=14)


def _padding_interno() -> ft.Padding:
    return ft.Padding(left=14, right=14, top=8, bottom=8)


def _opcao(chave: str) -> ft.dropdown.Option:
    return ft.dropdown.Option(chave, style=ft.ButtonStyle(color=tema.CAMPO_TEXTO))


def definir_opcoes(dropdown: ft.Dropdown, opcoes: Sequence[str]) -> None:
    """Troca as opções de um dropdown já criado, mantendo o estilo."""
    dropdown.options = [_opcao(str(o)) for o in opcoes]


def dropdown_campo(
    rotulo: str,
    opcoes: Sequence[str],
    valor: str | None = None,
    icone: str = ft.Icons.TUNE,
    largura: float = LARGURA_CAMPO,
) -> ft.Dropdown:
    """Dropdown estilizado; o valor escolhido fica em `.value`."""
    return ft.Dropdown(
        label=rotulo,
        color=tema.CAMPO_TEXTO,
        text_size=16,
        text_style=ft.TextStyle(weight=ft.FontWeight.W_600, color=tema.CAMPO_TEXTO),
        label_style=_estilo_rotulo(),
        leading_icon=icone,
        options=[_opcao(str(o)) for o in opcoes],
        value=valor,
        width=largura,
        height=ALTURA_CAMPO,
        bgcolor=tema.CARTAO,
        border_width=2,
        border_color=tema.CAMPO_BORDA,
        border_radius=tema.RAIO_PEQUENO,
        focused_border_width=3,
        focused_border_color=tema.LARANJA,
        content_padding=_padding_interno(),
    )


def campo_texto(
    rotulo: str,
    valor: str | None = None,
    icone: str = ft.Icons.ANALYTICS,
    largura: float = LARGURA_CAMPO,
) -> ft.TextField:
    """Campo de texto estilizado; o texto digitado fica em `.value`."""
    return ft.TextField(
        label=rotulo,
        value=valor,
        prefix_icon=icone,
        color=tema.CAMPO_TEXTO,
        text_size=16,
        text_style=ft.TextStyle(weight=ft.FontWeight.W_600),
        label_style=_estilo_rotulo(),
        bgcolor=tema.CARTAO,
        border_width=2,
        border_color=tema.CAMPO_BORDA,
        border_radius=tema.RAIO_PEQUENO,
        focused_border_width=3,
        focused_border_color=tema.LARANJA,
        cursor_color=tema.LARANJA,
        width=largura,
        height=ALTURA_CAMPO,
        content_padding=_padding_interno(),
    )


def caixa_selecao(rotulo: str, valor: bool = False) -> ft.Checkbox:
    """Checkbox com as cores do tema."""
    return ft.Checkbox(
        label=rotulo,
        value=valor,
        active_color=tema.LARANJA,
        label_style=ft.TextStyle(color=tema.CAMPO_TEXTO, size=14),
    )
