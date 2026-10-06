"""Pequenos helpers de layout."""

import flet as ft


def pad(
    all: float | None = None,
    horizontal: float | None = None,
    vertical: float | None = None,
    left: float = 0,
    top: float = 0,
    right: float = 0,
    bottom: float = 0,
) -> ft.Padding:
    """Padding uniforme (`all`), simétrico (`horizontal`/`vertical`) ou por lado."""
    if all is not None:
        return ft.Padding(left=all, top=all, right=all, bottom=all)
    if horizontal is not None or vertical is not None:
        h = horizontal or 0
        v = vertical or 0
        return ft.Padding(left=h, top=v, right=h, bottom=v)
    return ft.Padding(left=left, top=top, right=right, bottom=bottom)


def border_all(width: float, color: str) -> ft.Border:
    """Borda igual nos quatro lados."""
    side = ft.BorderSide(width, color)
    return ft.Border(top=side, right=side, bottom=side, left=side)


def border_only(
    top: ft.BorderSide | None = None,
    right: ft.BorderSide | None = None,
    bottom: ft.BorderSide | None = None,
    left: ft.BorderSide | None = None,
) -> ft.Border:
    """Borda só nos lados informados."""
    lados = {"top": top, "right": right, "bottom": bottom, "left": left}
    return ft.Border(**{lado: valor for lado, valor in lados.items() if valor is not None})


def esta_na_pagina(controle: ft.BaseControl) -> bool:
    """Se o controle já foi montado (no Flet 0.86.2, `.page` lança RuntimeError se não foi)."""
    try:
        return controle.page is not None
    except RuntimeError:
        return False
