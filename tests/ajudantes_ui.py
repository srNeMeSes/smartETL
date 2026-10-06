"""Ajudantes para inspecionar árvores de controles Flet nos testes."""

from collections.abc import Iterator

import flet as ft

_FILHOS = ("content", "controls", "tabs", "columns", "rows", "cells", "label")


def iterar_controles(controle) -> Iterator:
    """Percorre a árvore de controles Flet em profundidade."""
    yield controle
    for atributo in _FILHOS:
        filho = getattr(controle, atributo, None)
        if filho is None or isinstance(filho, str):
            continue
        filhos = filho if isinstance(filho, list) else [filho]
        for item in filhos:
            if hasattr(item, "_c"):  # só controles Flet
                yield from iterar_controles(item)


def textos(controle) -> list[str]:
    """Todos os valores de `ft.Text` dentro de um controle."""
    return [c.value for c in iterar_controles(controle) if isinstance(c, ft.Text)]


def do_tipo(controle, tipo) -> list:
    """Todos os controles de um tipo dentro de um controle."""
    return [c for c in iterar_controles(controle) if isinstance(c, tipo)]
