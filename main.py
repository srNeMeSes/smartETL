"""Ponto de entrada do smartETL: `python main.py`."""

import logging

import flet as ft

from app.controller import Controller
from app.state import AppState
from app.ui import tema
from app.ui.tela_principal import TelaPrincipal


def configurar_pagina(page: ft.Page) -> None:
    page.title = "smartETL — Processamento de dados"
    page.bgcolor = tema.FUNDO
    page.padding = 0
    page.theme = ft.Theme(font_family=tema.FONTE)
    page.window.width = 1440
    page.window.height = 900
    page.window.min_width = 1150
    page.window.min_height = 720
    # `Window.center` é assíncrono no Flet 0.86.2: chamar sem await não centraliza.
    page.run_task(page.window.center)


def main(page: ft.Page) -> None:
    configurar_pagina(page)
    tela = TelaPrincipal(page)
    controller = Controller(AppState(), tela)
    tela.conectar(controller)
    page.add(tela.raiz)
    controller.iniciar()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ft.run(main)
