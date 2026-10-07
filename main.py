"""Ponto de entrada do smartETL: `python main.py`.

No topo só entram Flet e a splash. `main` mostra a tela de abertura e retorna logo — no Flet
0.86.2 o que um `main` síncrono adiciona só chega à janela quando ele termina. O carregamento
pesado (pandas, scipy, statsmodels, via a tela principal) roda em `page.run_thread`, que troca a
splash pela tela principal ao terminar. A splash fica ao menos `DURACAO_MINIMA` segundos.
"""

import logging
import time

import flet as ft

from app.ui import tema
from app.ui.splash import Splash

log = logging.getLogger(__name__)

DURACAO_MINIMA = 1.2  # segundos
TRANSICAO_MS = 450


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
    inicio = time.monotonic()
    configurar_pagina(page)
    raiz = ft.AnimatedSwitcher(
        content=Splash(),
        transition=ft.AnimatedSwitcherTransition.FADE,
        duration=TRANSICAO_MS,
        expand=True,
    )
    page.add(raiz)
    page.run_thread(carregar_aplicativo, page, raiz, inicio)


def carregar_aplicativo(page: ft.Page, raiz: ft.AnimatedSwitcher, inicio: float) -> None:
    """Importa e monta a tela principal (fora da thread da UI) e a troca pela splash."""
    from app.controller import Controller
    from app.state import AppState
    from app.ui.tela_principal import TelaPrincipal

    tela = TelaPrincipal(page)
    controller = Controller(AppState(), tela)
    tela.conectar(controller)
    log.info("Aplicativo carregado em %.2f s", time.monotonic() - inicio)
    restante = DURACAO_MINIMA - (time.monotonic() - inicio)
    if restante > 0:
        time.sleep(restante)
    raiz.content = tela.raiz
    page.update()
    controller.iniciar()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ft.run(main)
