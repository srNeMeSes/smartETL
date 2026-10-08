"""Ponto de entrada do smartETL: `python main.py`.

No topo só entram Flet e a splash. `main` mostra a tela de abertura e retorna logo — no Flet
0.86.2 o que um `main` síncrono adiciona só chega à janela quando ele termina. O carregamento
pesado (pandas, scipy, statsmodels, via a tela principal) roda em `page.run_thread`, que troca a
splash pela tela principal ao terminar. A splash fica ao menos `DURACAO_MINIMA` segundos.
"""

import logging
import os
import sys
import threading
import time
from pathlib import Path

import flet as ft

from app.ui import tema
from app.ui.splash import Splash

log = logging.getLogger(__name__)

DURACAO_MINIMA = 1.2  # segundos
TRANSICAO_MS = 450
EMPACOTADO = getattr(sys, "frozen", False)  # executável gerado pelo `flet pack` (PyInstaller)


def recurso(caminho: str) -> Path:
    """Arquivo do projeto (ex.: o ícone), dentro do executável ou na pasta do código."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / caminho


ICONE = recurso("assets/icon2.ico")


def configurar_log() -> None:
    """No executável não há console: o log vai para %LOCALAPPDATA%/smartETL/smartetl.log
    (é o "Detalhes no log" das mensagens de erro); rodando do código, para o terminal."""
    formato = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    # O fontTools (usado pelo PDF) registra cada etapa do subconjunto de fontes em INFO.
    for ruidoso in ("fontTools", "matplotlib"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    if not EMPACOTADO:
        logging.basicConfig(level=logging.INFO, format=formato)
        return
    pasta = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "smartETL"
    pasta.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format=formato, filename=pasta / "smartetl.log", encoding="utf-8"
    )


def configurar_pagina(page: ft.Page) -> None:
    page.title = "smartETL — Processamento de dados"
    if ICONE.exists():
        page.window.icon = str(ICONE)  # canto superior esquerdo e barra de tarefas (Windows)
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
    identificar_na_barra_de_tarefas(page.title)


def identificar_na_barra_de_tarefas(titulo: str) -> threading.Thread:
    """Nome "smartETL", ícone e "Fixar na barra de tarefas" corretos no Windows (a janela é do
    cliente Flet; ver app/identidade_windows.py). Em thread própria: não atrasa a abertura."""
    from app import identidade_windows

    dados = identidade_windows.identidade(
        EMPACOTADO, sys.executable, Path(__file__).resolve(), ICONE
    )
    tarefa = threading.Thread(target=identidade_windows.aplicar, args=(titulo, dados), daemon=True)
    tarefa.start()
    return tarefa


if __name__ == "__main__":
    configurar_log()
    ft.run(main)
