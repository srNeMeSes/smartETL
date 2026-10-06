"""Painel com as abas Parâmetros / Análise / Visualização, acessadas por método (sem índices)."""

import flet as ft

from app.ui import tema
from app.ui.helpers import border_all

ABA_PARAMETROS, ABA_ANALISE, ABA_VISUALIZACAO = range(3)
ROTULOS_ABAS = ("Parâmetros", "Análise", "Visualização")


def mensagem(texto: str, titulo: str | None = None) -> ft.Column:
    """Estado vazio/informativo de uma aba."""
    controles: list[ft.Control] = []
    if titulo:
        controles.append(ft.Text(titulo, size=16, weight=ft.FontWeight.BOLD, color=tema.TEXTO))
    controles.append(
        ft.Row(
            [
                ft.Icon(ft.Icons.INFO_OUTLINE, size=18, color=tema.LARANJA),
                ft.Text(texto, size=14, color=tema.TEXTO),
            ],
            spacing=8,
        )
    )
    return ft.Column(controls=controles, spacing=8)


def processando() -> ft.Row:
    return ft.Row(
        [
            ft.ProgressRing(width=18, height=18, stroke_width=2, color=tema.LARANJA),
            ft.Text("Processando...", size=14, color=tema.TEXTO),
        ],
        spacing=10,
    )


class PainelAbas(ft.Container):
    def __init__(self):
        self._paineis = [ft.Container(padding=20, expand=True) for _ in ROTULOS_ABAS]
        self.tabs = ft.Tabs(
            selected_index=ABA_PARAMETROS,
            length=len(ROTULOS_ABAS),
            expand=True,
            content=ft.Column(
                expand=True,
                controls=[
                    ft.TabBar(
                        tabs=[ft.Tab(label=ft.Text(r, color=tema.TEXTO)) for r in ROTULOS_ABAS]
                    ),
                    ft.TabBarView(expand=True, controls=self._paineis),
                ],
            ),
        )
        super().__init__(
            content=self.tabs,
            expand=True,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            border=border_all(1, tema.BORDA),
            border_radius=tema.RAIO_PEQUENO,
            bgcolor=tema.CARTAO,
        )

    @property
    def parametros(self) -> ft.Control | None:
        return self._paineis[ABA_PARAMETROS].content

    @property
    def analise(self) -> ft.Control | None:
        return self._paineis[ABA_ANALISE].content

    @property
    def visualizacao(self) -> ft.Control | None:
        return self._paineis[ABA_VISUALIZACAO].content

    def definir_parametros(self, controle: ft.Control) -> None:
        self._paineis[ABA_PARAMETROS].content = controle

    def definir_analise(self, controle: ft.Control) -> None:
        self._paineis[ABA_ANALISE].content = controle

    def definir_visualizacao(self, controle: ft.Control) -> None:
        self._paineis[ABA_VISUALIZACAO].content = controle

    def ir_para(self, aba: int) -> None:
        self.tabs.selected_index = aba
