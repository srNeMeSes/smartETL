"""Abas Parâmetros / Análise / Visualização (+ Simulação, só quando o teste a oferece).

Acesso por método, sem índices na árvore de controles."""

import flet as ft

from app.ui import tema
from app.ui.helpers import border_all, esta_na_pagina

ABA_PARAMETROS, ABA_ANALISE, ABA_VISUALIZACAO, ABA_SIMULACAO = range(4)
ROTULOS_ABAS = ("Parâmetros", "Análise", "Visualização")
ROTULO_SIMULACAO = "Simulação"


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
        self._paineis = [ft.Container(padding=20, expand=True) for _ in range(4)]
        self._barra = ft.TabBar(tabs=[], indicator_color=tema.LARANJA)  # linha da aba ativa
        self._vista = ft.TabBarView(expand=True, controls=[])
        self.tabs = ft.Tabs(
            selected_index=ABA_PARAMETROS,
            length=len(ROTULOS_ABAS),
            expand=True,
            content=ft.Column(expand=True, controls=[self._barra, self._vista]),
        )
        self._montar_abas(com_simulacao=False)
        # Ação na mesma linha das abas, à direita (ex.: "Exportar PDF"); sobreposta à barra
        # para a linha divisória continuar ocupando a largura toda.
        self._acao = ft.Container(right=20, top=5)  # alinhado ao padding das abas
        super().__init__(
            content=ft.Stack([self.tabs, self._acao], fit=ft.StackFit.EXPAND, expand=True),
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

    @property
    def acao(self) -> ft.Control | None:
        return self._acao.content

    def definir_acao(self, controle: ft.Control | None) -> None:
        """Controle exibido à direita das abas, ou nenhum (None)."""
        self._acao.content = controle

    @property
    def rotulos(self) -> list[str]:
        """Rótulos das abas visíveis, na ordem."""
        return [*ROTULOS_ABAS, ROTULO_SIMULACAO][: self.tabs.length]

    @property
    def simulacao(self) -> ft.Control | None:
        return self._paineis[ABA_SIMULACAO].content if self.tabs.length > ABA_SIMULACAO else None

    def definir_simulacao(self, controle: ft.Control | None) -> None:
        """Mostra a aba Simulação com `controle`, ou a esconde (None)."""
        self._paineis[ABA_SIMULACAO].content = controle
        self._montar_abas(com_simulacao=controle is not None)

    def _montar_abas(self, com_simulacao: bool) -> None:
        rotulos = [*ROTULOS_ABAS, ROTULO_SIMULACAO] if com_simulacao else list(ROTULOS_ABAS)
        self._barra.tabs = [ft.Tab(label=ft.Text(r, color=tema.TEXTO)) for r in rotulos]
        self._vista.controls = self._paineis[: len(rotulos)]
        self.tabs.length = len(rotulos)
        if self.tabs.selected_index >= len(rotulos):
            self.tabs.selected_index = ABA_PARAMETROS

    def ir_para(self, aba: int) -> None:
        """Seleciona a aba. Já na tela, usa `Tabs.move_to` (no Flet 0.86.2, mudar
        `selected_index` de abas montadas não move a aba visível); o evento de mudança
        atualiza `selected_index`."""
        if esta_na_pagina(self.tabs):
            self.tabs.page.run_task(self.tabs.move_to, aba)
        else:
            self.tabs.selected_index = aba
