"""Barra lateral: logo, botão Arquivo, lista de testes agrupada e botão Executar."""

from collections.abc import Callable

import flet as ft

from app.ui import tema
from app.ui.helpers import border_only, pad
from core import registry


class Sidebar(ft.Container):
    """Sidebar de 260 px. Os callbacks só repassam o evento para o controller."""

    def __init__(
        self,
        on_arquivo: Callable,
        on_selecionar: Callable[[str], None],
        on_executar: Callable,
        selecionado: str | None = None,
    ):
        self._on_selecionar = on_selecionar
        self._linhas: dict[str, tuple[ft.Container, ft.Icon, ft.Text]] = {}
        self.selecionado = selecionado or registry.primeiro().id

        self.botao_arquivo = ft.Button(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.FOLDER_OUTLINED, size=19, color=tema.LARANJA),
                    ft.Text("Arquivo", size=14, color=tema.LARANJA, weight=ft.FontWeight.W_600),
                ],
                spacing=12,
            ),
            on_click=on_arquivo,
            bgcolor=tema.LARANJA_SUAVE,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=tema.RAIO_PEQUENO),
                padding=pad(horizontal=18, vertical=18),
                elevation=0,
            ),
        )
        self.botao_executar = ft.Button(
            align=ft.Alignment.CENTER,
            content=ft.Row(
                [
                    ft.Text(
                        "Executar teste",
                        size=14,
                        weight=ft.FontWeight.W_600,
                        color=tema.TEXTO_SOBRE_DESTAQUE,
                    ),
                    ft.Icon(ft.Icons.BALANCE, size=18, color=tema.TEXTO_SOBRE_DESTAQUE),
                ],
                spacing=8,
                tight=True,
            ),
            on_click=on_executar,
            bgcolor=tema.LARANJA,
            color=tema.TEXTO_SOBRE_DESTAQUE,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=tema.RAIO_PEQUENO),
                padding=pad(horizontal=18, vertical=18),
                elevation=0,
            ),
        )

        super().__init__(
            content=ft.Column(
                [
                    self._logo(),
                    ft.Text(
                        "Processamento e análise de dados", size=11, color=tema.TEXTO_SECUNDARIO
                    ),
                    ft.Divider(color=tema.TRANSPARENTE, height=12),
                    self.botao_arquivo,
                    ft.Divider(thickness=0.4, color=tema.TEXTO_SECUNDARIO),
                    self._lista_testes(),
                    ft.Divider(thickness=0.4, color=tema.TEXTO_SECUNDARIO),
                    self.botao_executar,
                ],
                spacing=6,
            ),
            width=260,
            bgcolor=tema.CARTAO,
            padding=pad(left=22, right=22, top=26, bottom=22),
            border=border_only(right=ft.BorderSide(1, tema.BORDA)),
        )

    @staticmethod
    def _logo() -> ft.Row:
        return ft.Row(
            [
                ft.Icon(ft.Icons.BAR_CHART, size=30, color=tema.LARANJA),
                ft.Text("smart", size=21, weight=ft.FontWeight.BOLD, color=tema.TEXTO),
                ft.Text("ETL", size=21, weight=ft.FontWeight.BOLD, color=tema.LARANJA),
            ],
            spacing=2,
        )

    def _lista_testes(self) -> ft.Container:
        controles: list[ft.Control] = []
        for i, (grupo, testes) in enumerate(registry.por_grupo().items()):
            controles.append(self._cabecalho_grupo(grupo, primeiro=i == 0))
            controles.extend(self._linha_teste(info) for info in testes)
        return ft.Container(
            content=ft.Column(
                controls=controles, spacing=4, scroll=ft.ScrollMode.AUTO, expand=True
            ),
            expand=True,
            bgcolor=tema.CARTAO,
            padding=1,
        )

    @staticmethod
    def _cabecalho_grupo(grupo: str, primeiro: bool) -> ft.Container:
        return ft.Container(
            content=ft.Text(
                grupo.upper(), size=11, weight=ft.FontWeight.W_600, color=tema.TEXTO_TERCIARIO
            ),
            padding=pad(left=12, top=2 if primeiro else 10, bottom=2),
        )

    def _linha_teste(self, info: registry.TesteInfo) -> ft.Container:
        icone = ft.Icon(ft.Icons.RADIO_BUTTON_UNCHECKED, size=20)
        texto = ft.Text(info.nome, size=14)
        linha = ft.Container(
            content=ft.Row(controls=[icone, texto], spacing=10),
            padding=ft.Padding.symmetric(horizontal=12, vertical=10),
            border_radius=tema.RAIO_PEQUENO,
            on_click=lambda e, tid=info.id: self.selecionar(tid),
            ink=True,
        )
        self._linhas[info.id] = (linha, icone, texto)
        self._aplicar_visual(info.id, info.id == self.selecionado)
        return linha

    def _aplicar_visual(self, teste_id: str, marcado: bool) -> None:
        linha, icone, texto = self._linhas[teste_id]
        icone.icon = ft.Icons.RADIO_BUTTON_CHECKED if marcado else ft.Icons.RADIO_BUTTON_UNCHECKED
        icone.color = tema.LARANJA if marcado else tema.TEXTO_TERCIARIO
        linha.bgcolor = tema.LARANJA_SUAVE if marcado else tema.TRANSPARENTE
        texto.color = tema.LARANJA if marcado else tema.TEXTO
        texto.weight = ft.FontWeight.W_600 if marcado else ft.FontWeight.NORMAL

    def selecionar(self, teste_id: str) -> None:
        """Marca o teste (só altera as duas linhas envolvidas) e avisa o controller."""
        if teste_id != self.selecionado:
            self._aplicar_visual(self.selecionado, False)
            self._aplicar_visual(teste_id, True)
            self.selecionado = teste_id
        self._on_selecionar(teste_id)
