"""Prévia do dataset: estado vazio (colunas fantasmas) separado do estado com dados."""

import flet as ft
import pandas as pd

from app.ui import tema
from app.ui.helpers import border_all

MAX_LINHAS = 100
COLUNAS_VAZIAS = 20
LINHAS_VAZIAS = 12


def formatar_celula(valor: object) -> str:
    """Texto de uma célula; NaN/None viram vazio."""
    try:
        if pd.isna(valor):
            return ""
    except (TypeError, ValueError):  # valores não escalares
        pass
    return str(valor)


def _titulo_coluna(texto: str, fantasma: bool = False) -> ft.DataColumn:
    cor = tema.TEXTO_TERCIARIO if fantasma else tema.TEXTO_SECUNDARIO
    return ft.DataColumn(ft.Text(texto, size=15, weight=ft.FontWeight.W_600, color=cor))


def _celula(texto: str) -> ft.DataCell:
    return ft.DataCell(
        ft.Text(
            texto, size=14, no_wrap=True, weight=ft.FontWeight.W_600, color=tema.TEXTO_SECUNDARIO
        )
    )


class TabelaDados(ft.Container):
    """Área de 320 px com rolagem horizontal e vertical. Reutiliza o mesmo `DataTable`."""

    def __init__(self):
        self.tabela = ft.DataTable(
            columns=[],
            rows=[],
            heading_row_color=tema.FUNDO,
            heading_row_height=44,
            data_row_min_height=44,
            data_row_max_height=44,
            divider_thickness=0.5,
            border=border_all(1, tema.BORDA),
            border_radius=tema.RAIO_PEQUENO,
            horizontal_lines=ft.BorderSide(1, tema.BORDA),
            column_spacing=48,
        )
        self.vazia = True
        self.mostrar_vazio()
        super().__init__(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[self.tabela],
                        scroll=ft.ScrollMode.AUTO,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                    )
                ],
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            ),
            height=320,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            border=border_all(1, tema.BORDA),
            border_radius=tema.RAIO_PEQUENO,
            bgcolor=tema.CARTAO,
        )

    def mostrar_vazio(self) -> None:
        """Estado sem arquivo: colunas `column1..20` e linhas em branco."""
        self.vazia = True
        self.tabela.columns = [
            _titulo_coluna(f"column{i}", fantasma=True) for i in range(1, COLUNAS_VAZIAS + 1)
        ]
        self.tabela.rows = [
            ft.DataRow(cells=[_celula("") for _ in range(COLUNAS_VAZIAS)])
            for _ in range(LINHAS_VAZIAS)
        ]

    def mostrar(self, df: pd.DataFrame) -> None:
        """Só as colunas reais e no máximo `MAX_LINHAS` linhas."""
        self.vazia = False
        self.tabela.columns = [_titulo_coluna(str(coluna)) for coluna in df.columns]
        self.tabela.rows = [
            ft.DataRow(cells=[_celula(formatar_celula(valor)) for valor in linha])
            for linha in df.head(MAX_LINHAS).itertuples(index=False, name=None)
        ]
