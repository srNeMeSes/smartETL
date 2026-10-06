import flet as ft
from utils import *
from load_table import *
from conteiner_parametros import teste_selecionado
import os
import pandas as pd


# ----------------------------------------------------------------------------
# Paleta de cores e tokens de estilo
# ----------------------------------------------------------------------------

ORANGE = "#FF6A1A"          # cor de destaque (identidade smartETL)
ORANGE_SOFT = "#FFF1E6"     # fundo suave para itens/estados selecionados
BG = "#F4F5F7"              # fundo geral da aplicação
CARD_BG = "#FFFFFF"         # fundo dos cartões
BORDER = "#E7E8EC"          # bordas suaves
TEXT_DARK = "#232529"       # texto principal
TEXT_GRAY = "#8A8D93"       # texto secundário / rótulos
TEXT_MUTED = "#B4B6BC"      # texto terciário
GREEN = "#3BAA6E"           # indicador de status positivo

FONT_FAMILY = "Segoe UI, Roboto, Arial, sans-serif"

RADIUS_CARD = 16
RADIUS_SM = 10

ALIGN_CENTER = ft.Alignment(0, 0)
ALIGN_TOP_RIGHT = ft.Alignment(1, -1)
ALIGN_BOTTOM_CENTER = ft.Alignment(0, 1)
ALIGN_CENTER_RIGHT = ft.Alignment(1, 0)





# Pequenos helpers de layout
def pad(all=None, horizontal=None, vertical=None, left=0, top=0, right=0, bottom=0):
    if all is not None:
        return ft.Padding(left=all, top=all, right=all, bottom=all)
    if horizontal is not None or vertical is not None:
        h = horizontal or 0
        v = vertical or 0
        return ft.Padding(left=h, top=v, right=h, bottom=v)
    return ft.Padding(left=left, top=top, right=right, bottom=bottom)


def radius(all=None, top_left=0, top_right=0, bottom_left=0, bottom_right=0):
    if all is not None:
        return ft.BorderRadius(top_left=all, top_right=all, bottom_left=all, bottom_right=all)
    return ft.BorderRadius(top_left=top_left, top_right=top_right, bottom_left=bottom_left, bottom_right=bottom_right)


def border_all(width, color):
    side = ft.BorderSide(width, color)
    return ft.Border(top=side, right=side, bottom=side, left=side)


def border_only(top=None, right=None, bottom=None, left=None):
    kwargs = {}
    if top is not None:
        kwargs["top"] = top
    if right is not None:
        kwargs["right"] = right
    if bottom is not None:
        kwargs["bottom"] = bottom
    if left is not None:
        kwargs["left"] = left
    return ft.Border(**kwargs)


def card_container(content, expand=None, padding=24):
    """Cartão branco padrão com borda suave e cantos arredondados."""
    return ft.Container(
        content=content,
        bgcolor=CARD_BG,
        border=border_all(1, BORDER),
        border_radius=RADIUS_CARD,
        padding=padding,
        expand=expand,
    )

def section_title(text_value: str, trailing=None):
    row_controls = [ft.Text(text_value, size=16, weight=ft.FontWeight.W_600, color=TEXT_DARK)]
    if trailing:
        return ft.Row(
            [row_controls[0], trailing],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )
    return ft.Row(row_controls)



def main(page: ft.Page):
    page.title = "smartETL — Processamento de dados"
    page.bgcolor = BG
    page.padding = 0
    page.fonts = {}
    page.theme = ft.Theme(font_family=FONT_FAMILY)
    page.window.width = 1440
    page.window.height = 900
    page.window.min_width = 1150
    page.window.min_height = 720
    page.window.center()

    DATABASE = None


    def carregar_base(df):
        # Limpa a tabela atual
        tabela.columns.clear()
        tabela.rows.clear()
        quantidade_colunas = len(df.columns)
        minimo_colunas = 20
        maximo_linhas = 100
        # COLUNAS
        tabela.columns = [
            coluna_titulo(str(coluna))
            for coluna in df.columns
        ]
        # Se tiver menos de 20 colunas,
        # adiciona colunas complementares vazias
        if quantidade_colunas < minimo_colunas:
            colunas_vazias = minimo_colunas - quantidade_colunas

            tabela.columns.extend(
                [
                    coluna_titulo(f"column{quantidade_colunas + i}")
                    for i in range(1, colunas_vazias + 1)
                ]
            )
        # LINHAS
        # Pega no máximo 100 linhas
        df_visualizacao = df.head(maximo_linhas)
        tabela.rows = []
        for linha in df_visualizacao.itertuples(index=False, name=None):
            valores = list(linha)
            # Completa as colunas vazias até chegar em 20
            if quantidade_colunas < minimo_colunas:
                valores.extend(
                    ["   "] * (minimo_colunas - quantidade_colunas)
                )
            tabela.rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(
                            ft.Text(
                                str(valor),
                                size=14,
                                no_wrap=True,
                                weight=ft.FontWeight.W_600,
                                color=TEXT_GRAY
                            )
                        )
                        for valor in valores
                    ]
                )
            )

        # Atualiza a página
        page.update()


    def mostrar_testes(teste, cols):
        if cols is None:
            return
        pa, an, vi = teste_selecionado(teste, cols)

        tabs.content.controls[1].controls[0].content = pa
        tabs.content.controls[1].controls[1].content = an
        tabs.content.controls[1].controls[2].content = vi

        page.update()



    # ---------------- Sidebar ----------------
    logo_row = ft.Row(
        [
            ft.Icon(ft.Icons.BAR_CHART, size=30, color=ORANGE), #SCATTER_PLOT, BALANCE, BAR_CHART
            ft.Text("smart", size=21, weight=ft.FontWeight.BOLD, color=TEXT_DARK),
            ft.Text("ETL", size=21, weight=ft.FontWeight.BOLD, color=ORANGE),
        ],
        spacing=2,
    )

    bt_teste = ft.ElevatedButton(
        align=ft.Alignment.CENTER,
        content=ft.Row(
            [ft.Text("Executar teste", size=14, weight=ft.FontWeight.W_600, color='white'),
            ft.Icon(ft.Icons.BALANCE, size=18, color='white')],
            spacing=8,
            tight=True,
        ),
        bgcolor=ORANGE,
        color="#FFFFFF",
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=RADIUS_SM),
            padding=pad(horizontal=18, vertical=18),
            elevation=0,
        )
    )

    #------------------ selecionar base ----------------------
    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    caminho_arquivo = None  # aqui fica o path completo depois de selecionar

    async def selecionar_arquivo(e):
        nonlocal caminho_arquivo
        nonlocal DATABASE
        files = await file_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=["xlsx", "csv"]
        )
        if files:
            caminho_arquivo = files[0].path
            nome_arquivo = os.path.basename(caminho_arquivo)
            extensao = os.path.splitext(nome_arquivo)[1].lower()
            print("Caminho:", caminho_arquivo)
            print("Nome:", nome_arquivo)
            print("Extensão:", extensao)
            DATABASE = importar_dados(caminho_arquivo, extensao)
            carregar_base(DATABASE)
        page.update()

    

    #--------------- Sidebar ------------------
    sidebar = ft.Container(
            content=ft.Column(
                [
                    logo_row,
                    ft.Text("Processamento e análise de dados", size=11, color=TEXT_GRAY),
                    ft.Divider(color='transparent', height=12),
                    ft.ElevatedButton(
                        content=ft.Row(
                            [
                            ft.Icon(ft.Icons.FOLDER_OUTLINED, size=19, color=ORANGE),
                            ft.Text("Arquivo", size=14,color=ORANGE, weight=ft.FontWeight.W_600)
                            ],
                            spacing=12,                            
                        ),
                        on_click=selecionar_arquivo,
                        bgcolor=ORANGE_SOFT,
                        color="#FFFFFF",
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=RADIUS_SM),
                            padding=pad(horizontal=18, vertical=18),
                            elevation=0,
                            )
                        ),
                        
                    ft.Divider(thickness=0.4, color=TEXT_GRAY),
                    criar_sidebar_testes(page, on_selecionar= lambda v: mostrar_testes(v, DATABASE.columns if DATABASE is not None else None)),
                    ft.Divider(thickness=0.4, color=TEXT_GRAY),
                    bt_teste
            
                ],
                spacing=6,
            ),
            width=260,
            bgcolor=CARD_BG,
            padding=pad(left=22, right=22, top=26, bottom=22),
            border=border_only(right=ft.BorderSide(1, BORDER)),
        )

    
    # ---------------- Cabeçalho da área principal ----------------
    header = ft.Row(
        [
            ft.Column(
                [
                    ft.Text("Processamento de dados", size=24, weight=ft.FontWeight.BOLD, color=TEXT_DARK),
                    ft.Text("Testes de Hipótese  |  paramétricos e não paramétricos", size=13, color=TEXT_GRAY),
                ],
                spacing=4,
            )
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.START,
        )

    
    # ---------------- Tabela geral (com paginação simples) ----------------
    def coluna_titulo(texto):
        if str(texto).startswith('column'):
            return ft.DataColumn(ft.Text(texto, size=15, weight=ft.FontWeight.W_600, color=TEXT_MUTED))
        return ft.DataColumn(ft.Text(texto, size=15, weight=ft.FontWeight.W_600, color=TEXT_GRAY))

    tabela = ft.DataTable(
    columns=[
        coluna_titulo(f"column{i}") for i in range(1, 21)
    ],
    rows=[
        ft.DataRow(
            cells=[
                ft.DataCell(ft.Text("", weight=ft.FontWeight.W_600, color=TEXT_GRAY))
                for _ in range(20)
            ]
        )
        for _ in range(12)
    ],
    heading_row_color=BG,
    heading_row_height=44,
    data_row_min_height=44,
    data_row_max_height=44,
    divider_thickness=0.5,
    border=border_all(1, BORDER),
    border_radius=RADIUS_SM,
    horizontal_lines=ft.BorderSide(1, BORDER),
    column_spacing=48,
    )

    # ---------------- Área da tabela ----------------
    area_tabela = ft.Container(
        content=ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        tabela
                    ],
                    scroll=ft.ScrollMode.AUTO,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                )
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        ),
        height=320,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        border=border_all(1, BORDER),
        border_radius=RADIUS_SM,
        bgcolor=CARD_BG,
    )

    # ---------------- Layout dos testes ----------------
    tabs = ft.Tabs(
        selected_index=0,
        length=3,
        expand=True,
        content=ft.Column(
            expand=True,
            controls=[
                ft.TabBar(
                    tabs=[
                        ft.Tab(label=ft.Text("Parâmetros", color=ft.Colors.BLACK)),
                        ft.Tab(label=ft.Text("Análise", color=ft.Colors.BLACK)),
                        ft.Tab(label=ft.Text("Visualização", color=ft.Colors.BLACK)),
                    ],
                ),
                ft.TabBarView(
                    expand=True,
                    controls=[
                        ft.Container(
                            content=ft.Column(controls=[ft.Text("Processando...", color=ft.Colors.BLACK)]),
                            padding=20,
                        ),
                        ft.Container(
                            content=ft.Column(controls=[ft.Text("Processando...", color=ft.Colors.BLACK)]),
                            padding=20,
                        ),
                        ft.Container(
                            content=ft.Column(controls=[ft.Text("Processando...", color=ft.Colors.BLACK)]),
                            padding=20,
                        ),
                    ],
                ),
            ],
        ),
    )

    tabs_container = ft.Container(
        content=tabs,
        expand=True,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        border=border_all(1, BORDER),
        border_radius=RADIUS_SM,
        bgcolor=CARD_BG
    )



    # ---------------- Layout final ----------------
    conteudo_principal = ft.Column(
        controls=[
            header,
            ft.Divider(
                color="transparent",
                height=7,
            ),
            area_tabela,
            tabs_container
        ],
        expand=True,
    )

    area_principal = ft.Container(
    content=conteudo_principal,
    expand=True,
    padding=pad(left=32, right=32, top=28, bottom=28),
    bgcolor=BG,
    )

    page.add(ft.Row([sidebar, area_principal], expand=True, spacing=0))


if __name__ == "__main__":
    ft.run(main)
