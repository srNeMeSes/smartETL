"""
smartETL — Protótipo de interface (Flet)
=========================================

Aplicativo desktop de análise e processamento de dados.
Este é um protótipo VISUAL: os dados são fictícios e gerados em memória,
e os botões (Gerar relatório, Filtros, Visualizar, Exportar, navegação
lateral) não possuem funcionalidade real — o foco é apresentar o
conceito de interface do SmartETL.

Como executar:
    pip install flet
    python smartetl_app.py

Requer Python 3.9+ e o pacote `flet` (testado com flet==0.86.2).
"""

import random
from datetime import datetime

import flet as ft

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


# ----------------------------------------------------------------------------
# Pequenos helpers de layout (Padding / BorderRadius / Border / Alignment
# não possuem funções de conveniência do tipo `.all()` / `.symmetric()` /
# `.only()` nesta versão do Flet, então construímos os objetos manualmente).
# ----------------------------------------------------------------------------

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


ALIGN_CENTER = ft.Alignment(0, 0)
ALIGN_TOP_RIGHT = ft.Alignment(1, -1)
ALIGN_BOTTOM_CENTER = ft.Alignment(0, 1)
ALIGN_CENTER_RIGHT = ft.Alignment(1, 0)


# ----------------------------------------------------------------------------
# Dados fictícios
# ----------------------------------------------------------------------------

FILIAIS = [f"Filial {str(i).zfill(2)}" for i in range(1, 13)]
MESES = ["01/2026", "02/2026", "03/2026", "04/2026", "05/2026", "06/2026"]

COLUMNS_INFO = ["filial", "mes", "faturamento", "taxa_conversao", "qtd_vendas"]


def gerar_dados(total_registros: int = 67):
    """Gera registros fictícios simulando bases mensais enviadas por filiais."""
    random.seed(42)  # reprodutível, mas com aparência variada
    registros = []
    for _ in range(total_registros):
        registros.append(
            {
                "filial": random.choice(FILIAIS),
                "mes": random.choice(MESES),
                "faturamento": round(random.uniform(48_000, 260_000), 2),
                "taxa_conversao": round(random.uniform(52.0, 94.0), 1),
                "qtd_vendas": random.randint(58, 210),
            }
        )
    # ordena por mês para dar uma sensação de consolidação cronológica
    registros.sort(key=lambda r: (r["mes"], r["filial"]))
    return registros


def formatar_moeda(valor: float) -> str:
    """Formata um número float no padrão monetário brasileiro (1.234,56)."""
    texto = f"{valor:,.2f}"
    texto = texto.replace(",", "§").replace(".", ",").replace("§", ".")
    return texto


def top_filiais_por_faturamento(dados, n=5):
    """Agrega o faturamento total por filial e retorna as `n` maiores."""
    totais = {}
    for r in dados:
        totais[r["filial"]] = totais.get(r["filial"], 0) + r["faturamento"]
    ordenado = sorted(totais.items(), key=lambda kv: kv[1], reverse=True)
    return ordenado[:n]


# ----------------------------------------------------------------------------
# Componentes reutilizáveis
# ----------------------------------------------------------------------------

def card_container(content, expand=None, padding=24, height=None):
    """Cartão branco padrão com borda suave e cantos arredondados."""
    return ft.Container(
        content=content,
        bgcolor=CARD_BG,
        border=border_all(1, BORDER),
        border_radius=RADIUS_CARD,
        padding=padding,
        expand=expand,
        height=height,
    )


def section_title(text_value: str, trailing=None):
    row_controls = [ft.Text(text_value, size=16, weight=ft.FontWeight.W_600, color=TEXT_DARK)]
    if trailing:
        return ft.Row(
            [row_controls[0], trailing],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )
    return ft.Row(row_controls)


def variaveis_analisadas_row(colunas):
    """Linha 'Variáveis analisadas: col1, col2, ...' com destaque em laranja."""
    spans = [ft.Text("Variáveis analisadas:  ", size=13, color=TEXT_GRAY)]
    for i, coluna in enumerate(colunas):
        spans.append(ft.Text(coluna, size=13, color=ORANGE, weight=ft.FontWeight.W_600))
        if i < len(colunas) - 1:
            spans.append(ft.Text(", ", size=13, color=TEXT_GRAY))
    return ft.Row(spans, wrap=True, spacing=0)


def nav_item(icon, label, selected=False):
    """Item de navegação da barra lateral."""
    color = ORANGE if selected else TEXT_GRAY
    return ft.Container(
        content=ft.Row(
            [
                ft.Icon(icon, size=19, color=color),
                ft.Text(
                    label,
                    size=14,
                    color=color,
                    weight=ft.FontWeight.W_600 if selected else ft.FontWeight.W_400,
                ),
            ],
            spacing=12,
        ),
        padding=pad(horizontal=16, vertical=13),
        border_radius=RADIUS_SM,
        bgcolor=ORANGE_SOFT if selected else None,
    )


def toolbar_button(icon, label):
    """Botão discreto usado na barra de ferramentas da tabela (visual)."""
    return ft.OutlinedButton(
        content=ft.Row(
            [ft.Icon(icon, size=16, color=TEXT_GRAY), ft.Text(label, size=13, color=TEXT_DARK)],
            spacing=6,
            tight=True,
        ),
        style=ft.ButtonStyle(
            side=ft.BorderSide(1, BORDER),
            shape=ft.RoundedRectangleBorder(radius=RADIUS_SM),
            bgcolor=CARD_BG,
            padding=pad(horizontal=14, vertical=10),
        ),
    )


def build_bar_chart(chart_data, height=210):
    """
    Gráfico de barras simples e minimalista construído com Containers,
    sem depender de bibliotecas externas de gráficos.
    `chart_data`: lista de tuplas (rotulo, valor).
    """
    max_val = max(v for _, v in chart_data) if chart_data else 1
    ticks = [max_val, max_val * 0.75, max_val * 0.5, max_val * 0.25, 0]

    eixo = ft.Column(
        [
            ft.Container(
                content=ft.Text(f"{t / 1000:.0f}K", size=10, color=TEXT_MUTED),
                height=height / (len(ticks) - 1),
                alignment=ALIGN_TOP_RIGHT,
            )
            for t in ticks[:-1]
        ]
        + [ft.Text("0", size=10, color=TEXT_MUTED)],
        height=height,
        spacing=0,
    )

    barras = []
    for rotulo, valor in chart_data:
        altura_barra = max(6, height * (valor / max_val)) if max_val else 6
        barras.append(
            ft.Column(
                [
                    ft.Container(
                        content=ft.Container(
                            width=46,
                            height=altura_barra,
                            bgcolor=ORANGE,
                            border_radius=radius(top_left=8, top_right=8),
                        ),
                        height=height,
                        alignment=ALIGN_BOTTOM_CENTER,
                    ),
                    ft.Container(height=8),
                    ft.Text(rotulo, size=11, color=TEXT_GRAY, text_align=ft.TextAlign.CENTER),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=0,
            )
        )

    return ft.Row(
        [
            eixo,
            ft.Container(width=1, height=height, bgcolor=BORDER),
            ft.Container(
                content=ft.Row(
                    barras,
                    alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                    vertical_alignment=ft.CrossAxisAlignment.END,
                    expand=True,
                ),
                padding=pad(left=16, right=8),
                expand=True,
            ),
        ],
        vertical_alignment=ft.CrossAxisAlignment.START,
    )


# ----------------------------------------------------------------------------
# Aplicação principal
# ----------------------------------------------------------------------------

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

    dados = gerar_dados(67)
    total_bases = len(dados)
    chart_data = top_filiais_por_faturamento(dados, n=5)

    # ---------------- Sidebar ----------------

    logo_row = ft.Row(
        [
            ft.Container(width=10, height=10, bgcolor=ORANGE, border_radius=3),
            ft.Text("smart", size=21, weight=ft.FontWeight.BOLD, color=TEXT_DARK),
            ft.Text("ETL", size=21, weight=ft.FontWeight.BOLD, color=ORANGE),
        ],
        spacing=6,
    )

    status_card = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(ft.Icons.STORAGE_ROUNDED, size=18, color=TEXT_GRAY),
                        ft.Column(
                            [
                                ft.Text(f"{total_bases} bases de dados", size=12, weight=ft.FontWeight.W_600, color=TEXT_DARK),
                                ft.Text("Carregadas com sucesso", size=11, color=TEXT_GRAY),
                            ],
                            spacing=0,
                        ),
                    ],
                    spacing=10,
                ),
                ft.Divider(height=14, color=BORDER),
                ft.Row(
                    [
                        ft.Text(datetime.now().strftime("%d/%m/%Y %H:%M"), size=11, color=TEXT_GRAY),
                        ft.Container(width=8, height=8, bgcolor=GREEN, border_radius=4),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
            ],
            spacing=8,
        ),
        bgcolor=BG,
        border=border_all(1, BORDER),
        border_radius=RADIUS_SM,
        padding=14,
    )

    sidebar = ft.Container(
        content=ft.Column(
            [
                logo_row,
                ft.Text("Processamento e análise de dados", size=11, color=TEXT_GRAY),
                ft.Container(height=28),
                nav_item(ft.Icons.FOLDER_OUTLINED, "Arquivo", selected=True),
                nav_item(ft.Icons.HISTORY_ROUNDED, "Histórico"),
                nav_item(ft.Icons.DESCRIPTION_OUTLINED, "Logs"),
                ft.Container(expand=True),
                status_card,
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
                    ft.Text("Análise consolidada das bases mensais das filiais", size=13, color=TEXT_GRAY),
                ],
                spacing=4,
            ),
            ft.ElevatedButton(
                content=ft.Row(
                    [ft.Text("Gerar relatório", size=13, weight=ft.FontWeight.W_600), ft.Icon(ft.Icons.FILE_DOWNLOAD_OUTLINED, size=18)],
                    spacing=8,
                    tight=True,
                ),
                bgcolor=ORANGE,
                color="#FFFFFF",
                style=ft.ButtonStyle(
                    shape=ft.RoundedRectangleBorder(radius=RADIUS_SM),
                    padding=pad(horizontal=18, vertical=18),
                    elevation=0,
                ),
            ),
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )

    # ---------------- Descrição geral ----------------

    descricao_texto = (
        "Com base nas bases de dados mensais fornecidas pelas filiais, observou-se "
        "um cenário geral positivo nos principais indicadores analisados. Os testes "
        "de hipótese realizados apresentaram resultados estatisticamente significativos, "
        "enquanto a análise de regressão logística indicou tendências favoráveis de "
        "crescimento e evolução dos resultados. De forma geral, os dados consolidados "
        "apontam para um desempenho positivo da empresa no período analisado."
    )

    descricao_card = card_container(
        ft.Column(
            [
                section_title("Descrição geral"),
                variaveis_analisadas_row(COLUMNS_INFO),
                ft.Container(height=6),
                ft.Text(descricao_texto, size=13, color=TEXT_DARK, height=1.6),
            ],
            spacing=14,
        ),
        expand=1,
        height=320,
    )

    # ---------------- Visualização geral ----------------

    chart_series = [(filial, valor) for filial, valor in chart_data]

    visualizacao_card = card_container(
        ft.Column(
            [
                section_title("Visualização geral", trailing=ft.Icon(ft.Icons.MORE_VERT, size=18, color=TEXT_MUTED)),
                ft.Text("Faturamento total — top 5 filiais", size=12, color=TEXT_GRAY),
                ft.Container(height=6),
                build_bar_chart(chart_series),
            ],
            spacing=6,
        ),
        expand=1,
        height=320,
    )

    top_row = ft.Row([descricao_card, visualizacao_card], spacing=20, vertical_alignment=ft.CrossAxisAlignment.START)

    # ---------------- Tabela geral (com paginação simples) ----------------

    PAGE_SIZE = 10
    total_paginas = max(1, (len(dados) + PAGE_SIZE - 1) // PAGE_SIZE)
    estado_pagina = {"atual": 1}

    def linhas_da_pagina(pagina):
        inicio = (pagina - 1) * PAGE_SIZE
        return dados[inicio: inicio + PAGE_SIZE]

    def construir_linhas(registros):
        return [
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(r["filial"], size=13, color=TEXT_DARK)),
                    ft.DataCell(ft.Text(r["mes"], size=13, color=TEXT_GRAY)),
                    ft.DataCell(ft.Text(f"R$ {formatar_moeda(r['faturamento'])}", size=13, color=TEXT_DARK)),
                    ft.DataCell(ft.Text(f"{r['taxa_conversao']}%", size=13, color=TEXT_DARK)),
                    ft.DataCell(ft.Text(str(r["qtd_vendas"]), size=13, color=TEXT_DARK)),
                ]
            )
            for r in registros
        ]

    def coluna_titulo(texto):
        return ft.DataColumn(ft.Text(texto, size=12, weight=ft.FontWeight.W_600, color=TEXT_GRAY))

    tabela = ft.DataTable(
        columns=[
            coluna_titulo("Filial"),
            coluna_titulo("Mês"),
            coluna_titulo("Faturamento"),
            coluna_titulo("Taxa de Conversão"),
            coluna_titulo("Qtd. Vendas"),
        ],
        rows=construir_linhas(linhas_da_pagina(1)),
        heading_row_color=BG,
        heading_row_height=44,
        data_row_min_height=44,
        data_row_max_height=44,
        divider_thickness=1,
        border=border_all(1, BORDER),
        border_radius=RADIUS_SM,
        horizontal_lines=ft.BorderSide(1, BORDER),
        column_spacing=48,
        expand=True,
    )

    intervalo_texto = ft.Text(f"1–{min(PAGE_SIZE, len(dados))} de {len(dados)}", size=12, color=TEXT_GRAY)
    pagina_atual_badge = ft.Container(
        content=ft.Text("1", size=12, color="#FFFFFF", weight=ft.FontWeight.W_600),
        width=28,
        height=28,
        bgcolor=ORANGE,
        border_radius=8,
        alignment=ALIGN_CENTER,
    )

    def ir_para_pagina(nova_pagina):
        nova_pagina = max(1, min(total_paginas, nova_pagina))
        estado_pagina["atual"] = nova_pagina
        tabela.rows = construir_linhas(linhas_da_pagina(nova_pagina))
        inicio = (nova_pagina - 1) * PAGE_SIZE + 1
        fim = min(nova_pagina * PAGE_SIZE, len(dados))
        intervalo_texto.value = f"{inicio}–{fim} de {len(dados)}"
        pagina_atual_badge.content.value = str(nova_pagina)
        page.update()

    paginacao = ft.Row(
        [
            ft.IconButton(ft.Icons.KEYBOARD_DOUBLE_ARROW_LEFT, icon_size=16, icon_color=TEXT_GRAY, on_click=lambda e: ir_para_pagina(1)),
            ft.IconButton(ft.Icons.CHEVRON_LEFT, icon_size=18, icon_color=TEXT_GRAY, on_click=lambda e: ir_para_pagina(estado_pagina["atual"] - 1)),
            pagina_atual_badge,
            ft.IconButton(ft.Icons.CHEVRON_RIGHT, icon_size=18, icon_color=TEXT_GRAY, on_click=lambda e: ir_para_pagina(estado_pagina["atual"] + 1)),
            ft.IconButton(ft.Icons.KEYBOARD_DOUBLE_ARROW_RIGHT, icon_size=16, icon_color=TEXT_GRAY, on_click=lambda e: ir_para_pagina(total_paginas)),
        ],
        spacing=2,
    )

    tabela_toolbar = ft.Row(
        [
            toolbar_button(ft.Icons.FILTER_LIST_ROUNDED, "Filtros"),
            toolbar_button(ft.Icons.VISIBILITY_OUTLINED, "Visualizar"),
            toolbar_button(ft.Icons.FILE_DOWNLOAD_OUTLINED, "Exportar"),
            ft.Container(expand=True),
            ft.Container(
                content=ft.TextField(
                    hint_text="Buscar...",
                    prefix_icon=ft.Icons.SEARCH,
                    height=42,
                    text_size=13,
                    border_color=BORDER,
                    border_radius=RADIUS_SM,
                    bgcolor=CARD_BG,
                    content_padding=pad(horizontal=12, vertical=8),
                ),
                width=220,
            ),
        ],
        spacing=10,
    )

    tabela_card = card_container(
        ft.Column(
            [
                ft.Row(
                    [
                        ft.Column(
                            [
                                ft.Text("Tabela geral", size=16, weight=ft.FontWeight.W_600, color=TEXT_DARK),
                                ft.Text(f"{total_bases} bases de dados analisadas", size=12, color=TEXT_GRAY),
                            ],
                            spacing=2,
                        ),
                        tabela_toolbar,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
                ft.Container(height=6),
                ft.Container(
                    content=ft.Row(
                        [ft.Column([tabela], scroll=ft.ScrollMode.AUTO, expand=True)],
                        scroll=ft.ScrollMode.AUTO,
                    ),
                    height=430,
                ),
                ft.Row(
                    [intervalo_texto, ft.Row([ft.Text("Linhas por página: 10", size=12, color=TEXT_GRAY), paginacao], spacing=16)],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
            ],
            spacing=14,
        )
    )

    # ---------------- Layout final ----------------

    conteudo_principal = ft.Column(
        [
            header,
            ft.Container(height=20),
            top_row,
            ft.Container(height=20),
            tabela_card,
        ],
        scroll=ft.ScrollMode.AUTO,
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
