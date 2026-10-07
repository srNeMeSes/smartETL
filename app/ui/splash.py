"""Tela de abertura (splash), exibida enquanto os módulos estatísticos são carregados.

Só depende de Flet e do tema: é montada antes dos imports pesados (pandas, scipy, statsmodels),
para a janela aparecer na hora.
"""

import flet as ft

from app.ui import tema

MENSAGEM_CARREGANDO = "Carregando módulos estatísticos..."
LARGURA_CARTAO = 560


class Splash(ft.Container):
    def __init__(self):
        self.status = ft.Text(MENSAGEM_CARREGANDO, size=13, color=tema.TEXTO_SECUNDARIO)
        logo = ft.Row(
            [
                ft.Icon(ft.Icons.BAR_CHART, size=64, color=tema.LARANJA),
                ft.Text("smart", size=44, weight=ft.FontWeight.BOLD, color=tema.TEXTO),
                ft.Text("ETL", size=44, weight=ft.FontWeight.BOLD, color=tema.LARANJA),
            ],
            spacing=4,
            alignment=ft.MainAxisAlignment.CENTER,
        )
        cartao = ft.Container(
            content=ft.Column(
                [
                    logo,
                    ft.Text(
                        "Processamento e análise de dados",
                        size=15,
                        color=tema.TEXTO_SECUNDARIO,
                    ),
                    ft.Text(
                        "Testes de hipótese  |  ANOVA  |  Regressão",
                        size=13,
                        color=tema.TEXTO_TERCIARIO,
                    ),
                    ft.Container(height=24),
                    ft.ProgressRing(width=28, height=28, stroke_width=3, color=tema.LARANJA),
                    self.status,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
                tight=True,
            ),
            width=LARGURA_CARTAO,
            padding=ft.Padding.symmetric(horizontal=48, vertical=56),
            bgcolor=tema.CARTAO,
            border_radius=tema.RAIO_PEQUENO * 2,
            shadow=ft.BoxShadow(blur_radius=24, color=tema.SOMBRA_CARTAO, offset=ft.Offset(0, 6)),
        )
        super().__init__(
            content=cartao,
            alignment=ft.Alignment.CENTER,
            bgcolor=tema.FUNDO,
            expand=True,
        )
