import flet as ft
from utils import testes_hipotese
from ConteinerTestes import CardComparacaoTestes


def teste_selecionado(teste_name, colunas):

    print(teste_name)

    TESTES = testes_hipotese
    for nome, descricao in TESTES:

        if nome == teste_name:

            funcao = globals().get(nome)

            if funcao is None:
                return teste_t_1am(colunas)

            return funcao(colunas)

    raise ValueError(
        f"O teste '{teste_name}' não está cadastrado."
    )


def teste_t_1am(colunas):
    card = CardComparacaoTestes()
    
    parametro = ft.Row([
        ft.Column(
        controls=[
            ft.Column(spacing=0.9, controls=[
            ft.Text(
                "Parâmetros do teste",
                size=16,
                weight=ft.FontWeight.BOLD,
                color='#232529'
            ),

            ft.Text(
                "Selecione a variável e informe os parâmetros.", color='#232529'
            )]),
            ft.Divider(color='transparent', height=1),

            ft.Dropdown(
                label="Variável", color=ft.Colors.BLACK,
                text_size=16,
                text_style=ft.TextStyle(
                    weight=ft.FontWeight.W_600,
                    color=ft.Colors.BLACK,
                ),
                label_style=ft.TextStyle(
                    color=ft.Colors.BLACK_54,
                    weight=ft.FontWeight.BOLD,
                    size=14,
                ),
                leading_icon=ft.Icons.TUNE,

                options=[
                    ft.dropdown.Option(str(c), style=ft.ButtonStyle(color=ft.Colors.BLACK))
                    for c in colunas
                ],
                width=300,
                height=58,
                bgcolor=ft.Colors.WHITE,
                border_width=2,
                border_color=ft.Colors.BLACK_54,
                border_radius=10,
                focused_border_width=3,
                focused_border_color="#FF6A1A",
                content_padding=ft.Padding(
                    left=14,
                    right=14,
                    top=8,
                    bottom=8,
                ),
            ),

            ft.Column(
                spacing=6,
                controls=[
                    ft.Row(
                        controls=[
                            ft.TextField(
                                label="Média Hipotética",
                                prefix_icon=ft.Icons.ANALYTICS,
                                

                                # TEXTO
                                color=ft.Colors.BLACK,
                                text_size=16,
                                text_style=ft.TextStyle(
                                    weight=ft.FontWeight.W_600
                                ),

                                # LABEL
                                label_style=ft.TextStyle(
                                    color=ft.Colors.BLACK_54,
                                    weight=ft.FontWeight.BOLD,
                                    size=14,
                                ),

                                # CAMPO
                                bgcolor=ft.Colors.WHITE,
                                border_width=2,
                                border_color=ft.Colors.BLACK_54,
                                border_radius=10,

                                # FOCO
                                focused_border_width=3,
                                focused_border_color="#FF6A1A",
                                cursor_color="#FF6A1A",

                                # DIMENSÕES
                                height=58,
                                content_padding=ft.Padding(
                                    left=14,
                                    right=14,
                                    top=8,
                                    bottom=8,
                                ),
                            )
                        ]
                    )
                ]
            ),

            ft.Dropdown(
                label="Nível de significância (α)",

                color=ft.Colors.BLACK,
                text_size=16,
                text_style=ft.TextStyle(
                    weight=ft.FontWeight.W_600,
                    color=ft.Colors.BLACK,
                ),

                label_style=ft.TextStyle(
                    color=ft.Colors.BLACK_54,
                    weight=ft.FontWeight.BOLD,
                    size=14,
                ),

                leading_icon=ft.Icons.TUNE,
                
                options=[
                    ft.dropdown.Option("0.01",
                    style=ft.ButtonStyle(
                    color=ft.Colors.BLACK
                    )),
                    ft.dropdown.Option("0.05",
                    style=ft.ButtonStyle(
                    color=ft.Colors.BLACK
                    )),
                    ft.dropdown.Option("0.10",
                    style=ft.ButtonStyle(
                    color=ft.Colors.BLACK
                    )),
                ],

                value="0.05",

                width=300,
                height=58,

                bgcolor=ft.Colors.WHITE,

                border_width=2,
                border_color=ft.Colors.BLACK_54,
                border_radius=10,

                focused_border_width=3,
                focused_border_color="#FF6A1A",

                content_padding=ft.Padding(
                    left=14,
                    right=14,
                    top=8,
                    bottom=8,
                ),
            ),
        ],
        spacing=15
    ),   
        ft.Column(width=12),
        card
    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN,)

    analise = ft.Row([
        ft.Column(
            expand=True,
        controls=[
            card
        ]
    ),

    ])

    visual = ft.Column(
        controls=[
            ft.Text(
                "Visualização",
                size=16,
                weight=ft.FontWeight.BOLD,
                color='#232529'
            ),

            ft.Text(
                "O gráfico será exibido aqui.", color='#232529'
            )
        ]
    )

    return parametro, analise, visual