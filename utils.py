import flet as ft

testes_hipotese = [

    # Médias
    ("teste_t_1am", "Teste t (uma amostra)"),
    ("teste_t_2am", "Teste t (duas amostras)"),
    ("teste_t_pareado", "Teste t (pareado)"),

    # Proporções
    ("teste_z_1prop", "Teste Z (uma proporção)"),
    ("teste_z_2prop", "Teste Z (duas proporções)"),

    # Categóricos
    ("qui_quadrado", "Qui-quadrado"),
    ("fisher", "Teste exato de Fisher"),
    ("mcnemar", "McNemar"),

    # Não paramétricos
    ("teste_sinal", "Teste do sinal"),
    ("wilcoxon", "Wilcoxon"),
    ("mann_whitney", "Mann-Whitney U"),
    ("kruskal_wallis", "Kruskal-Wallis"),
    ("friedman", "Friedman"),

    # ANOVA
    ("anova_1fator", "ANOVA (1 fator)"),
    ("anova_2fator", "ANOVA (2 fatores)"),

    # Regressão / diagnóstico
    ("regres_linear", "Regressão Linear"),
    ("regres_logit", "Regressão Logística"),
    ("durbin_watson", "Durbin-Watson"),
    ("breusch_pagan", "Breusch-Pagan"),
    ("white", "White"),
    ("vif", "VIF"),
]

COR_DESTAQUE = "#FF8A3D"
COR_FUNDO_SELECIONADO = "#FFF1E6"
COR_TEXTO = "#333333"
COR_BORDA = "#D0D0D0"

def criar_sidebar_testes(page: ft.Page, on_selecionar=None):

    linhas = {}

    def aplicar_visual(valor):
        for v, linha in linhas.items():
            marcado = v == valor
            row = linha.content
            texto = row.controls[1]

            # substitui o ícone inteiro em vez de mudar .name no existente
            row.controls[0] = ft.Icon(
                ft.Icons.RADIO_BUTTON_CHECKED if marcado else ft.Icons.RADIO_BUTTON_UNCHECKED,
                size=20,
                color=COR_DESTAQUE if marcado else COR_BORDA,
            )

            linha.bgcolor = COR_FUNDO_SELECIONADO if marcado else "transparent"
            texto.color = COR_DESTAQUE if marcado else COR_TEXTO
            texto.weight = ft.FontWeight.W_600 if marcado else ft.FontWeight.NORMAL

    def selecionar(valor):
        aplicar_visual(valor)
        for linha in linhas.values():
            linha.update()  # update direto no controle, mais confiável que só page.update()
        if on_selecionar:
            on_selecionar(valor)

    linhas_controls = []
    for valor, rotulo in testes_hipotese:
        icone = ft.Icon(ft.Icons.RADIO_BUTTON_UNCHECKED, size=20, color=COR_BORDA)
        texto = ft.Text(rotulo, size=14, color=COR_TEXTO)
        linha = ft.Container(
            content=ft.Row(controls=[icone, texto], spacing=10),
            padding=ft.Padding.symmetric(horizontal=12, vertical=10),
            border_radius=10,
            bgcolor="transparent",
            on_click=lambda e, v=valor: selecionar(v),
            ink=True,
        )
        linhas[valor] = linha
        linhas_controls.append(linha)

    aplicar_visual(testes_hipotese[0][0])  # só define o estado inicial, sem chamar update

    content = ft.Container(
        content=ft.Column(
            controls=linhas_controls,
            spacing=4,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        ),
        expand=True,
        bgcolor="white",
        padding=1,
    )

    return content




