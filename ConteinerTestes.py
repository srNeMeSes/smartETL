import flet as ft


# --------------------------------------------------------------------------
# Paleta de cores do card
# --------------------------------------------------------------------------
COR_LARANJA = "#F97316"
COR_LARANJA_CLARO = "#FEF1E6"
COR_ROXO = "#7C3AED"
COR_ROXO_CLARO = "#EDE9FE"
COR_CINZA_TEXTO = "#6B7280"
COR_CINZA_ESCURO = "#111827"
COR_BORDA = "#E5E7EB"
COR_CHIP_BG = "#F4F4F5"
COR_RODAPE_BG = "#FDF1E7"


# --------------------------------------------------------------------------
# Peças reutilizáveis (não guardam estado, então continuam como funções soltas)
# --------------------------------------------------------------------------
def _icone_caixa(icone: str, cor: str, cor_fundo: str) -> ft.Container:
    """Ícone dentro de uma caixinha arredondada (estilo dos ícones do card)."""
    return ft.Container(
        content=ft.Icon(icone, color=cor, size=22),
        width=48,
        height=48,
        bgcolor=cor_fundo,
        border_radius=12,
        alignment=ft.Alignment.CENTER,
    )


def _titulo_com_sublinhado(texto: str, cor: str) -> ft.Column:
    """Título do cabeçalho com uma barrinha colorida embaixo, como na imagem."""
    return ft.Column(
        controls=[
            ft.Text(texto, size=19, weight=ft.FontWeight.BOLD, color=COR_CINZA_ESCURO),
            ft.Container(height=3, width=max(60, len(texto) * 10), bgcolor=cor, border_radius=2),
        ],
        spacing=6,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        tight=True,
    )


def _cabecalho(titulo_esquerda: str, titulo_direita: str, icone_central: str) -> ft.Row:
    """Linha superior: título A | ícone central (círculo) | título B."""
    return ft.Row(
        controls=[
            ft.Container(
                content=_titulo_com_sublinhado(titulo_esquerda, COR_LARANJA),
                expand=1,
                alignment=ft.Alignment.CENTER,
            ),
            ft.Container(width=1, height=36, bgcolor=COR_BORDA),
            ft.Container(
                content=ft.Icon(icone_central, color=ft.Colors.WHITE, size=22),
                width=56,
                height=56,
                bgcolor=COR_LARANJA,
                border_radius=28,
                alignment=ft.Alignment.CENTER,
            ),
            ft.Container(width=1, height=36, bgcolor=COR_BORDA),
            ft.Container(
                content=_titulo_com_sublinhado(titulo_direita, COR_ROXO),
                expand=1,
                alignment=ft.Alignment.CENTER,
            ),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


# --------------------------------------------------------------------------
# Classe reaproveitável
# --------------------------------------------------------------------------
class CardComparacaoTestes(ft.Container):
    """
    Card comparativo entre dois testes de hipótese.

    Diferente de uma função solta, a classe guarda referência aos
    ft.Text de cada p-value, permitindo chamar `atualizar_p_values(...)`
    (ou `atualizar_p_value_linha(...)`) depois que o card já está na
    tela, sem precisar reconstruir o componente inteiro.
    """

    def __init__(
        self,
        titulo_esquerda: str = "t Student",
        titulo_direita: str = "Wilcoxon",
        linhas: list[dict] | None = None,
        nota_rodape: str = (
            "Os valores de p (p-value) serão calculados após a execução do teste de hipótese."
        ),
        
    ):
        self._linhas_config = linhas if linhas is not None else self._linhas_padrao()

        # Referências aos textos de p-value, na mesma ordem das linhas,
        # usadas depois por atualizar_p_values / atualizar_p_value_linha.
        self._textos_p_esquerda: list[ft.Text] = []
        self._textos_p_direita: list[ft.Text] = []

        corpo = ft.Column(
            controls=[_cabecalho(titulo_esquerda, titulo_direita, ft.Icons.BALANCE)],
            spacing=10,
            scroll='auto'
           
        )
        corpo.controls.append(ft.Divider(height=1, color=COR_BORDA))

        for i, linha in enumerate(self._linhas_config):
            linha_row = self._criar_linha_comparacao(
                linha["icone_esquerda"],
                linha["icone_direita"],
                linha["p_esquerda"],
                linha["hipotese"],
                linha["p_direita"],
            )
            corpo.controls.append(linha_row)
            if i < len(self._linhas_config) - 1:
                corpo.controls.append(ft.Divider(height=1, color=COR_BORDA))

        rodape = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.INFO_OUTLINE, color=COR_LARANJA, size=18),
                    ft.Text(nota_rodape, size=12, color=COR_CINZA_TEXTO, expand=1),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=COR_RODAPE_BG,
            padding=ft.Padding.symmetric(vertical=14, horizontal=22),
            border_radius=ft.BorderRadius.only(bottom_left=16, bottom_right=16),
        )

        super().__init__(
            content=ft.Column(
                controls=[
                    ft.Container(content=corpo, padding=ft.Padding.symmetric(vertical=14, horizontal=26), expand=True),
                    rodape,
                ],
                spacing=0,
            ),
            bgcolor=ft.Colors.WHITE,
            border=ft.Border.all(1, COR_BORDA),
            border_radius=16,
            shadow=ft.BoxShadow(
                blur_radius=24,
                spread_radius=0,
                color=ft.Colors.with_opacity(0.06, ft.Colors.BLACK),
                offset=ft.Offset(0, 6),
            ),
            
            expand=True
        )

    # ----------------------------------------------------------------
    # Construção interna
    # ----------------------------------------------------------------
    @staticmethod
    def _linhas_padrao() -> list[dict]:
        """Os três cenários padrão (diferente / maior / menor), como na imagem."""
        return [
            {
                "icone_esquerda": ft.Icons.SHOW_CHART,
                "icone_direita": ft.Icons.BAR_CHART,
                "p_esquerda": "0.001",
                "hipotese": "Hₐ:  μ¹ ≠ μ²",
                "p_direita": "0.001",
            },
            {
                "icone_esquerda": ft.Icons.TRENDING_UP,
                "icone_direita": ft.Icons.TRENDING_UP,
                "p_esquerda": "0.001",
                "hipotese": "Hₐ:  μ¹ > μ²",
                "p_direita": "0.001",
            },
            {
                "icone_esquerda": ft.Icons.TRENDING_DOWN,
                "icone_direita": ft.Icons.TRENDING_DOWN,
                "p_esquerda": "0.001",
                "hipotese": "Hₐ:  μ¹ < μ²",
                "p_direita": "0.001",
            },
        ]

    def _criar_linha_comparacao(
        self,
        icone_esquerda: str,
        icone_direita: str,
        p_value_esquerda: str,
        hipotese: str,
        p_value_direita: str,
    ) -> ft.Row:
        """Monta uma linha e guarda os ft.Text de p-value em self.*."""
        texto_p_esquerda = ft.Text(p_value_esquerda, size=19, weight=ft.FontWeight.BOLD, color=COR_LARANJA)
        texto_p_direita = ft.Text(p_value_direita, size=19, weight=ft.FontWeight.BOLD, color=COR_ROXO)

        self._textos_p_esquerda.append(texto_p_esquerda)
        self._textos_p_direita.append(texto_p_direita)

        lado_esquerdo = ft.Row(
            controls=[
                _icone_caixa(icone_esquerda, COR_LARANJA, COR_LARANJA_CLARO),
                ft.Column(
                    controls=[
                    ft.Text("p-value", size=14, color=COR_CINZA_TEXTO, weight=ft.FontWeight.BOLD),
                    texto_p_esquerda
                    ],
                    spacing=2,
                    tight=True,
                ),
            ],
            spacing=12,
            tight=True,
        )

        centro = ft.Column(
            controls=[
                ft.Text("•  Hipótese  •", size=13, color=COR_CINZA_TEXTO, weight=ft.FontWeight.W_600),
                ft.Container(
                    content=ft.Text(hipotese, size=15, color=COR_CINZA_TEXTO, weight=ft.FontWeight.W_500),
                    padding=ft.Padding.symmetric(vertical=8, horizontal=16),
                    bgcolor=COR_CHIP_BG,
                    border_radius=8,
                ),
            ],
            spacing=6,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            tight=True,
        )

        lado_direito = ft.Row(
            controls=[
                ft.Column(
                    controls=[
                    ft.Text("p-value", size=14, color=COR_CINZA_TEXTO, weight=ft.FontWeight.BOLD), texto_p_direita],
                    spacing=2,
                    horizontal_alignment=ft.CrossAxisAlignment.END,
                    tight=True,
                ),
                _icone_caixa(icone_direita, COR_ROXO, COR_ROXO_CLARO),
            ],
            spacing=12,
            tight=True,
        )

        return ft.Row(
            controls=[
                ft.Container(content=lado_esquerdo, expand=1, alignment=ft.Alignment.CENTER_LEFT),
                ft.Container(content=centro, expand=1, alignment=ft.Alignment.CENTER),
                ft.Container(content=lado_direito, expand=1, alignment=ft.Alignment.CENTER_RIGHT),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    # ----------------------------------------------------------------
    # API pública para atualizar os p-values depois de o card já estar na tela
    # ----------------------------------------------------------------
    def atualizar_p_values(self, valores: list[dict], atualizar_pagina: bool = True) -> None:
        for i, valor in enumerate(valores):
            if i >= len(self._textos_p_esquerda):
                break
            if "esquerda" in valor and valor["esquerda"] is not None:
                self._textos_p_esquerda[i].value = valor["esquerda"]
            if "direita" in valor and valor["direita"] is not None:
                self._textos_p_direita[i].value = valor["direita"]

        if atualizar_pagina and self.page is not None:
            self.update()

    def atualizar_p_value_linha(
        self,
        indice: int,
        esquerda: str | None = None,
        direita: str | None = None,
        atualizar_pagina: bool = True,
    ) -> None:
        """Atualiza apenas uma linha, pelo índice (0 = primeira linha do card)."""
        if not (0 <= indice < len(self._textos_p_esquerda)):
            raise IndexError(f"Índice de linha inválido: {indice}")

        if esquerda is not None:
            self._textos_p_esquerda[indice].value = esquerda
        if direita is not None:
            self._textos_p_direita[indice].value = direita

        if atualizar_pagina and self.page is not None:
            self.update()


