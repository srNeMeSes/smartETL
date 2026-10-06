"""Card que compara os p-valores de um teste paramétrico e do equivalente não paramétrico."""

import flet as ft

from app.ui import tema
from app.ui.helpers import esta_na_pagina
from core.base import ComparacaoPValores

SEM_VALOR = "—"
NOTA_RODAPE = "Os valores de p (p-value) serão calculados após a execução do teste de hipótese."

# Ícones (esquerda, direita) de cada linha: ≠, >, <
_ICONES_LINHAS = [
    (ft.Icons.SHOW_CHART, ft.Icons.BAR_CHART),
    (ft.Icons.TRENDING_UP, ft.Icons.TRENDING_UP),
    (ft.Icons.TRENDING_DOWN, ft.Icons.TRENDING_DOWN),
]


def formatar_p_valor(p: float | None) -> str:
    """p-valor com vírgula decimal; "—" quando ainda não calculado."""
    if p is None:
        return SEM_VALOR
    if p < 0.001:
        return "< 0,001"
    return f"{p:.3f}".replace(".", ",")


def _icone_caixa(icone: str, cor: str, cor_fundo: str) -> ft.Container:
    """Ícone dentro de uma caixinha arredondada."""
    return ft.Container(
        content=ft.Icon(icone, color=cor, size=22),
        width=48,
        height=48,
        bgcolor=cor_fundo,
        border_radius=12,
        alignment=ft.Alignment.CENTER,
    )


def _titulo_com_sublinhado(texto: str, cor: str) -> ft.Column:
    """Título do cabeçalho com uma barrinha colorida embaixo."""
    return ft.Column(
        controls=[
            ft.Text(texto, size=19, weight=ft.FontWeight.BOLD, color=tema.TEXTO),
            ft.Container(height=3, width=max(60, len(texto) * 10), bgcolor=cor, border_radius=2),
        ],
        spacing=6,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        tight=True,
    )


def _cabecalho(titulo_esquerda: str, titulo_direita: str) -> ft.Row:
    """Linha superior: título A | ícone central (círculo) | título B."""
    return ft.Row(
        controls=[
            ft.Container(
                content=_titulo_com_sublinhado(titulo_esquerda, tema.LARANJA),
                expand=1,
                alignment=ft.Alignment.CENTER,
            ),
            ft.Container(width=1, height=36, bgcolor=tema.BORDA),
            ft.Container(
                content=ft.Icon(ft.Icons.BALANCE, color=tema.TEXTO_SOBRE_DESTAQUE, size=22),
                width=56,
                height=56,
                bgcolor=tema.LARANJA,
                border_radius=28,
                alignment=ft.Alignment.CENTER,
            ),
            ft.Container(width=1, height=36, bgcolor=tema.BORDA),
            ft.Container(
                content=_titulo_com_sublinhado(titulo_direita, tema.ROXO),
                expand=1,
                alignment=ft.Alignment.CENTER,
            ),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


class CardComparacaoTestes(ft.Container):
    """Card comparativo entre dois testes de hipótese.

    Guarda referência aos `ft.Text` de cada p-valor, para que `aplicar(...)`,
    `atualizar_p_values(...)` ou `atualizar_p_value_linha(...)` mudem os valores
    sem reconstruir o componente.
    """

    def __init__(
        self,
        titulo_esquerda: str = "t Student",
        titulo_direita: str = "Wilcoxon",
        linhas: list[dict] | None = None,
        nota_rodape: str = NOTA_RODAPE,
    ):
        self._linhas_config = linhas if linhas is not None else self._linhas_padrao()
        self._textos_p_esquerda: list[ft.Text] = []
        self._textos_p_direita: list[ft.Text] = []

        corpo = ft.Column(
            controls=[_cabecalho(titulo_esquerda, titulo_direita)],
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
        )
        corpo.controls.append(ft.Divider(height=1, color=tema.BORDA))
        for i, linha in enumerate(self._linhas_config):
            corpo.controls.append(
                self._criar_linha_comparacao(
                    linha["icone_esquerda"],
                    linha["icone_direita"],
                    linha["p_esquerda"],
                    linha["hipotese"],
                    linha["p_direita"],
                )
            )
            if i < len(self._linhas_config) - 1:
                corpo.controls.append(ft.Divider(height=1, color=tema.BORDA))

        rodape = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.INFO_OUTLINE, color=tema.LARANJA, size=18),
                    ft.Text(nota_rodape, size=12, color=tema.TEXTO_SECUNDARIO, expand=1),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=tema.LARANJA_SUAVE,
            padding=ft.Padding.symmetric(vertical=14, horizontal=22),
            border_radius=ft.BorderRadius.only(
                bottom_left=tema.RAIO_CARTAO, bottom_right=tema.RAIO_CARTAO
            ),
        )

        super().__init__(
            content=ft.Column(
                controls=[
                    ft.Container(
                        content=corpo,
                        padding=ft.Padding.symmetric(vertical=14, horizontal=26),
                        expand=True,
                    ),
                    rodape,
                ],
                spacing=0,
            ),
            bgcolor=tema.CARTAO,
            border=ft.Border.all(1, tema.BORDA),
            border_radius=tema.RAIO_CARTAO,
            shadow=ft.BoxShadow(
                blur_radius=24,
                spread_radius=0,
                color=tema.SOMBRA_CARTAO,
                offset=ft.Offset(0, 6),
            ),
            expand=True,
        )

    @classmethod
    def de_comparacao(cls, comparacao: ComparacaoPValores) -> "CardComparacaoTestes":
        """Cria o card a partir da estrutura declarada pelo teste."""
        linhas = []
        for i, (hipotese, (p_esq, p_dir)) in enumerate(
            zip(comparacao.hipoteses, comparacao.linhas, strict=True)
        ):
            icone_esq, icone_dir = _ICONES_LINHAS[i % len(_ICONES_LINHAS)]
            linhas.append(
                {
                    "icone_esquerda": icone_esq,
                    "icone_direita": icone_dir,
                    "p_esquerda": formatar_p_valor(p_esq),
                    "hipotese": f"Hₐ:  {hipotese}",
                    "p_direita": formatar_p_valor(p_dir),
                }
            )
        return cls(comparacao.titulo_esquerda, comparacao.titulo_direita, linhas)

    @staticmethod
    def _linhas_padrao() -> list[dict]:
        """Os três cenários padrão (diferente / maior / menor), ainda sem p-valores."""
        hipoteses = ["μ¹ ≠ μ²", "μ¹ > μ²", "μ¹ < μ²"]
        return [
            {
                "icone_esquerda": esq,
                "icone_direita": dir_,
                "p_esquerda": SEM_VALOR,
                "hipotese": f"Hₐ:  {hipotese}",
                "p_direita": SEM_VALOR,
            }
            for (esq, dir_), hipotese in zip(_ICONES_LINHAS, hipoteses, strict=True)
        ]

    def _criar_linha_comparacao(
        self,
        icone_esquerda: str,
        icone_direita: str,
        p_value_esquerda: str,
        hipotese: str,
        p_value_direita: str,
    ) -> ft.Row:
        """Monta uma linha e guarda os ft.Text de p-valor."""
        texto_p_esquerda = ft.Text(
            p_value_esquerda, size=19, weight=ft.FontWeight.BOLD, color=tema.LARANJA
        )
        texto_p_direita = ft.Text(
            p_value_direita, size=19, weight=ft.FontWeight.BOLD, color=tema.ROXO
        )
        self._textos_p_esquerda.append(texto_p_esquerda)
        self._textos_p_direita.append(texto_p_direita)

        def rotulo_p() -> ft.Text:
            return ft.Text(
                "p-value", size=14, color=tema.TEXTO_SECUNDARIO, weight=ft.FontWeight.BOLD
            )

        lado_esquerdo = ft.Row(
            controls=[
                _icone_caixa(icone_esquerda, tema.LARANJA, tema.LARANJA_SUAVE),
                ft.Column(controls=[rotulo_p(), texto_p_esquerda], spacing=2, tight=True),
            ],
            spacing=12,
            tight=True,
        )
        centro = ft.Column(
            controls=[
                ft.Text(
                    "•  Hipótese  •",
                    size=13,
                    color=tema.TEXTO_SECUNDARIO,
                    weight=ft.FontWeight.W_600,
                ),
                ft.Container(
                    content=ft.Text(
                        hipotese, size=15, color=tema.TEXTO_SECUNDARIO, weight=ft.FontWeight.W_500
                    ),
                    padding=ft.Padding.symmetric(vertical=8, horizontal=16),
                    bgcolor=tema.FUNDO,
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
                    controls=[rotulo_p(), texto_p_direita],
                    spacing=2,
                    horizontal_alignment=ft.CrossAxisAlignment.END,
                    tight=True,
                ),
                _icone_caixa(icone_direita, tema.ROXO, tema.ROXO_SUAVE),
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
    # API pública
    # ----------------------------------------------------------------
    def _esta_na_pagina(self) -> bool:
        return esta_na_pagina(self)

    def _atualizar_se_montado(self, atualizar_pagina: bool) -> None:
        if atualizar_pagina and self._esta_na_pagina():
            self.update()

    def aplicar(self, comparacao: ComparacaoPValores, atualizar_pagina: bool = True) -> None:
        """Preenche os p-valores a partir de `ResultadoTeste.comparacao`."""
        for i, (p_esq, p_dir) in enumerate(comparacao.linhas[: len(self._textos_p_esquerda)]):
            self._textos_p_esquerda[i].value = formatar_p_valor(p_esq)
            self._textos_p_direita[i].value = formatar_p_valor(p_dir)
        self._atualizar_se_montado(atualizar_pagina)

    def atualizar_p_values(self, valores: list[dict], atualizar_pagina: bool = True) -> None:
        """Atualiza várias linhas; chaves "esquerda"/"direita" com None preservam o valor."""
        for i, valor in enumerate(valores):
            if i >= len(self._textos_p_esquerda):
                break
            if valor.get("esquerda") is not None:
                self._textos_p_esquerda[i].value = valor["esquerda"]
            if valor.get("direita") is not None:
                self._textos_p_direita[i].value = valor["direita"]
        self._atualizar_se_montado(atualizar_pagina)

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
        self._atualizar_se_montado(atualizar_pagina)
