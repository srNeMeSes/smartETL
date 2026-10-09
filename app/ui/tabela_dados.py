"""Prévia do dataset: estado vazio (colunas fantasmas) separado do estado com dados.

Grade própria em vez de `DataTable`: o cabeçalho fica fixo, o corpo (um `ListView`, que só monta as
linhas visíveis) rola na vertical e a grade inteira rola na horizontal, com a barra sempre visível
na base do cartão. No `DataTable` dentro de dois scrolls, a barra horizontal ficava depois da
última linha e não dava para arrastar a tabela para o lado sem antes rolar até o fim.

A largura de cada coluna é estimada pelo texto mais longo (nome ou valor) entre `LARGURA_MINIMA` e
`LARGURA_MAXIMA`; textos maiores terminam em "…" e mostram o valor inteiro ao passar o mouse.
Números ficam à direita, textos à esquerda. A prévia mostra até `MAX_LINHAS` linhas: o Flet envia
cada célula à janela, e mais linhas só deixariam a troca de arquivo mais lenta.
"""

from numbers import Integral, Real

import flet as ft
import numpy as np
import pandas as pd

from app.ui import tema
from app.ui.helpers import border_all, border_only, esta_na_pagina, pad

MAX_LINHAS = 100
COLUNAS_VAZIAS = 20
LINHAS_VAZIAS = 12

ALTURA = 320
ALTURA_CABECALHO = 40
ALTURA_LINHA = 34
ALTURA_RODAPE = 32
ALTURA_GRADE = ALTURA - ALTURA_RODAPE - 2  # 2 = borda de cima e de baixo do cartão
MARGEM = 16  # à esquerda e à direita de cada linha
ESPACO = 24  # entre colunas
LARGURA_NUMERO = 36  # coluna com o número da linha
LARGURA_MINIMA = 72
LARGURA_MAXIMA = 260
LARGURA_FANTASMA = 96
LARGURA_ICONE = 20  # ícone do tipo + espaço, no cabeçalho
PX_POR_CARACTERE = 7.4  # média aproximada da fonte de 13 px
FONTE_CELULA = 13
FONTE_CABECALHO = 12.5

# Barras sempre visíveis; a cor vem de `tema.tema_da_pagina` (no Flet 0.86.2 o tema local de um
# Container não muda a cor da barra).
BARRA = ft.Scrollbar(thumb_visibility=True, interactive=True, thickness=9, radius=5)


def formatar_celula(valor: object) -> str:
    """Texto de uma célula; NaN/None viram vazio e números decimais usam vírgula."""
    try:
        if pd.isna(valor):
            return ""
    except (TypeError, ValueError):  # valores não escalares
        pass
    if isinstance(valor, Real) and not isinstance(valor, (Integral, bool, np.bool_)):
        return str(valor).replace(".", ",")
    return str(valor)


def _formatar_inteiro(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _quantidade(n: int, singular: str, plural: str) -> str:
    return f"{_formatar_inteiro(n)} {singular if n == 1 else plural}"


def _largura_texto(caracteres: int) -> float:
    return caracteres * PX_POR_CARACTERE


def _icone_do_tipo(serie: pd.Series) -> tuple[str, str]:
    """Ícone e dica do tipo da coluna, só para leitura rápida do cabeçalho."""
    if pd.api.types.is_bool_dtype(serie):
        return ft.Icons.TOGGLE_ON_OUTLINED, "Lógica"
    if pd.api.types.is_numeric_dtype(serie):
        return ft.Icons.NUMBERS, "Numérica"
    if pd.api.types.is_datetime64_any_dtype(serie):
        return ft.Icons.CALENDAR_TODAY_OUTLINED, "Data"
    return ft.Icons.TEXT_FIELDS, "Texto"


def _alinhamento(serie: pd.Series) -> ft.TextAlign:
    numerica = pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie)
    return ft.TextAlign.RIGHT if numerica else ft.TextAlign.LEFT


class _Coluna:
    """O que a grade precisa saber de uma coluna para desenhá-la."""

    def __init__(self, nome: str, largura: float, alinhamento: ft.TextAlign, icone=None, dica=""):
        self.nome = nome
        self.largura = largura
        self.alinhamento = alinhamento
        self.icone = icone
        self.dica = dica


def _largura_coluna(nome: str, valores: list[str], com_icone: bool) -> float:
    maior = max((len(v) for v in valores), default=0)
    cabecalho = _largura_texto(len(nome)) + (LARGURA_ICONE if com_icone else 0)
    return float(min(LARGURA_MAXIMA, max(LARGURA_MINIMA, cabecalho, _largura_texto(maior))))


def _celula(texto: str, coluna: _Coluna) -> ft.Text:
    cabe = _largura_texto(len(texto)) <= coluna.largura
    return ft.Text(
        texto,
        width=coluna.largura,
        size=FONTE_CELULA,
        color=tema.TEXTO,
        text_align=coluna.alinhamento,
        no_wrap=True,
        max_lines=1,
        overflow=ft.TextOverflow.ELLIPSIS,
        tooltip=None if cabe else texto,
    )


def _numero(texto: str) -> ft.Text:
    return ft.Text(
        texto,
        width=LARGURA_NUMERO,
        size=12,
        color=tema.TEXTO_TERCIARIO,
        text_align=ft.TextAlign.RIGHT,
        no_wrap=True,
    )


def _titulo(coluna: _Coluna, fantasma: bool) -> ft.Text:
    largura = coluna.largura - (LARGURA_ICONE if coluna.icone else 0)
    return ft.Text(
        coluna.nome,
        width=largura,
        size=FONTE_CABECALHO,
        weight=ft.FontWeight.W_600,
        color=tema.TEXTO_TERCIARIO if fantasma else tema.TEXTO_SECUNDARIO,
        text_align=coluna.alinhamento,
        no_wrap=True,
        max_lines=1,
        overflow=ft.TextOverflow.ELLIPSIS,
        tooltip=coluna.nome if _largura_texto(len(coluna.nome)) > largura else None,
    )


def _celula_cabecalho(coluna: _Coluna, titulo: ft.Text) -> ft.Control:
    if coluna.icone is None:
        return titulo
    icone = ft.Icon(coluna.icone, size=14, color=tema.TEXTO_TERCIARIO, tooltip=coluna.dica)
    direita = coluna.alinhamento == ft.TextAlign.RIGHT
    return ft.Row(
        [titulo, icone] if direita else [icone, titulo],
        spacing=LARGURA_ICONE - 14,
        width=coluna.largura,
        alignment=ft.MainAxisAlignment.END if direita else ft.MainAxisAlignment.START,
    )


class TabelaDados(ft.Container):
    """Cartão de 320 px: cabeçalho fixo, rolagem vertical das linhas e horizontal da grade.

    `titulos` são os nomes das colunas (sem a coluna do número da linha) e `linhas`, as células de
    cada linha exibida, na mesma ordem.
    """

    def __init__(self):
        self.vazia = True
        self.titulos: list[ft.Text] = []
        self.linhas: list[list[ft.Text]] = []
        self._largura_conteudo = 0.0
        self._largura_visivel = 0.0
        self.cabecalho = ft.Container(
            height=ALTURA_CABECALHO,
            padding=pad(horizontal=MARGEM),
            bgcolor=tema.TABELA_CABECALHO,
            border=border_only(bottom=ft.BorderSide(1, tema.BORDA)),
        )
        self.corpo = ft.ListView(
            item_extent=ALTURA_LINHA,
            scroll=BARRA,
            padding=pad(bottom=12),  # a barra horizontal não cobre a última linha
            expand=True,
        )
        self.grade = ft.Container(
            content=ft.Column([self.cabecalho, self.corpo], spacing=0),
            height=ALTURA_GRADE,
        )
        self.rodape = ft.Text(size=12, color=tema.TEXTO_SECUNDARIO)
        self.mostrar_vazio()
        super().__init__(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[self.grade],
                        scroll=BARRA,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                        on_size_change=self._ao_redimensionar,
                        size_change_interval=100,
                        height=ALTURA_GRADE,
                    ),
                    ft.Container(
                        content=self.rodape,
                        height=ALTURA_RODAPE,
                        padding=pad(horizontal=MARGEM),
                        alignment=ft.Alignment.CENTER_LEFT,
                        border=border_only(top=ft.BorderSide(1, tema.BORDA)),
                    ),
                ],
                spacing=0,
            ),
            height=ALTURA,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            border=border_all(1, tema.BORDA),
            border_radius=tema.RAIO_PEQUENO,
            bgcolor=tema.CARTAO,
        )

    # ---------------- Estados ----------------
    def mostrar_vazio(self) -> None:
        """Estado sem arquivo: colunas `column1..20` e linhas em branco."""
        self.vazia = True
        colunas = [
            _Coluna(f"column{i}", LARGURA_FANTASMA, ft.TextAlign.LEFT)
            for i in range(1, COLUNAS_VAZIAS + 1)
        ]
        self._desenhar(colunas, [[""] * COLUNAS_VAZIAS] * LINHAS_VAZIAS, fantasma=True)
        self.rodape.value = f"Nenhum arquivo carregado  ·  a prévia mostra até {MAX_LINHAS} linhas"

    def mostrar(self, df: pd.DataFrame) -> None:
        """Só as colunas reais e no máximo `MAX_LINHAS` linhas."""
        self.vazia = False
        amostra = df.head(MAX_LINHAS)
        valores = [[formatar_celula(v) for v in linha] for linha in amostra.itertuples(False, None)]
        colunas = []
        for j, nome in enumerate(df.columns):
            serie = df.iloc[:, j]
            icone, dica = _icone_do_tipo(serie)
            largura = _largura_coluna(str(nome), [linha[j] for linha in valores], True)
            colunas.append(_Coluna(str(nome), largura, _alinhamento(serie), icone, dica))
        self._desenhar(colunas, valores, fantasma=False)
        n_linhas, n_colunas = df.shape
        exibidas = len(valores)
        linhas = (
            _quantidade(n_linhas, "linha", "linhas")
            if exibidas == n_linhas
            else f"primeiras {_formatar_inteiro(exibidas)} de {_formatar_inteiro(n_linhas)} linhas"
        )
        colunas_texto = _quantidade(n_colunas, "coluna", "colunas")
        self.rodape.value = f"Prévia: {linhas}  ·  {colunas_texto}"

    # ---------------- Desenho ----------------
    def _desenhar(self, colunas: list[_Coluna], valores: list[list[str]], fantasma: bool) -> None:
        self.titulos = [_titulo(c, fantasma) for c in colunas]
        self.cabecalho.content = ft.Row(
            [_numero("#" if not fantasma else ""), *map(_celula_cabecalho, colunas, self.titulos)],
            spacing=ESPACO,
        )
        self.linhas = []
        controles = []
        for i, linha in enumerate(valores):
            celulas = [_celula(texto, coluna) for texto, coluna in zip(linha, colunas, strict=True)]
            self.linhas.append(celulas)
            controles.append(
                ft.Container(
                    content=ft.Row(
                        [_numero("" if fantasma else str(i + 1)), *celulas], spacing=ESPACO
                    ),
                    height=ALTURA_LINHA,
                    padding=pad(horizontal=MARGEM),
                    bgcolor=tema.TABELA_ZEBRA if i % 2 else tema.CARTAO,
                )
            )
        self.corpo.controls = controles
        self._largura_conteudo = (
            2 * MARGEM + LARGURA_NUMERO + sum(c.largura + ESPACO for c in colunas)
        )
        self._ajustar_largura()

    def _ajustar_largura(self) -> None:
        # Tabela estreita ocupa o cartão inteiro (cabeçalho e listras até a borda, barra vertical
        # no canto); larga, fica com a própria largura e a barra horizontal permite percorrê-la.
        self.grade.width = max(self._largura_conteudo, self._largura_visivel)

    def _ao_redimensionar(self, e: ft.LayoutSizeChangeEvent) -> None:
        if abs(e.width - self._largura_visivel) < 1:
            return
        self._largura_visivel = e.width
        self._ajustar_largura()
        if esta_na_pagina(self.grade):
            self.grade.update()
