"""Gráficos nativos do Flet (flet.canvas), simples e minimalistas, a partir de `Figura`."""

import flet as ft
import flet.canvas as cv

from app.ui import tema
from core.base import Figura
from core.interpretacao import formatar_numero

LARGURA = 640
ALTURA = 260
_MARGEM_ESQ, _MARGEM_DIR, _MARGEM_TOPO, _MARGEM_BASE = 36, 16, 26, 26


def desenhar_figura(figura: Figura, largura: float = LARGURA, altura: float = ALTURA) -> ft.Control:
    """Título + gráfico + legenda."""
    if figura.tipo == "histograma":
        grafico, legenda = histograma(figura.dados, largura, altura)
    else:
        raise ValueError(f"Tipo de figura desconhecido: {figura.tipo}")
    controles: list[ft.Control] = [
        ft.Text(figura.titulo, size=14, weight=ft.FontWeight.W_600, color=tema.TEXTO),
        grafico,
    ]
    if legenda is not None:
        controles.append(legenda)
    return ft.Column(controles, spacing=8)


def _texto(x: float, y: float, valor: str, alinhamento: ft.Alignment) -> cv.Text:
    return cv.Text(
        x,
        y,
        valor,
        style=ft.TextStyle(size=11, color=tema.TEXTO_SECUNDARIO),
        alignment=alinhamento,
    )


def _paint_referencia(estilo: str) -> ft.Paint:
    if estilo == "tracejado":
        return ft.Paint(
            color=tema.GRAFICO_REFERENCIA,
            stroke_width=2,
            style=ft.PaintingStyle.STROKE,
            stroke_dash_pattern=[5, 4],
        )
    return ft.Paint(color=tema.GRAFICO_DESTAQUE, stroke_width=2, style=ft.PaintingStyle.STROKE)


def histograma(dados: dict, largura: float, altura: float) -> tuple[cv.Canvas, ft.Control | None]:
    bordas: list[float] = dados["bordas"]
    contagens: list[int] = dados["contagens"]
    referencias: list[dict] = dados.get("referencias", [])

    x0, x1 = bordas[0], bordas[-1]
    maximo = max(contagens) or 1
    area_l = largura - _MARGEM_ESQ - _MARGEM_DIR
    area_a = altura - _MARGEM_TOPO - _MARGEM_BASE
    base = _MARGEM_TOPO + area_a

    def sx(valor: float) -> float:
        return _MARGEM_ESQ + (valor - x0) / (x1 - x0) * area_l

    formas: list[cv.Shape] = []
    preenchimento = ft.Paint(color=tema.GRAFICO_BARRA, style=ft.PaintingStyle.FILL)
    contorno = ft.Paint(
        color=tema.GRAFICO_BARRA_BORDA, stroke_width=1, style=ft.PaintingStyle.STROKE
    )
    for inicio, fim, contagem in zip(bordas[:-1], bordas[1:], contagens, strict=True):
        if contagem == 0:
            continue
        esquerda, direita = sx(inicio) + 1, sx(fim) - 1
        topo = base - contagem / maximo * area_a
        largura_barra = max(direita - esquerda, 1)
        for paint in (preenchimento, contorno):
            formas.append(cv.Rect(esquerda, topo, largura_barra, base - topo, paint=paint))

    eixo = ft.Paint(color=tema.BORDA, stroke_width=1, style=ft.PaintingStyle.STROKE)
    formas.append(cv.Line(_MARGEM_ESQ, base, _MARGEM_ESQ + area_l, base, paint=eixo))
    formas.append(cv.Line(_MARGEM_ESQ, _MARGEM_TOPO, _MARGEM_ESQ, base, paint=eixo))
    formas.append(_texto(_MARGEM_ESQ - 6, base, "0", ft.Alignment.CENTER_RIGHT))
    formas.append(_texto(_MARGEM_ESQ - 6, _MARGEM_TOPO, str(maximo), ft.Alignment.CENTER_RIGHT))
    for valor in (x0, (x0 + x1) / 2, x1):
        formas.append(
            _texto(sx(valor), base + 6, formatar_numero(valor, 2), ft.Alignment.TOP_CENTER)
        )

    for ref in referencias:
        x = sx(ref["valor"])
        formas.append(cv.Line(x, _MARGEM_TOPO, x, base, paint=_paint_referencia(ref["estilo"])))

    canvas = cv.Canvas(width=largura, height=altura, shapes=formas)
    legenda = _legenda(referencias) if referencias else None
    return canvas, legenda


def _legenda(referencias: list[dict]) -> ft.Row:
    itens = []
    for ref in referencias:
        tracejado = ref["estilo"] == "tracejado"
        cor = tema.GRAFICO_REFERENCIA if tracejado else tema.GRAFICO_DESTAQUE
        amostra = ft.Row(
            [ft.Container(width=5, height=2, bgcolor=cor) for _ in range(3 if tracejado else 1)],
            spacing=3,
            tight=True,
        )
        if not tracejado:
            amostra.controls[0].width = 21
        itens.append(
            ft.Row(
                [amostra, ft.Text(ref["rotulo"], size=12, color=tema.TEXTO)],
                spacing=6,
                tight=True,
            )
        )
    return ft.Row(itens, spacing=20)
