"""Gráficos nativos do Flet (flet.canvas), simples e minimalistas, a partir de `Figura`."""

import itertools

import flet as ft
import flet.canvas as cv

from app.ui import tema
from core.base import Figura
from core.interpretacao import formatar_numero

LARGURA = 640
ALTURA = 220
_MARGEM_ESQ, _MARGEM_DIR, _MARGEM_TOPO, _MARGEM_BASE = 36, 16, 26, 26
FOLGA_NEGATIVOS = 0.12  # fração da amplitude abaixo do menor valor negativo


def desenhar_figura(figura: Figura, largura: float = LARGURA, altura: float = ALTURA) -> ft.Control:
    """Título + gráfico + legenda."""
    if figura.tipo == "histograma":
        grafico, legenda = histograma(figura.dados, largura, altura)
    elif figura.tipo == "boxplot":
        grafico, legenda = boxplot(figura.dados, largura, altura)
    elif figura.tipo == "barras":
        grafico, legenda = barras(figura.dados, largura, altura)
    elif figura.tipo == "barras_agrupadas":
        grafico, legenda = barras_agrupadas(figura.dados, largura, altura)
    elif figura.tipo == "dispersao":
        grafico, legenda = dispersao(figura.dados, largura, altura)
    elif figura.tipo == "cascata":
        grafico, legenda = cascata(figura.dados, largura, altura)
    else:
        raise ValueError(f"Tipo de figura desconhecido: {figura.tipo}")
    titulo = ft.Text(figura.titulo, size=14, weight=ft.FontWeight.W_600, color=tema.TEXTO)
    # Legenda na mesma linha do título: o gráfico inteiro cabe na aba sem rolar.
    cabecalho = ft.Row(
        [titulo, legenda] if legenda is not None else [titulo],
        spacing=32,
        width=largura,
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
    )
    return ft.Column([cabecalho, grafico], spacing=8)


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
    # Rótulos das pontas alinhados para dentro, para não serem cortados pela borda do canvas.
    for valor, alinhamento in (
        (x0, ft.Alignment.TOP_LEFT),
        ((x0 + x1) / 2, ft.Alignment.TOP_CENTER),
        (x1, ft.Alignment.TOP_RIGHT),
    ):
        formas.append(_texto(sx(valor), base + 6, formatar_numero(valor, 2), alinhamento))

    for ref in referencias:
        x = sx(ref["valor"])
        formas.append(cv.Line(x, _MARGEM_TOPO, x, base, paint=_paint_referencia(ref["estilo"])))

    canvas = cv.Canvas(width=largura, height=altura, shapes=formas)
    legenda = _legenda([(r["rotulo"], r["estilo"]) for r in referencias]) if referencias else None
    return canvas, legenda


def _amostra_legenda(estilo: str) -> ft.Control:
    """Pedacinho de linha cheia, tracejada ou um ponto, para a legenda."""
    if estilo == "tracejado":
        return ft.Row(
            [ft.Container(width=5, height=2, bgcolor=tema.GRAFICO_REFERENCIA) for _ in range(3)],
            spacing=3,
            tight=True,
        )
    if estilo == "ponto":
        return ft.Container(width=8, height=8, border_radius=4, bgcolor=tema.GRAFICO_DESTAQUE)
    return ft.Container(width=21, height=2, bgcolor=tema.GRAFICO_DESTAQUE)


def _legenda(itens: list[tuple[str, str]]) -> ft.Row:
    """Itens (rótulo, estilo) com estilo "destaque", "tracejado" ou "ponto"."""
    return ft.Row(
        [
            ft.Row(
                [_amostra_legenda(estilo), ft.Text(rotulo, size=12, color=tema.TEXTO)],
                spacing=6,
                tight=True,
            )
            for rotulo, estilo in itens
        ],
        spacing=20,
    )


def boxplot(dados: dict, largura: float, altura: float) -> tuple[cv.Canvas, ft.Control]:
    """Boxplots verticais lado a lado: caixa Q1–Q3, mediana, bigodes, média (ponto) e outliers."""
    grupos: list[dict] = dados["grupos"]
    esq = 70  # espaço para os rótulos do eixo y
    area_l = largura - esq - _MARGEM_DIR
    area_a = altura - _MARGEM_TOPO - _MARGEM_BASE
    base = _MARGEM_TOPO + area_a

    referencias: list[dict] = dados.get("referencias", [])
    valores = [
        v for g in grupos for v in (g["bigode_inf"], g["bigode_sup"], g["media"], *g["outliers"])
    ]
    valores += [r["valor"] for r in referencias]
    minimo, maximo = min(valores), max(valores)
    if minimo == maximo:
        minimo, maximo = minimo - 0.5, maximo + 0.5
    folga = (maximo - minimo) * 0.08
    minimo, maximo = minimo - folga, maximo + folga

    def sy(valor: float) -> float:
        return base - (valor - minimo) / (maximo - minimo) * area_a

    traco = ft.Paint(color=tema.GRAFICO_REFERENCIA, stroke_width=1, style=ft.PaintingStyle.STROKE)
    preenchimento = ft.Paint(color=tema.GRAFICO_BARRA, style=ft.PaintingStyle.FILL)
    contorno = ft.Paint(
        color=tema.GRAFICO_BARRA_BORDA, stroke_width=1.5, style=ft.PaintingStyle.STROKE
    )
    mediana = ft.Paint(color=tema.GRAFICO_DESTAQUE, stroke_width=2, style=ft.PaintingStyle.STROKE)
    ponto = ft.Paint(color=tema.GRAFICO_DESTAQUE, style=ft.PaintingStyle.FILL)

    eixo = ft.Paint(color=tema.BORDA, stroke_width=1, style=ft.PaintingStyle.STROKE)
    formas: list[cv.Shape] = [
        cv.Line(esq, base, esq + area_l, base, paint=eixo),
        cv.Line(esq, _MARGEM_TOPO, esq, base, paint=eixo),
    ]
    for valor in (minimo, (minimo + maximo) / 2, maximo):
        formas.append(
            _texto(esq - 6, sy(valor), formatar_numero(valor, 2), ft.Alignment.CENTER_RIGHT)
        )

    vaga = area_l / len(grupos)
    meia = min(vaga * 0.22, 45)
    for i, g in enumerate(grupos):
        cx = esq + vaga * (i + 0.5)
        formas += [
            cv.Line(cx, sy(g["bigode_sup"]), cx, sy(g["q3"]), paint=traco),
            cv.Line(cx, sy(g["q1"]), cx, sy(g["bigode_inf"]), paint=traco),
            cv.Line(
                cx - meia / 2, sy(g["bigode_sup"]), cx + meia / 2, sy(g["bigode_sup"]), paint=traco
            ),
            cv.Line(
                cx - meia / 2, sy(g["bigode_inf"]), cx + meia / 2, sy(g["bigode_inf"]), paint=traco
            ),
        ]
        topo, altura_caixa = sy(g["q3"]), max(sy(g["q1"]) - sy(g["q3"]), 1)
        for paint in (preenchimento, contorno):
            formas.append(cv.Rect(cx - meia, topo, 2 * meia, altura_caixa, paint=paint))
        formas.append(
            cv.Line(cx - meia, sy(g["mediana"]), cx + meia, sy(g["mediana"]), paint=mediana)
        )
        formas.append(cv.Circle(cx, sy(g["media"]), 3.5, paint=ponto))
        formas += [cv.Circle(cx, sy(v), 3, paint=contorno) for v in g["outliers"]]
        rotulo = f"{g['rotulo']} (n = {g['n']})"
        formas.append(_texto(cx, base + 6, rotulo, ft.Alignment.TOP_CENTER))

    for ref in referencias:
        y = sy(ref["valor"])
        formas.append(cv.Line(esq, y, esq + area_l, y, paint=_paint_referencia(ref["estilo"])))

    canvas = cv.Canvas(width=largura, height=altura, shapes=formas)
    itens = [("Mediana", "destaque"), ("Média", "ponto")]
    return canvas, _legenda(itens + [(r["rotulo"], r["estilo"]) for r in referencias])


def _formatar_valor(valor: float, percentual: bool) -> str:
    if percentual:
        return f"{formatar_numero(valor * 100, 1)}%"
    return formatar_numero(valor, 2)


def _barra_vertical(zero: float, extremo: float) -> tuple[float, float]:
    """(topo, altura) de uma barra entre a linha do zero e o valor (para cima ou para baixo)."""
    return min(zero, extremo), max(abs(zero - extremo), 1)


def _rotulo_barra(cx: float, topo: float, altura: float, valor: float, texto: str) -> cv.Text:
    """Valor acima da barra positiva e abaixo da negativa."""
    if valor < 0:
        return _texto(cx, topo + altura + 3, texto, ft.Alignment.TOP_CENTER)
    return _texto(cx, topo - 3, texto, ft.Alignment.BOTTOM_CENTER)


def barras(dados: dict, largura: float, altura: float) -> tuple[cv.Canvas, ft.Control | None]:
    """Barras verticais com o valor sobre cada barra e linhas horizontais de referência."""
    categorias: list[dict] = dados["categorias"]
    referencias: list[dict] = dados.get("referencias", [])
    percentual = bool(dados.get("percentual"))
    maximo = float(dados["maximo"])
    minimo = float(dados.get("minimo", 0.0))  # < 0 quando há valores negativos
    if minimo < 0:  # folga abaixo da barra mais baixa para o rótulo do valor não tocar o eixo
        minimo -= (maximo - minimo) * FOLGA_NEGATIVOS
    esq = 56
    area_l = largura - esq - _MARGEM_DIR
    area_a = altura - _MARGEM_TOPO - _MARGEM_BASE
    base = _MARGEM_TOPO + area_a

    def sy(valor: float) -> float:
        return base - (min(max(valor, minimo), maximo) - minimo) / (maximo - minimo) * area_a

    eixo = ft.Paint(color=tema.BORDA, stroke_width=1, style=ft.PaintingStyle.STROKE)
    preenchimento = ft.Paint(color=tema.GRAFICO_BARRA, style=ft.PaintingStyle.FILL)
    contorno = ft.Paint(
        color=tema.GRAFICO_BARRA_BORDA, stroke_width=1.5, style=ft.PaintingStyle.STROKE
    )
    zero = sy(0.0)  # as barras saem da linha do zero (a base, se não houver negativos)
    formas: list[cv.Shape] = [
        cv.Line(esq, zero, esq + area_l, zero, paint=eixo),
        cv.Line(esq, _MARGEM_TOPO, esq, base, paint=eixo),
    ]
    for valor in (minimo, (minimo + maximo) / 2, maximo):
        formas.append(
            _texto(
                esq - 6, sy(valor), _formatar_valor(valor, percentual), ft.Alignment.CENTER_RIGHT
            )
        )

    vaga = area_l / len(categorias)
    meia = min(vaga * 0.3, 60)
    for i, categoria in enumerate(categorias):
        cx = esq + vaga * (i + 0.5)
        topo, altura_barra = _barra_vertical(zero, sy(categoria["valor"]))
        for paint in (preenchimento, contorno):
            formas.append(cv.Rect(cx - meia, topo, 2 * meia, altura_barra, paint=paint))
        valor_txt = _formatar_valor(categoria["valor"], percentual)
        formas.append(_rotulo_barra(cx, topo, altura_barra, categoria["valor"], valor_txt))
        formas.append(_texto(cx, base + 6, categoria["rotulo"], ft.Alignment.TOP_CENTER))

    for ref in referencias:
        y = sy(ref["valor"])
        formas.append(cv.Line(esq, y, esq + area_l, y, paint=_paint_referencia(ref["estilo"])))

    canvas = cv.Canvas(width=largura, height=altura, shapes=formas)
    legenda = _legenda([(r["rotulo"], r["estilo"]) for r in referencias]) if referencias else None
    return canvas, legenda


def cor_serie(indice: int) -> str:
    return tema.GRAFICO_SERIES[indice % len(tema.GRAFICO_SERIES)]


def barras_agrupadas(dados: dict, largura: float, altura: float) -> tuple[cv.Canvas, ft.Control]:
    """Um grupo de barras por categoria do eixo x; cada série com uma cor da paleta."""
    grupos: list[dict] = dados["grupos"]
    series: list[str] = dados["series"]
    percentual = bool(dados.get("percentual"))
    maximo = float(dados["maximo"])
    minimo = float(dados.get("minimo", 0.0))  # < 0 quando há valores negativos
    if minimo < 0:  # folga abaixo da barra mais baixa para o rótulo do valor não tocar o eixo
        minimo -= (maximo - minimo) * FOLGA_NEGATIVOS
    esq = 56
    area_l = largura - esq - _MARGEM_DIR
    area_a = altura - _MARGEM_TOPO - _MARGEM_BASE
    base = _MARGEM_TOPO + area_a

    def sy(valor: float) -> float:
        return base - (min(max(valor, minimo), maximo) - minimo) / (maximo - minimo) * area_a

    eixo = ft.Paint(color=tema.BORDA, stroke_width=1, style=ft.PaintingStyle.STROKE)
    zero = sy(0.0)  # as barras saem da linha do zero (a base, se não houver negativos)
    formas: list[cv.Shape] = [
        cv.Line(esq, zero, esq + area_l, zero, paint=eixo),
        cv.Line(esq, _MARGEM_TOPO, esq, base, paint=eixo),
    ]
    for valor in (minimo, (minimo + maximo) / 2, maximo):
        formas.append(
            _texto(
                esq - 6, sy(valor), _formatar_valor(valor, percentual), ft.Alignment.CENTER_RIGHT
            )
        )

    vaga = area_l / len(grupos)
    largura_barra = min(vaga * 0.75 / len(series), 40)
    for i, grupo in enumerate(grupos):
        cx = esq + vaga * (i + 0.5)
        inicio = cx - largura_barra * len(series) / 2
        for j, valor in enumerate(grupo["valores"]):
            topo, altura_barra = _barra_vertical(zero, sy(valor))
            x = inicio + j * largura_barra
            paint = ft.Paint(color=cor_serie(j), style=ft.PaintingStyle.FILL)
            formas.append(cv.Rect(x + 1, topo, largura_barra - 2, altura_barra, paint=paint))
            if largura_barra >= 26:  # rótulo só quando cabe sobre a barra
                texto = _formatar_valor(valor, percentual)
                formas.append(
                    _rotulo_barra(x + largura_barra / 2, topo, altura_barra, valor, texto)
                )
        formas.append(_texto(cx, base + 6, grupo["rotulo"], ft.Alignment.TOP_CENTER))

    legenda = ft.Row(
        [
            ft.Row(
                [
                    ft.Container(width=10, height=10, border_radius=2, bgcolor=cor_serie(j)),
                    ft.Text(serie, size=12, color=tema.TEXTO),
                ],
                spacing=6,
                tight=True,
            )
            for j, serie in enumerate(series)
        ],
        spacing=16,
        wrap=True,
    )
    return cv.Canvas(width=largura, height=altura, shapes=formas), legenda


def _escala(minimo: float, maximo: float) -> tuple[float, float]:
    """Limites com 5% de folga (e um intervalo mínimo quando todos os valores são iguais)."""
    if minimo == maximo:
        return minimo - 0.5, maximo + 0.5
    folga = (maximo - minimo) * 0.05
    return minimo - folga, maximo + folga


def dispersao(dados: dict, largura: float, altura: float) -> tuple[cv.Canvas, ft.Control | None]:
    """Pontos (x, y) e segmentos de referência; rótulos dos eixos nas pontas."""
    esq, base_extra = 70, 14
    area_l = largura - esq - _MARGEM_DIR
    area_a = altura - _MARGEM_TOPO - _MARGEM_BASE - base_extra
    base = _MARGEM_TOPO + area_a
    x0, x1 = _escala(dados["x_min"], dados["x_max"])
    y0, y1 = _escala(dados["y_min"], dados["y_max"])

    def sx(valor: float) -> float:
        return esq + (valor - x0) / (x1 - x0) * area_l

    def sy(valor: float) -> float:
        return base - (valor - y0) / (y1 - y0) * area_a

    eixo = ft.Paint(color=tema.BORDA, stroke_width=1, style=ft.PaintingStyle.STROKE)
    ponto = ft.Paint(color=tema.GRAFICO_BARRA_BORDA, style=ft.PaintingStyle.FILL)
    formas: list[cv.Shape] = [
        cv.Line(esq, base, esq + area_l, base, paint=eixo),
        cv.Line(esq, _MARGEM_TOPO, esq, base, paint=eixo),
    ]
    for valor in (y0, (y0 + y1) / 2, y1):
        formas.append(
            _texto(esq - 6, sy(valor), formatar_numero(valor, 2), ft.Alignment.CENTER_RIGHT)
        )
    for valor, alinhamento in (
        (x0, ft.Alignment.TOP_LEFT),
        ((x0 + x1) / 2, ft.Alignment.TOP_CENTER),
        (x1, ft.Alignment.TOP_RIGHT),
    ):
        formas.append(_texto(sx(valor), base + 6, formatar_numero(valor, 2), alinhamento))
    formas.append(_texto(esq + area_l / 2, base + 20, dados["rotulo_x"], ft.Alignment.TOP_CENTER))
    pontos = list(zip(dados["x"], dados["y"], strict=True))
    if dados.get("conectar"):  # ex.: curva ROC
        traco = ft.Paint(
            color=tema.GRAFICO_BARRA_BORDA, stroke_width=2, style=ft.PaintingStyle.STROKE
        )
        formas += [
            cv.Line(sx(x1), sy(y1), sx(x2), sy(y2), paint=traco)
            for (x1, y1), (x2, y2) in itertools.pairwise(pontos)
        ]
    else:
        formas += [cv.Circle(sx(x), sy(y), 2.5, paint=ponto) for x, y in pontos]
    for linha in dados.get("linhas", []):
        formas.append(
            cv.Line(
                sx(linha["x1"]),
                sy(linha["y1"]),
                sx(linha["x2"]),
                sy(linha["y2"]),
                paint=_paint_referencia(linha["estilo"]),
            )
        )
    itens = [(linha["rotulo"], linha["estilo"]) for linha in dados.get("linhas", [])]
    if dados["n_total"] > len(dados["x"]):
        itens.append((f"{len(dados['x'])} de {dados['n_total']} pontos", "ponto"))
    canvas = cv.Canvas(width=largura, height=altura, shapes=formas)
    return canvas, _legenda(itens) if itens else None


def cascata(dados: dict, largura: float, altura: float) -> tuple[cv.Canvas, ft.Control]:
    """Barra inicial (intercepto), uma barra flutuante por contribuição e a barra do total."""
    inicio, etapas, final = dados["inicio"], dados["etapas"], dados["final"]
    niveis = [0.0, inicio["valor"], final["valor"], *(e["de"] for e in etapas)]
    niveis += [e["ate"] for e in etapas]
    y0, y1 = _escala(min(niveis), max(niveis))
    esq = 70
    area_l = largura - esq - _MARGEM_DIR
    area_a = altura - _MARGEM_TOPO - _MARGEM_BASE
    base = _MARGEM_TOPO + area_a

    def sy(valor: float) -> float:
        return base - (valor - y0) / (y1 - y0) * area_a

    eixo = ft.Paint(color=tema.BORDA, stroke_width=1, style=ft.PaintingStyle.STROKE)
    formas: list[cv.Shape] = [
        cv.Line(esq, _MARGEM_TOPO, esq, base, paint=eixo),
        cv.Line(esq, sy(0.0), esq + area_l, sy(0.0), paint=eixo),
    ]
    for valor in (y0, (y0 + y1) / 2, y1):
        formas.append(
            _texto(esq - 6, sy(valor), formatar_numero(valor, 2), ft.Alignment.CENTER_RIGHT)
        )
    barras_ = [(inicio["rotulo"], 0.0, inicio["valor"], tema.GRAFICO_BARRA_BORDA)]
    for etapa in etapas:
        cor = tema.GRAFICO_SERIES[1] if etapa["valor"] < 0 else tema.GRAFICO_SERIES[2]
        barras_.append((etapa["rotulo"], etapa["de"], etapa["ate"], cor))
    barras_.append((final["rotulo"], 0.0, final["valor"], tema.GRAFICO_DESTAQUE))
    vaga = area_l / len(barras_)
    meia = min(vaga * 0.32, 40)
    for i, (rotulo, de, ate, cor) in enumerate(barras_):
        cx = esq + vaga * (i + 0.5)
        topo, fundo = sy(max(de, ate)), sy(min(de, ate))
        paint = ft.Paint(color=cor, style=ft.PaintingStyle.FILL)
        formas.append(cv.Rect(cx - meia, topo, 2 * meia, max(fundo - topo, 1), paint=paint))
        valor = ate - de if 0 < i < len(barras_) - 1 else ate
        texto = ("+" if valor > 0 and 0 < i < len(barras_) - 1 else "") + formatar_numero(valor, 2)
        formas.append(_texto(cx, topo - 3, texto, ft.Alignment.BOTTOM_CENTER))
        formas.append(_texto(cx, base + 6, rotulo, ft.Alignment.TOP_CENTER))
    legenda = ft.Row(
        [
            ft.Row(
                [
                    ft.Container(width=10, height=10, border_radius=2, bgcolor=cor),
                    ft.Text(texto, size=12, color=tema.TEXTO),
                ],
                spacing=6,
                tight=True,
            )
            for cor, texto in (
                (tema.GRAFICO_SERIES[2], "Aumenta"),
                (tema.GRAFICO_SERIES[1], "Diminui"),
            )
        ],
        spacing=16,
    )
    return cv.Canvas(width=largura, height=altura, shapes=formas), legenda
