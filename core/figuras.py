"""Especificações de gráficos (`Figura`): os dados são calculados aqui; a UI só desenha."""

from collections.abc import Sequence

import numpy as np
from scipy import stats

from core.base import Figura

MAX_CLASSES = 30


def histograma(
    valores: Sequence[float] | np.ndarray,
    titulo: str,
    rotulo_x: str,
    referencias: Sequence[tuple[str, float, str]] = (),
) -> Figura:
    """Histograma com linhas verticais de referência.

    `referencias`: (rótulo, valor, estilo) com estilo "destaque" (linha cheia) ou "tracejado".
    As classes seguem `numpy.histogram_bin_edges(bins="auto")`, limitadas a `MAX_CLASSES`, e o
    eixo é estendido para que todas as referências fiquem visíveis.
    """
    x = np.asarray(valores, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("histograma sem valores")
    pontos = [x.min(), x.max(), *(valor for _, valor, _ in referencias)]
    inicio, fim = float(min(pontos)), float(max(pontos))
    if inicio == fim:
        inicio, fim = inicio - 0.5, fim + 0.5
    bordas = np.histogram_bin_edges(x, bins="auto", range=(inicio, fim))
    if len(bordas) - 1 > MAX_CLASSES:
        bordas = np.histogram_bin_edges(x, bins=MAX_CLASSES, range=(inicio, fim))
    contagens, bordas = np.histogram(x, bins=bordas)
    return Figura(
        tipo="histograma",
        titulo=titulo,
        dados={
            "bordas": [float(b) for b in bordas],
            "contagens": [int(c) for c in contagens],
            "rotulo_x": rotulo_x,
            "referencias": [
                {"rotulo": rotulo, "valor": float(valor), "estilo": estilo}
                for rotulo, valor, estilo in referencias
            ],
        },
    )


def resumo_boxplot(valores: Sequence[float] | np.ndarray) -> dict[str, float | list[float]]:
    """Cinco números de Tukey: quartis (interpolação linear do numpy), bigodes até o dado mais
    extremo dentro de 1,5·IQR e pontos além disso como outliers; inclui a média."""
    x = np.asarray(valores, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("boxplot sem valores")
    q1, mediana, q3 = (float(v) for v in np.percentile(x, [25, 50, 75]))
    iqr = q3 - q1
    dentro = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
    return {
        "n": int(x.size),
        "q1": q1,
        "mediana": mediana,
        "q3": q3,
        "bigode_inf": float(dentro.min()),
        "bigode_sup": float(dentro.max()),
        "media": float(x.mean()),
        "outliers": sorted(float(v) for v in x[(x < dentro.min()) | (x > dentro.max())]),
    }


def boxplot(
    grupos: Sequence[tuple[str, Sequence[float] | np.ndarray]],
    titulo: str,
    rotulo_y: str,
    referencias: Sequence[tuple[str, float, str]] = (),
) -> Figura:
    """Boxplots lado a lado, um por grupo: (rótulo, valores); linhas horizontais opcionais."""
    return Figura(
        tipo="boxplot",
        titulo=titulo,
        dados={
            "rotulo_y": rotulo_y,
            "grupos": [{"rotulo": rotulo, **resumo_boxplot(valores)} for rotulo, valores in grupos],
            "referencias": [
                {"rotulo": rotulo, "valor": float(valor), "estilo": estilo}
                for rotulo, valor, estilo in referencias
            ],
        },
    )


def barras(
    categorias: Sequence[tuple[str, float]],
    titulo: str,
    rotulo_y: str,
    maximo: float | None = None,
    referencias: Sequence[tuple[str, float, str]] = (),
    percentual: bool = False,
) -> Figura:
    """Barras verticais (rótulo, valor) com linhas horizontais de referência opcionais.

    `maximo` fixa o topo do eixo (ex.: 1 para proporções); `percentual` exibe os valores como %.
    """
    valores = [float(v) for _, v in categorias]
    topo = maximo if maximo is not None else max([*valores, *(v for _, v, _ in referencias)])
    return Figura(
        tipo="barras",
        titulo=titulo,
        dados={
            "rotulo_y": rotulo_y,
            "maximo": float(topo) if topo > 0 else 1.0,
            "percentual": percentual,
            "categorias": [
                {"rotulo": rotulo, "valor": valor}
                for (rotulo, _), valor in zip(categorias, valores, strict=True)
            ],
            "referencias": [
                {"rotulo": rotulo, "valor": float(valor), "estilo": estilo}
                for rotulo, valor, estilo in referencias
            ],
        },
    )


def barras_agrupadas(
    grupos: Sequence[tuple[str, Sequence[float]]],
    series: Sequence[str],
    titulo: str,
    rotulo_y: str,
    maximo: float | None = None,
    percentual: bool = False,
) -> Figura:
    """Barras agrupadas: para cada grupo (rótulo, [valor por série]), uma barra por série."""
    valores = [[float(v) for v in vals] for _, vals in grupos]
    if any(len(v) != len(series) for v in valores):
        raise ValueError("cada grupo precisa de um valor por série")
    topo = maximo if maximo is not None else max((v for vals in valores for v in vals), default=0)
    return Figura(
        tipo="barras_agrupadas",
        titulo=titulo,
        dados={
            "rotulo_y": rotulo_y,
            "maximo": float(topo) if topo > 0 else 1.0,
            "percentual": percentual,
            "series": [str(s) for s in series],
            "grupos": [
                {"rotulo": rotulo, "valores": vals}
                for (rotulo, _), vals in zip(grupos, valores, strict=True)
            ],
        },
    )


MAX_PONTOS = 2000


def dispersao(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    titulo: str,
    rotulo_x: str,
    rotulo_y: str,
    linhas: Sequence[tuple[str, float, float, float, float, str]] = (),
    max_pontos: int = MAX_PONTOS,
) -> Figura:
    """Pontos (x, y) e segmentos de referência (rótulo, x₁, y₁, x₂, y₂, estilo).

    Com mais de `max_pontos` pontos, desenha uma amostra regular (sempre a mesma) e informa o
    total em `n_total`; os limites dos eixos usam todos os pontos.
    """
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    validos = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[validos], ys[validos]
    if xs.size == 0:
        raise ValueError("dispersão sem pontos")
    limites_x = [xs.min(), xs.max(), *(v for seg in linhas for v in (seg[1], seg[3]))]
    limites_y = [ys.min(), ys.max(), *(v for seg in linhas for v in (seg[2], seg[4]))]
    indices = np.arange(xs.size)
    if xs.size > max_pontos:
        indices = np.unique(np.linspace(0, xs.size - 1, max_pontos).round().astype(int))
    return Figura(
        tipo="dispersao",
        titulo=titulo,
        dados={
            "rotulo_x": rotulo_x,
            "rotulo_y": rotulo_y,
            "x": [float(v) for v in xs[indices]],
            "y": [float(v) for v in ys[indices]],
            "n_total": int(xs.size),
            "x_min": float(min(limites_x)),
            "x_max": float(max(limites_x)),
            "y_min": float(min(limites_y)),
            "y_max": float(max(limites_y)),
            "linhas": [
                {
                    "rotulo": r,
                    "x1": float(a),
                    "y1": float(b),
                    "x2": float(c),
                    "y2": float(d),
                    "estilo": e,
                }
                for r, a, b, c, d, e in linhas
            ],
        },
    )


def quantis_normais(n: int) -> np.ndarray:
    """Quantis teóricos N(0, 1) nas posições de `ppoints` do R: (i − a)/(n + 1 − 2a), a = 3/8
    se n ≤ 10, senão 1/2 (os mesmos do `qqnorm`)."""
    a = 3 / 8 if n <= 10 else 0.5
    return stats.norm.ppf((np.arange(1, n + 1) - a) / (n + 1 - 2 * a))


def qq_normal(valores: Sequence[float] | np.ndarray, titulo: str, rotulo_y: str) -> Figura:
    """Gráfico Q-Q normal com a reta de referência pelos quartis (como o `qqline` do R)."""
    y = np.sort(np.asarray(valores, dtype=float))
    x = quantis_normais(y.size)
    q1, q3 = np.percentile(y, [25, 75])
    z1, z3 = stats.norm.ppf([0.25, 0.75])
    inclinacao = (q3 - q1) / (z3 - z1)
    intercepto = q1 - inclinacao * z1
    linha = (
        "Referência normal",
        float(x[0]),
        float(intercepto + inclinacao * x[0]),
        float(x[-1]),
        float(intercepto + inclinacao * x[-1]),
        "tracejado",
    )
    return dispersao(x, y, titulo, "Quantis teóricos (normal)", rotulo_y, [linha])


def cascata(
    inicio: tuple[str, float],
    etapas: Sequence[tuple[str, float]],
    rotulo_final: str,
    rotulo_y: str,
    titulo: str,
) -> Figura:
    """Cascata: barra inicial, uma barra por contribuição (+/−) e a barra do total."""
    acumulado = float(inicio[1])
    passos = []
    for rotulo, valor in etapas:
        passos.append(
            {
                "rotulo": rotulo,
                "valor": float(valor),
                "de": acumulado,
                "ate": acumulado + float(valor),
            }
        )
        acumulado += float(valor)
    return Figura(
        tipo="cascata",
        titulo=titulo,
        dados={
            "rotulo_y": rotulo_y,
            "inicio": {"rotulo": inicio[0], "valor": float(inicio[1])},
            "etapas": passos,
            "final": {"rotulo": rotulo_final, "valor": acumulado},
        },
    )
