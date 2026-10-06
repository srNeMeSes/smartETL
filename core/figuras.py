"""Especificações de gráficos (`Figura`): os dados são calculados aqui; a UI só desenha."""

from collections.abc import Sequence

import numpy as np

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
) -> Figura:
    """Boxplots lado a lado, um por grupo: (rótulo, valores)."""
    return Figura(
        tipo="boxplot",
        titulo=titulo,
        dados={
            "rotulo_y": rotulo_y,
            "grupos": [{"rotulo": rotulo, **resumo_boxplot(valores)} for rotulo, valores in grupos],
        },
    )
