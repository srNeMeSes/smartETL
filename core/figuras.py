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
