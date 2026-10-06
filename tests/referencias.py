"""Cálculos de referência independentes do código de produção (usados pelos testes)."""

import itertools

import numpy as np
from scipy.stats import rankdata


def wilcoxon_exato(diferencas) -> tuple[float, float, float]:
    """p-valores exatos do Wilcoxon dos postos sinalizados por enumeração das 2^n atribuições
    de sinal aos postos: (bilateral, maior, menor). Requer diferenças sem empates nem zeros."""
    d = np.asarray(diferencas, dtype=float)
    postos = rankdata(np.abs(d))
    w_obs = postos[d > 0].sum()
    distribuicao = np.array(
        [
            sum(p for p, s in zip(postos, sinais, strict=True) if s)
            for sinais in itertools.product([0, 1], repeat=len(d))
        ]
    )
    maior = float(np.mean(distribuicao >= w_obs - 1e-9))
    menor = float(np.mean(distribuicao <= w_obs + 1e-9))
    return min(1.0, 2 * min(maior, menor)), maior, menor
