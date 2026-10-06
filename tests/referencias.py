"""Cálculos de referência independentes do código de produção (usados pelos testes)."""

import itertools
import math

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


# ---------------------------------------------------------------------------
# Tabela 2×2: hipergeométrica (central e não central) com a biblioteca padrão
# ---------------------------------------------------------------------------
def _suporte_2x2(a: int, b: int, c: int, d: int) -> tuple[range, list[int]]:
    n1, n2, m = a + b, c + d, a + c
    suporte = range(max(0, m - n2), min(n1, m) + 1)
    return suporte, [math.comb(n1, x) * math.comb(n2, m - x) for x in suporte]


def fisher_exato(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """p-valores exatos de Fisher (bilateral, maior, menor) pela hipergeométrica com math.comb.
    Bilateral: soma das probabilidades ≤ à da tabela observada (método do R)."""
    suporte, pesos = _suporte_2x2(a, b, c, d)
    total = sum(pesos)
    pmf = {x: w / total for x, w in zip(suporte, pesos, strict=True)}
    maior = sum(p for x, p in pmf.items() if x >= a)
    menor = sum(p for x, p in pmf.items() if x <= a)
    bilateral = min(1.0, sum(p for p in pmf.values() if p <= pmf[a] * (1 + 1e-7)))
    return bilateral, maior, menor


def _caudas_nao_central(a, b, c, d, psi):
    """(P(X ≥ a), P(X ≤ a), E[X]) da hipergeométrica não central com odds ratio psi."""
    suporte, pesos = _suporte_2x2(a, b, c, d)
    centro = a  # escala para evitar estouro: psi^(x − a)
    termos = [
        w * math.exp((x - centro) * math.log(psi)) for x, w in zip(suporte, pesos, strict=True)
    ]
    total = sum(termos)
    maior = sum(t for x, t in zip(suporte, termos, strict=True) if x >= a) / total
    menor = sum(t for x, t in zip(suporte, termos, strict=True) if x <= a) / total
    media = sum(x * t for x, t in zip(suporte, termos, strict=True)) / total
    return maior, menor, media


def _bissecao_log(funcao, alvo, crescente=True, iteracoes=200):
    """Encontra psi > 0 com funcao(psi) = alvo, buscando em log(psi) ∈ [-50, 50]."""
    baixo, alto = -50.0, 50.0
    for _ in range(iteracoes):
        meio = (baixo + alto) / 2
        valor = funcao(math.exp(meio))
        if (valor < alvo) == crescente:
            baixo = meio
        else:
            alto = meio
    return math.exp((baixo + alto) / 2)


def odds_ratio_condicional(a: int, b: int, c: int, d: int) -> float:
    """EMV condicional da odds ratio: psi com E[X | psi] = a (sem célula extrema)."""
    return _bissecao_log(lambda psi: _caudas_nao_central(a, b, c, d, psi)[2], a)


def ic_exato_odds_ratio(a, b, c, d, confianca: float, alternativa: str) -> tuple[float, float]:
    """IC exato condicional (inversão das caudas, como o fisher.test do R)."""
    alfa = 1 - confianca
    cauda = alfa / 2 if alternativa == "two-sided" else alfa
    baixo, alto = 0.0, math.inf
    if alternativa in ("two-sided", "greater"):
        # P(X ≥ a; psi) cresce com psi: limite inferior onde vale `cauda`.
        baixo = _bissecao_log(lambda psi: _caudas_nao_central(a, b, c, d, psi)[0], cauda)
    if alternativa in ("two-sided", "less"):
        # P(X ≤ a; psi) decresce com psi: limite superior onde vale `cauda`.
        alto = _bissecao_log(
            lambda psi: _caudas_nao_central(a, b, c, d, psi)[1], cauda, crescente=False
        )
    return baixo, alto
