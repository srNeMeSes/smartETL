"""Odds ratio condicional de uma tabela 2×2 (hipergeométrica não central), vetorizada.

Os mesmos valores de `scipy.stats.contingency.odds_ratio(kind="conditional")` e do
`fisher.test` do R, mas em tempo proporcional ao suporte da distribuição (o scipy avalia a
densidade termo a termo em Python e levava ~20 s com n = 100 000). Sem Flet.

Com a tabela [[a, b], [c, d]], X = célula a tem, dadas as margens, distribuição hipergeométrica
não central: P(X = x; ψ) ∝ C(a + b, x)·C(c + d, a + c − x)·ψ^x, x no suporte. A EMV condicional
de ψ resolve E[X; ψ] = a; o IC exato inverte as caudas: P(X ≥ a; ψ_inf) = cauda e
P(X ≤ a; ψ_sup) = cauda (cauda = α/2 no bilateral, α no unilateral). Tudo em log, com
`logsumexp`, para não estourar com tabelas grandes.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp

_LIMITE_LOG = 60.0  # busca de log(ψ) em [−60, 60] (ψ entre ~1e−26 e ~1e26)


def _log_comb(n: int, k: np.ndarray) -> np.ndarray:
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


@dataclass(frozen=True)
class _Suporte:
    x: np.ndarray
    log_pesos: np.ndarray

    def probabilidades(self, log_psi: float) -> np.ndarray:
        log_p = self.log_pesos + self.x * log_psi
        return np.exp(log_p - logsumexp(log_p))


def _suporte(a: int, b: int, c: int, d: int) -> _Suporte:
    linha1, coluna1, total = a + b, a + c, a + b + c + d
    x = np.arange(max(0, coluna1 - (c + d)), min(linha1, coluna1) + 1, dtype=float)
    return _Suporte(x, _log_comb(linha1, x) + _log_comb(total - linha1, coluna1 - x))


def _raiz(funcao, alvo: float) -> float:
    """log(ψ) com funcao(log ψ) = alvo (funcao monótona em log ψ)."""
    return brentq(lambda t: funcao(t) - alvo, -_LIMITE_LOG, _LIMITE_LOG, xtol=1e-13, rtol=1e-14)


def odds_ratio_condicional(a: int, b: int, c: int, d: int) -> float:
    """EMV condicional de ψ (0 ou +∞ quando a está no extremo do suporte)."""
    s = _suporte(a, b, c, d)
    if a <= s.x[0]:
        return 0.0
    if a >= s.x[-1]:
        return math.inf
    return math.exp(_raiz(lambda t: float(s.x @ s.probabilidades(t)), a))


def ic_odds_ratio_condicional(
    a: int, b: int, c: int, d: int, confianca: float, alternativa: str
) -> tuple[float, float]:
    """IC exato condicional de ψ (alternativa "two-sided", "greater" ou "less")."""
    s = _suporte(a, b, c, d)
    cauda = (1 - confianca) / 2 if alternativa == "two-sided" else 1 - confianca
    acima = s.x >= a
    abaixo = s.x <= a
    inferior, superior = 0.0, math.inf
    if alternativa in ("two-sided", "greater") and a > s.x[0]:
        inferior = math.exp(_raiz(lambda t: float(s.probabilidades(t)[acima].sum()), cauda))
    if alternativa in ("two-sided", "less") and a < s.x[-1]:
        superior = math.exp(_raiz(lambda t: float(s.probabilidades(t)[abaixo].sum()), cauda))
    return inferior, superior
