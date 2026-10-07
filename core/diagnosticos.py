"""Diagnósticos de regressão linear sem implementação equivalente no statsmodels (sem Flet).

- Goldfeld-Quandt e Harrison-McCabe reproduzem `lmtest::gqtest` e `lmtest::hmctest` do R
  (divisão em `point = 0,5`, sem omitir observações centrais).
- GVIF (Fox & Monette, 1992) reproduz `car::vif`.
- p-valor do Lilliefors reproduz `nortest::lillie.test` (Dallal-Wilkinson + Stephens).
- Faixas de classificação do Durbin-Watson e do VIF: constantes internas, nunca exibidas inteiras
  na interface (só a interpretação da faixa em que o valor caiu).
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Faixas de referência (uso interno — não exibir na interface)
# ---------------------------------------------------------------------------
DW_FORTE_POSITIVA = "Forte autocorrelação positiva"
DW_POSITIVA = "Autocorrelação positiva"
DW_SEM = "Sem autocorrelação relevante"
DW_NEGATIVA = "Autocorrelação negativa"
DW_FORTE_NEGATIVA = "Forte autocorrelação negativa"

VIF_SEM = "Sem colinearidade relevante"
VIF_MODERADA = "Baixa/moderada"
VIF_PROBLEMATICA = "Problemática"
VIF_GRAVE = "Grave"
VIF_MUITO_GRAVE = "Muito grave"

# VIF: intervalos fechados à esquerda [limite, próximo limite).
FAIXAS_VIF: tuple[tuple[float, str], ...] = (
    (2.0, VIF_SEM),
    (5.0, VIF_MODERADA),
    (10.0, VIF_PROBLEMATICA),
    (20.0, VIF_GRAVE),
)


def classificar_dw(dw: float) -> str:
    """Faixa do Durbin-Watson: [0; 1) [1; 1,5) [1,5; 2,5] (2,5; 3] (3; 4]."""
    if dw < 1.0:
        return DW_FORTE_POSITIVA
    if dw < 1.5:
        return DW_POSITIVA
    if dw <= 2.5:
        return DW_SEM
    if dw <= 3.0:
        return DW_NEGATIVA
    return DW_FORTE_NEGATIVA


def classificar_vif(vif: float) -> str:
    """Faixa do VIF: [1; 2) [2; 5) [5; 10) [10; 20) [20; ∞)."""
    for limite, rotulo in FAIXAS_VIF:
        if vif < limite:
            return rotulo
    return VIF_MUITO_GRAVE


# ---------------------------------------------------------------------------
# Goldfeld-Quandt e Harrison-McCabe (lmtest)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResultadoGQ:
    estatistica: float
    gl1: int
    gl2: int
    p_valor: float
    degenerado: bool  # alguma metade com matriz de posto incompleto (dummy constante etc.)


def _rss(y: np.ndarray, x: np.ndarray) -> tuple[float, bool]:
    coef, _, posto, _ = np.linalg.lstsq(x, y, rcond=None)
    return float(((y - x @ coef) ** 2).sum()), posto < x.shape[1]


def ordem_estavel(z: np.ndarray) -> np.ndarray:
    """Índices que ordenam `z` com empates na ordem original (como `order()` do R)."""
    return np.argsort(np.asarray(z, dtype=float), kind="stable")


def goldfeld_quandt(y: np.ndarray, x: np.ndarray, ordem: np.ndarray) -> ResultadoGQ | None:
    """Goldfeld-Quandt como `gqtest(point = 0.5, fraction = 0, alternative = "greater")`.

    Dados ordenados por `ordem`; metade 1 = primeiras ⌊n/2⌋ linhas, metade 2 = a partir da linha
    ⌈n/2 + 0,01⌉ (base 1); GQ = [RSS₂/(n − point₂ + 1 − k)] / [RSS₁/(point₁ − k)] com k = número
    de colunas de X (mesmo se uma metade tiver posto incompleto, como no R); p = P(F > GQ).
    Devolve None se não houver graus de liberdade suficientes em alguma metade.
    """
    n, k = x.shape
    ponto1 = math.floor(0.5 * n)
    ponto2 = math.ceil(0.5 * n + 0.01)
    if ponto2 > n - k + 1 or ponto1 <= k or n - ponto2 + 1 - k < 1:
        return None
    ys, xs = y[ordem], x[ordem]
    rss1, deg1 = _rss(ys[:ponto1], xs[:ponto1])
    rss2, deg2 = _rss(ys[ponto2 - 1 :], xs[ponto2 - 1 :])
    gl1, gl2 = n - ponto2 + 1 - k, ponto1 - k
    if rss1 <= 0:
        return None
    gq = (rss2 / gl1) / (rss1 / gl2)
    return ResultadoGQ(gq, gl1, gl2, float(stats.f.sf(gq, gl1, gl2)), deg1 or deg2)


@dataclass(frozen=True)
class ResultadoHMC:
    estatistica: float
    p_valor: float
    simulacoes: int


def harrison_mccabe(
    residuos: np.ndarray, ordem: np.ndarray, k: int, simulacoes: int = 1000, semente: int = 1
) -> ResultadoHMC | None:
    """Harrison-McCabe como `hmctest(point = 0.5)`.

    HMC = Σ e²ᵢ (i ≤ ⌊n/2⌋) / Σ e²ᵢ, com os resíduos do modelo na ordem dada (os resíduos de MQO
    não dependem da ordem das linhas). p-valor por simulação, como no R: para cada réplica, n
    normais padronizadas (média 0, desvio 1 amostral) e a mesma razão; p = proporção das réplicas
    com estatística ≤ HMC (valores pequenos indicam variância crescente). `semente` fixa o
    resultado entre execuções. Devolve None se ⌊n/2⌋ < k ou > n − k.
    """
    n = len(residuos)
    ponto = math.floor(0.5 * n)
    if ponto > n - k or ponto < k:
        return None
    e2 = np.asarray(residuos, dtype=float)[ordem] ** 2
    hmc = float(e2[:ponto].sum() / e2.sum())
    rng = np.random.default_rng(semente)
    lote = max(1, 2_000_000 // n)
    abaixo, feitas = 0, 0
    while feitas < simulacoes:
        m = min(lote, simulacoes - feitas)
        z = rng.standard_normal((m, n))
        z = (z - z.mean(axis=1, keepdims=True)) / z.std(axis=1, ddof=1, keepdims=True)
        z2 = z**2
        razao = z2[:, :ponto].sum(axis=1) / z2.sum(axis=1)
        abaixo += int((razao <= hmc).sum())
        feitas += m
    return ResultadoHMC(hmc, abaixo / simulacoes, simulacoes)


# ---------------------------------------------------------------------------
# GVIF (car::vif)
# ---------------------------------------------------------------------------
def gvif(vcov: np.ndarray, grupos: list[list[int]]) -> list[float]:
    """GVIF de cada variável original (grupo de colunas), como `car::vif`.

    R = matriz de correlação de `vcov` (covariância dos coeficientes, SEM o intercepto);
    GVIF = det(R₁₁)·det(R₂₂)/det(R), com R₁₁ = bloco das colunas da variável e R₂₂ = o resto.
    Para uma coluna, GVIF = VIF. Calculado com logaritmos dos determinantes (estável).
    """
    desvio = np.sqrt(np.diag(vcov))
    r = vcov / np.outer(desvio, desvio)
    _, log_total = np.linalg.slogdet(r)
    todos = np.arange(r.shape[0])
    valores = []
    for colunas in grupos:
        resto = np.setdiff1d(todos, colunas)
        _, log1 = np.linalg.slogdet(r[np.ix_(colunas, colunas)])
        log2 = np.linalg.slogdet(r[np.ix_(resto, resto)])[1] if len(resto) else 0.0
        valores.append(float(math.exp(log1 + log2 - log_total)))
    return valores


# ---------------------------------------------------------------------------
# Normalidade dos resíduos
# ---------------------------------------------------------------------------
LIMITE_SHAPIRO = 5000


def lilliefors(x: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov com correção de Lilliefors como `nortest::lillie.test`: (D, p).

    D com média e desvio amostral estimados; p pela aproximação de Dallal-Wilkinson (1986) e,
    quando ela passa de 0,1, pela fórmula de Stephens (1974) — as mesmas constantes do nortest.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    p = stats.norm.cdf((x - x.mean()) / x.std(ddof=1))
    i = np.arange(1, n + 1)
    d = float(max((i / n - p).max(), (p - (i - 1) / n).max()))
    kd, nd = (d, n) if n <= 100 else (d * (n / 100) ** 0.49, 100)
    pvalor = math.exp(
        -7.01256 * kd**2 * (nd + 2.78019)
        + 2.99587 * kd * math.sqrt(nd + 2.78019)
        - 0.122119
        + 0.974598 / math.sqrt(nd)
        + 1.67997 / nd
    )
    if pvalor > 0.1:
        kk = (math.sqrt(n) - 0.01 + 0.85 / math.sqrt(n)) * d
        if kk <= 0.302:
            pvalor = 1.0
        elif kk <= 0.5:
            pvalor = 2.76773 - 19.828315 * kk + 80.709644 * kk**2 - 138.55152 * kk**3
            pvalor += 81.218052 * kk**4
        elif kk <= 0.9:
            pvalor = -4.901232 + 40.662806 * kk - 97.490286 * kk**2 + 94.029866 * kk**3
            pvalor -= 32.355711 * kk**4
        elif kk <= 1.31:
            pvalor = 6.198765 - 19.558097 * kk + 23.186922 * kk**2 - 12.234627 * kk**3
            pvalor += 2.423045 * kk**4
        else:
            pvalor = 0.0
    return d, float(pvalor)


def normalidade(residuos: np.ndarray) -> tuple[str, float, float]:
    """(nome do teste, estatística, p): Shapiro-Wilk até 5000 resíduos; acima, Lilliefors."""
    if len(residuos) <= LIMITE_SHAPIRO:
        resultado = stats.shapiro(residuos)
        return "Shapiro-Wilk", float(resultado.statistic), float(resultado.pvalue)
    d, p = lilliefors(residuos)
    return "Kolmogorov-Smirnov (Lilliefors)", d, p
