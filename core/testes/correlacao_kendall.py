"""Correlação de Kendall (τ-b) — grupo Relação."""

import math

import numpy as np
import pandas as pd
from scipy import stats

from core.base import ComparacaoPValores, ErroValidacao, ResultadoTeste
from core.figuras import dispersao
from core.interpretacao import decidir, formatar_numero, formatar_p_valor
from core.testes.correlacao import (
    ALTERNATIVA_PADRAO_COR,
    ALTERNATIVAS_COR,
    _Correlacao,
    descrever,
    ic_fisher,
)

LIMITE_EXATO = 50  # como o cor.test do R: exato com n < 50 e sem empates
VAR_FIELLER = 0.437  # var(atanh τ) ≈ 0,437/(n − 4) (Fieller, Hartley e Pearson, 1957)
AVISO_IC_N4 = "Com n ≤ 4 o intervalo de confiança de τ não é calculado."


def _pares_empatados(contagens: np.ndarray) -> float:
    return float((contagens * (contagens - 1) / 2).sum())


def contagens_kendall(x: np.ndarray, y: np.ndarray, tau_b: float) -> dict[str, float]:
    """Pares concordantes (C) e discordantes (D), S = C − D e z com a correção de empates do
    `cor.test` do R, sem enumerar os n(n − 1)/2 pares (O(n log n)).

    τ-b = S/√((n₀ − n₁)(n₀ − n₂)), n₀ = n(n − 1)/2, n₁/n₂ = pares empatados em x/y; C + D =
    n₀ − n₁ − n₂ + n₃ (n₃ = empatados nos dois). var(S) = [n(n−1)(2n+5) − Σt(t−1)(2t+5) −
    Σu(u−1)(2u+5)]/18 + Σt(t−1)(t−2)·Σu(u−1)(u−2)/(9n(n−1)(n−2)) + Σt(t−1)·Σu(u−1)/(2n(n−1)).
    """
    n = len(x)
    t = np.unique(x, return_counts=True)[1].astype(float)
    u = np.unique(y, return_counts=True)[1].astype(float)
    conjunto = np.unique(np.column_stack([x, y]), axis=0, return_counts=True)[1].astype(float)
    n0 = n * (n - 1) / 2
    n1, n2, n3 = _pares_empatados(t), _pares_empatados(u), _pares_empatados(conjunto)
    s = tau_b * math.sqrt((n0 - n1) * (n0 - n2))
    soma = n0 - n1 - n2 + n3
    var = (
        n * (n - 1) * (2 * n + 5)
        - (t * (t - 1) * (2 * t + 5)).sum()
        - (u * (u - 1) * (2 * u + 5)).sum()
    ) / 18
    if n > 2:
        var += (
            (t * (t - 1) * (t - 2)).sum()
            * (u * (u - 1) * (u - 2)).sum()
            / (9 * n * (n - 1) * (n - 2))
        )
    var += (t * (t - 1)).sum() * (u * (u - 1)).sum() / (2 * n * (n - 1))
    return {
        "concordantes": round((soma + s) / 2),
        "discordantes": round((soma - s) / 2),
        "s": s,
        "z": s / math.sqrt(var) if var > 0 else math.nan,
    }


class TesteCorrelacaoKendall(_Correlacao):
    """Correlação de postos de Kendall, τ-b (H₀: τ = 0).

    Linhas com X ou Y ausente são descartadas (aviso); n ≥ 3 e variação nas duas colunas.
    `scipy.stats.kendalltau` (τ-b, corrigido para empates): p exato quando n < 50 e não há
    empates, senão aproximação normal com a correção de empates e sem correção de continuidade
    — os valores do `cor.test(method = "kendall")` do R (o Resumo informa o método). IC pela z de
    Fisher com var = 0,437/(n − 4) (Fieller, Hartley e Pearson), unilateral quando H₁ é. Força
    pelas faixas de Cohen sobre |τ|. Card Kendall × Spearman. Aviso de empates. Gráfico:
    dispersão dos valores. Decisão: p ≤ α.
    """

    id = "correlacao_kendall"
    nome = "Correlação de Kendall"
    grupo = "Relação"
    simbolo = "τ"
    parametro = "τ"

    def comparacao_inicial(self) -> ComparacaoPValores:
        return ComparacaoPValores("Kendall", "Spearman", list(ALTERNATIVAS_COR), [(None, None)] * 3)

    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)
        x, y = params["x"], params["y"]
        alfa = float(params.get("alfa", 0.05))
        alternativa = ALTERNATIVAS_COR[params.get("alternativa", ALTERNATIVA_PADRAO_COR)]
        xs, ys, descartadas = self.pares(df, x, y)
        n = len(xs)

        empates = len(np.unique(xs)) < n or len(np.unique(ys)) < n
        exato = n < LIMITE_EXATO and not empates
        metodo = "exact" if exato else "asymptotic"
        kendall = {
            a: stats.kendalltau(xs, ys, method=metodo, alternative=a)
            for a in ALTERNATIVAS_COR.values()
        }
        tau = float(kendall["two-sided"].statistic)
        p_valor = float(kendall[alternativa].pvalue)
        contagens = contagens_kendall(xs, ys, tau)
        ic = (
            ic_fisher(tau, n, 1 - alfa, alternativa, VAR_FIELLER / (n - 4))
            if n > 4
            else (math.nan, math.nan)
        )
        spearman = {a: stats.spearmanr(xs, ys, alternative=a) for a in ALTERNATIVAS_COR.values()}

        estatisticas = {
            "n": float(n),
            "n_descartadas": float(descartadas),
            "tau": tau,
            "p_valor": p_valor,
            "usou_exato": float(exato),
            "concordantes": float(contagens["concordantes"]),
            "discordantes": float(contagens["discordantes"]),
            "s": contagens["s"],
            "z": contagens["z"],
            "ic_inferior": ic[0],
            "ic_superior": ic[1],
            "rho_spearman": float(spearman["two-sided"].statistic),
        }
        avisos = []
        if descartadas:
            avisos.append(f"{descartadas} linha(s) com X ou Y ausente foram descartadas.")
        if empates:
            avisos.append(
                "Há empates nos valores: τ-b (corrigido para empates) e p-valor pela aproximação "
                "normal."
            )
        if n <= 4:
            avisos.append(AVISO_IC_N4)

        metodo_txt = (
            f"Exato (n = {n} < {LIMITE_EXATO}, sem empates)"
            if exato
            else "Aproximação normal (correção de empates, sem continuidade)"
        )
        resumo = [
            ("Observações (n)", f"{n}"),
            ("τ de Kendall (τ-b)", formatar_numero(tau, 4)),
            ("Força", descrever(tau).capitalize()),
            ("Pares concordantes (C)", f"{contagens['concordantes']}"),
            ("Pares discordantes (D)", f"{contagens['discordantes']}"),
            ("Estatística z = S/√var(S)", formatar_numero(contagens["z"], 4)),
            ("Método do p-valor", metodo_txt),
            ("p-valor", formatar_p_valor(p_valor)),
            (self._rotulo_ic(alfa, alternativa).replace("ρ", "τ"), self._formatar_ic(*ic)),
            ("ρ de Spearman (comparação)", formatar_numero(estatisticas["rho_spearman"], 4)),
        ]
        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=self._interpretacao(tau, p_valor, alfa, alternativa, x, y),
            tabelas={"Resumo": pd.DataFrame(resumo, columns=["Medida", "Valor"])},
            figuras=[dispersao(xs, ys, f"'{y}' × '{x}'", x, y)],
            avisos=avisos,
            comparacao=ComparacaoPValores(
                "Kendall",
                "Spearman",
                list(ALTERNATIVAS_COR),
                [
                    (float(kendall[a].pvalue), float(spearman[a].pvalue))
                    for a in ALTERNATIVAS_COR.values()
                ],
            ),
        )
