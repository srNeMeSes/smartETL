"""Correlação parcial (grupo Relação): X e Y controlando por uma ou mais variáveis Z."""

import math

import numpy as np
import pandas as pd
from scipy import stats

from core.base import ComparacaoPValores, ErroValidacao, ParametroSpec, ResultadoTeste, TesteBase
from core.figuras import dispersao
from core.interpretacao import decidir, formatar_numero, formatar_p_valor, interpretar
from core.testes.correlacao import (
    ALTERNATIVA_PADRAO_COR,
    ALTERNATIVAS_COR,
    DIVERGENCIA_PEARSON_SPEARMAN,
    descrever,
    ic_fisher,
)
from core.validacao import erro_alfa, erro_opcao, erros_coluna

PEARSON, SPEARMAN = "Pearson", "Spearman"
METODOS = (PEARSON, SPEARMAN)
_TEXTO_H1 = {
    "two-sided": "a correlação parcial é diferente de zero",
    "greater": "a correlação parcial é positiva",
    "less": "a correlação parcial é negativa",
}


def residuos(v: np.ndarray, controles: np.ndarray) -> np.ndarray:
    """Resíduos de v na regressão por mínimos quadrados sobre [1, Z]."""
    a = np.column_stack([np.ones(len(v)), controles])
    coef, *_ = np.linalg.lstsq(a, v, rcond=None)
    return v - a @ coef


def p_valores_t(t: float, gl: int) -> dict[str, float]:
    """p do t com `gl` graus de liberdade nas três alternativas."""
    return {
        "two-sided": float(2 * stats.t.sf(abs(t), gl)),
        "greater": float(stats.t.sf(t, gl)),
        "less": float(stats.t.cdf(t, gl)),
    }


class TesteCorrelacaoParcial(TesteBase):
    """Correlação parcial entre X e Y controlando por k variáveis Z (H₀: ρ parcial = 0).

    Linhas com X, Y ou algum controle ausente são descartadas (aviso). Pearson (padrão): r entre
    os resíduos de X e de Y nas regressões sobre [1, Z]; Spearman: o mesmo sobre os postos
    (postos médios nos empates) — os valores do `ppcor::pcor.test` do R. t = r·√gl/√(1 − r²)
    com gl = n − 2 − k; IC pela z de Fisher com variância 1/(n − 3 − k) (unilateral quando H₁
    é). Card parcial × simples (a correlação de X e Y sem controles, pelo mesmo método), aviso
    quando as duas diferem mais que 0,2. Gráfico: dispersão dos resíduos. Decisão: p ≤ α.
    """

    id = "correlacao_parcial"
    nome = "Correlação parcial"
    grupo = "Relação"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec("x", "Variável X", "coluna_numerica"),
            ParametroSpec("y", "Variável Y", "coluna_numerica"),
            ParametroSpec(
                "controles",
                "Variáveis de controle (Z)",
                "multi_coluna",
                ajuda="A correlação entre X e Y é medida depois de retirar o efeito destas.",
            ),
            ParametroSpec("metodo", "Método", "opcao", padrao=PEARSON, opcoes=list(METODOS)),
            ParametroSpec(
                "alternativa",
                "Hipótese alternativa (H₁)",
                "opcao",
                padrao=ALTERNATIVA_PADRAO_COR,
                opcoes=list(ALTERNATIVAS_COR),
            ),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    def comparacao_inicial(self) -> ComparacaoPValores:
        return ComparacaoPValores("Parcial", "Simples", list(ALTERNATIVAS_COR), [(None, None)] * 3)

    @staticmethod
    def dados(df: pd.DataFrame, x: str, y: str, controles: list[str]) -> tuple[pd.DataFrame, int]:
        """(linhas completas de X, Y e controles, linhas descartadas)."""
        completas = df[[x, y, *controles]].dropna().astype(float)
        return completas, len(df) - len(completas)

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        x, y = params.get("x"), params.get("y")
        controles = list(params.get("controles") or [])
        erros = erros_coluna(df, x, "a variável X") + erros_coluna(df, y, "a variável Y")
        if not erros and x == y:
            erros.append("As variáveis X e Y devem ser colunas diferentes.")
        if not controles:
            erros.append("Selecione ao menos uma variável de controle.")
        for nome in controles:
            erros += erros_coluna(df, nome, "a variável de controle")
        if {x, y} & set(controles):
            erros.append("X e Y não podem ser também variáveis de controle.")
        if len(set(controles)) != len(controles):
            erros.append("As variáveis de controle devem ser colunas diferentes.")
        erros += erro_opcao(params.get("metodo", PEARSON), METODOS, "Escolha o método.")
        erros += erro_opcao(
            params.get("alternativa", ALTERNATIVA_PADRAO_COR),
            ALTERNATIVAS_COR,
            "Escolha uma hipótese alternativa válida.",
        )
        erros += erro_alfa(params.get("alfa", 0.05))
        if erros:
            return erros
        return self._erros_dados(df, x, y, controles)

    def _erros_dados(self, df: pd.DataFrame, x: str, y: str, controles: list[str]) -> list[str]:
        dados, _ = self.dados(df, x, y, controles)
        n, k = len(dados), len(controles)
        if n < k + 4:
            return [
                f"São necessárias ao menos {k + 4} linhas completas (X, Y e {k} controle(s)); "
                f"há {n}."
            ]
        z = dados[controles].to_numpy()
        if np.linalg.matrix_rank(np.column_stack([np.ones(n), z])) < k + 1:
            return [
                "As variáveis de controle são constantes ou combinação exata umas das outras: "
                "remova alguma."
            ]
        erros = []
        for nome in (x, y):
            v = dados[nome].to_numpy()
            if np.ptp(residuos(v, z)) <= 1e-10 * max(1.0, np.ptp(v)):
                erros.append(
                    f"A variável '{nome}' é explicada totalmente pelos controles: não sobra "
                    "variação para correlacionar."
                )
        return erros

    # ---------------- Execução ----------------
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)
        x, y = params["x"], params["y"]
        controles = list(params["controles"])
        metodo = params.get("metodo", PEARSON)
        alfa = float(params.get("alfa", 0.05))
        alternativa = ALTERNATIVAS_COR[params.get("alternativa", ALTERNATIVA_PADRAO_COR)]

        dados, descartadas = self.dados(df, x, y, controles)
        n, k = len(dados), len(controles)
        matriz = dados.to_numpy()
        if metodo == SPEARMAN:
            matriz = np.apply_along_axis(stats.rankdata, 0, matriz)
        vx, vy, z = matriz[:, 0], matriz[:, 1], matriz[:, 2:]
        ex, ey = residuos(vx, z), residuos(vy, z)
        r = float(np.clip(np.corrcoef(ex, ey)[0, 1], -1.0, 1.0))
        gl = n - 2 - k
        t = r * math.sqrt(gl) / math.sqrt(1 - r**2) if abs(r) < 1 else math.copysign(math.inf, r)
        p_parcial = p_valores_t(t, gl)
        p_valor = p_parcial[alternativa]
        ic = ic_fisher(r, n, 1 - alfa, alternativa, 1 / (n - 3 - k))
        funcao = stats.pearsonr if metodo == PEARSON else stats.spearmanr
        simples = {a: funcao(vx, vy, alternative=a) for a in ALTERNATIVAS_COR.values()}
        r_simples = float(simples["two-sided"].statistic)

        estatisticas = {
            "n": float(n),
            "k": float(k),
            "n_descartadas": float(descartadas),
            "r": r,
            "r_simples": r_simples,
            "t": t,
            "gl": float(gl),
            "p_valor": p_valor,
            "ic_inferior": ic[0],
            "ic_superior": ic[1],
        }
        lista = ", ".join(f"'{c}'" for c in controles)
        avisos = []
        if descartadas:
            avisos.append(
                f"{descartadas} linha(s) com X, Y ou algum controle ausente foram descartadas."
            )
        if abs(r - r_simples) > DIVERGENCIA_PEARSON_SPEARMAN:
            avisos.append(
                f"Controlar por {lista} muda bastante a correlação (simples "
                f"{formatar_numero(r_simples, 3)}, parcial {formatar_numero(r, 3)}): parte da "
                f"relação entre '{x}' e '{y}' passa pelos controles."
            )

        tipo_ic = "" if alternativa == "two-sided" else " (unilateral)"
        resumo = [
            ("Observações (n)", f"{n}"),
            ("Variáveis de controle", f"{lista} (k = {k})"),
            ("Método", metodo),
            ("Correlação parcial", formatar_numero(r, 4)),
            ("Correlação simples (sem controles)", formatar_numero(r_simples, 4)),
            ("Força (parcial)", descrever(r).capitalize()),
            ("Estatística t", formatar_numero(t, 4)),
            ("Graus de liberdade (n − 2 − k)", f"{gl}"),
            ("p-valor", formatar_p_valor(p_valor)),
            (
                f"IC {(1 - alfa) * 100:.0f}% para ρ parcial{tipo_ic}",
                f"[{formatar_numero(ic[0], 3)}; {formatar_numero(ic[1], 3)}]",
            ),
        ]
        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=self._interpretacao(r, p_valor, alfa, alternativa, x, y, lista),
            tabelas={"Resumo": pd.DataFrame(resumo, columns=["Medida", "Valor"])},
            figuras=[self._figura(ex, ey, x, y, lista, metodo)],
            avisos=avisos,
            comparacao=ComparacaoPValores(
                "Parcial",
                "Simples",
                list(ALTERNATIVAS_COR),
                [(p_parcial[a], float(simples[a].pvalue)) for a in ALTERNATIVAS_COR.values()],
            ),
        )

    @staticmethod
    def _interpretacao(r, p, alfa, alternativa, x, y, lista) -> str:
        par = f"'{x}' e '{y}', controlando por {lista}"
        resumo = f"r parcial = {formatar_numero(r, 3)}: {descrever(r)}"
        sentido = {"two-sided": "", "greater": " positiva", "less": " negativa"}[alternativa]
        return interpretar(
            p,
            alfa,
            h0=f"ρ parcial = 0: não há correlação entre {par}",
            h1=_TEXTO_H1[alternativa],
            conclusao_rejeita=(
                f"Há evidência estatística de correlação{sentido} entre {par} ({resumo})."
            ),
            conclusao_nao_rejeita=(
                f"Não há evidência suficiente de correlação{sentido} entre {par} ({resumo})."
            ),
        )

    @staticmethod
    def _figura(ex, ey, x, y, lista, metodo):
        inclinacao, intercepto = np.polyfit(ex, ey, 1)
        x0, x1 = float(ex.min()), float(ex.max())
        reta = (
            "Reta de mínimos quadrados",
            x0,
            float(intercepto + inclinacao * x0),
            x1,
            float(intercepto + inclinacao * x1),
            "destaque",
        )
        sufixo = " (postos)" if metodo == SPEARMAN else ""
        return dispersao(
            ex,
            ey,
            f"Resíduos de '{y}' × resíduos de '{x}', controlando por {lista}{sufixo}",
            f"Resíduo de '{x}'",
            f"Resíduo de '{y}'",
            [reta],
        )
