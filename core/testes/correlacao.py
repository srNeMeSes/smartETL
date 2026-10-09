"""Correlação de Pearson e de Spearman (grupo Relação)."""

import math

import numpy as np
import pandas as pd
from scipy import stats

from core.base import (
    ComparacaoPValores,
    ErroValidacao,
    Figura,
    GrupoFiguras,
    ParametroSpec,
    ResultadoTeste,
    TesteBase,
)
from core.figuras import dispersao
from core.interpretacao import decidir, formatar_numero, formatar_p_valor, interpretar
from core.validacao import erro_alfa, erro_opcao, erros_coluna

ALTERNATIVAS_COR = {"ρ ≠ 0": "two-sided", "ρ > 0": "greater", "ρ < 0": "less"}
ALTERNATIVA_PADRAO_COR = "ρ ≠ 0"
_TEXTO_H1 = {
    "two-sided": "a correlação é diferente de zero",
    "greater": "a correlação é positiva",
    "less": "a correlação é negativa",
}
# Força pelas faixas de Cohen (1988) sobre |r|: [0; 0,1) [0,1; 0,3) [0,3; 0,5) [0,5; 1].
FAIXAS_FORCA = ((0.1, "desprezível"), (0.3, "fraca"), (0.5, "moderada"))
DIVERGENCIA_PEARSON_SPEARMAN = 0.2
MIN_OBSERVACOES = 3
VALORES, POSTOS = "Valores", "Postos"
AVISO_IC_N3 = "Com n = 3 o intervalo de confiança não é calculado."


def classificar_forca(r: float) -> str:
    """ "desprezível", "fraca", "moderada" ou "forte" (faixas de Cohen sobre |r|)."""
    for limite, rotulo in FAIXAS_FORCA:
        if abs(r) < limite:
            return rotulo
    return "forte"


def descrever(r: float) -> str:
    """Ex.: "correlação positiva moderada"; "correlação desprezível" (sem sentido)."""
    forca = classificar_forca(r)
    if forca == "desprezível" or r == 0:
        return "correlação desprezível"
    return f"correlação {'positiva' if r > 0 else 'negativa'} {forca}"


def ic_fisher(r: float, n: int, confianca: float, alternativa: str, var_z: float) -> tuple:
    """IC de ρ pela z de Fisher: tanh(atanh(r) ± z·√var_z); unilateral quando H₁ é unilateral."""
    if abs(r) >= 1:
        return r, r
    z, ep = math.atanh(r), math.sqrt(var_z)
    if alternativa == "two-sided":
        critico = float(stats.norm.ppf(1 - (1 - confianca) / 2))
        return math.tanh(z - critico * ep), math.tanh(z + critico * ep)
    critico = float(stats.norm.ppf(confianca))
    if alternativa == "greater":
        return math.tanh(z - critico * ep), 1.0
    return -1.0, math.tanh(z + critico * ep)


class _Correlacao(TesteBase):
    """Base comum: duas colunas numéricas, linhas completas, H₀: ρ = 0."""

    simbolo = "r"
    parametro = "ρ"  # parâmetro populacional citado no H₀ (Kendall: τ)
    metodo = ""

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec("x", "Variável X", "coluna_numerica"),
            ParametroSpec("y", "Variável Y", "coluna_numerica"),
            ParametroSpec(
                "alternativa",
                "Hipótese alternativa (H₁)",
                "opcao",
                padrao=ALTERNATIVA_PADRAO_COR,
                opcoes=list(ALTERNATIVAS_COR),
            ),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    @staticmethod
    def pares(df: pd.DataFrame, x: str, y: str) -> tuple[np.ndarray, np.ndarray, int]:
        """(x, y) das linhas completas e quantas linhas foram descartadas."""
        validas = df[[x, y]].dropna()
        return (
            validas[x].to_numpy(dtype=float),
            validas[y].to_numpy(dtype=float),
            len(df) - len(validas),
        )

    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        x, y = params.get("x"), params.get("y")
        erros = erros_coluna(df, x, "a variável X") + erros_coluna(df, y, "a variável Y")
        if not erros and x == y:
            erros.append("As variáveis X e Y devem ser colunas diferentes.")
        erros += erro_opcao(
            params.get("alternativa", ALTERNATIVA_PADRAO_COR),
            ALTERNATIVAS_COR,
            "Escolha uma hipótese alternativa válida.",
        )
        erros += erro_alfa(params.get("alfa", 0.05))
        if erros:
            return erros
        xs, ys, _ = self.pares(df, x, y)
        if len(xs) < MIN_OBSERVACOES:
            return [
                f"São necessárias ao menos {MIN_OBSERVACOES} linhas com X e Y preenchidos "
                f"(há {len(xs)})."
            ]
        constantes = [nome for nome, v in ((x, xs), (y, ys)) if np.ptp(v) == 0]
        if constantes:
            return [
                f"A coluna '{c}' não varia nas linhas completas: a correlação não é definida."
                for c in constantes
            ]
        return []

    # ---------------- Comum ----------------
    def _interpretacao(self, coef: float, p: float, alfa: float, alternativa: str, x, y) -> str:
        h0 = f"{self.parametro} = 0: não há correlação entre '{x}' e '{y}'"
        resumo = f"{self.simbolo} = {formatar_numero(coef, 3)}: {descrever(coef)}"
        if alternativa == "two-sided":
            rejeita = f"Há evidência estatística de correlação entre '{x}' e '{y}' ({resumo})."
            nao = f"Não há evidência suficiente de correlação entre '{x}' e '{y}' ({resumo})."
        else:
            sentido = "positiva" if alternativa == "greater" else "negativa"
            rejeita = (
                f"Há evidência estatística de correlação {sentido} entre '{x}' e '{y}' ({resumo})."
            )
            nao = (
                f"Não há evidência suficiente de correlação {sentido} entre '{x}' e '{y}' "
                f"({resumo})."
            )
        return interpretar(
            p,
            alfa,
            h0=h0,
            h1=_TEXTO_H1[alternativa],
            conclusao_rejeita=rejeita,
            conclusao_nao_rejeita=nao,
        )

    @staticmethod
    def _rotulo_ic(alfa: float, alternativa: str) -> str:
        tipo = "" if alternativa == "two-sided" else " (unilateral)"
        return f"IC {(1 - alfa) * 100:.0f}% para ρ{tipo}"

    @staticmethod
    def _formatar_ic(inferior: float, superior: float) -> str:
        return f"[{formatar_numero(inferior, 3)}; {formatar_numero(superior, 3)}]"


class TesteCorrelacaoPearson(_Correlacao):
    """Correlação linear de Pearson entre duas colunas numéricas (H₀: ρ = 0).

    Linhas com X ou Y ausente são descartadas (aviso); exige n ≥ 3 e variação nas duas colunas.
    `scipy.stats.pearsonr`: r, t = r·√(n − 2)/√(1 − r²) com n − 2 gl e p na alternativa
    escolhida; IC pela z de Fisher (var = 1/(n − 3); unilateral quando H₁ é unilateral) — os
    mesmos valores do `cor.test` do R. Força pelas faixas de Cohen sobre |r|. Card Pearson ×
    Spearman nas três alternativas; aviso quando |r − ρₛ| > 0,2 (relação não linear ou
    outliers). Gráfico: dispersão com a reta de mínimos quadrados. Decisão: p ≤ α.
    """

    id = "correlacao_pearson"
    nome = "Correlação de Pearson"
    grupo = "Relação"
    simbolo = "r"

    def comparacao_inicial(self) -> ComparacaoPValores:
        return ComparacaoPValores(
            titulo_esquerda="Pearson",
            titulo_direita="Spearman",
            hipoteses=list(ALTERNATIVAS_COR),
            linhas=[(None, None)] * len(ALTERNATIVAS_COR),
        )

    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)
        x, y = params["x"], params["y"]
        alfa = float(params.get("alfa", 0.05))
        alternativa = ALTERNATIVAS_COR[params.get("alternativa", ALTERNATIVA_PADRAO_COR)]
        xs, ys, descartadas = self.pares(df, x, y)
        n = len(xs)

        resultados = {a: stats.pearsonr(xs, ys, alternative=a) for a in ALTERNATIVAS_COR.values()}
        principal = resultados[alternativa]
        r, p_valor = float(principal.statistic), float(principal.pvalue)
        # Com n = 3 a variância da z de Fisher, 1/(n − 3), é infinita: o scipy devolve [−1; 1],
        # que não informa nada. Como na Spearman, o IC não é calculado (com aviso).
        if n > 3:
            ic = principal.confidence_interval(confidence_level=1 - alfa)
            ic_inf, ic_sup = float(ic.low), float(ic.high)
        else:
            ic_inf, ic_sup = math.nan, math.nan
        gl = n - 2
        t = r * math.sqrt(gl) / math.sqrt(1 - r**2) if abs(r) < 1 else math.copysign(math.inf, r)
        spearman = {a: stats.spearmanr(xs, ys, alternative=a) for a in ALTERNATIVAS_COR.values()}
        rho = float(spearman["two-sided"].statistic)

        estatisticas = {
            "n": float(n),
            "n_descartadas": float(descartadas),
            "r": r,
            "r2": r**2,
            "t": t,
            "gl": float(gl),
            "p_valor": p_valor,
            "ic_inferior": ic_inf,
            "ic_superior": ic_sup,
            "rho_spearman": rho,
        }
        avisos = []
        if descartadas:
            avisos.append(f"{descartadas} linha(s) com X ou Y ausente foram descartadas.")
        if abs(r - rho) > DIVERGENCIA_PEARSON_SPEARMAN:
            avisos.append(
                f"A correlação de Pearson (r = {formatar_numero(r, 3)}) e a de Spearman "
                f"(ρₛ = {formatar_numero(rho, 3)}) diferem bastante: a relação pode não ser "
                "linear ou haver valores extremos influentes. Veja o gráfico de dispersão."
            )
        if n <= 3:
            avisos.append(AVISO_IC_N3)
        resumo = [
            ("Observações (n)", f"{n}"),
            ("Coeficiente de Pearson (r)", formatar_numero(r, 4)),
            ("Coeficiente de determinação (r²)", formatar_numero(r**2, 4)),
            ("Força", descrever(r).capitalize()),
            ("Estatística t", formatar_numero(t, 4)),
            ("Graus de liberdade", f"{gl}"),
            ("p-valor", formatar_p_valor(p_valor)),
            (self._rotulo_ic(alfa, alternativa), self._formatar_ic(ic_inf, ic_sup)),
            ("ρ de Spearman (comparação)", formatar_numero(rho, 4)),
        ]
        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=self._interpretacao(r, p_valor, alfa, alternativa, x, y),
            tabelas={"Resumo": pd.DataFrame(resumo, columns=["Medida", "Valor"])},
            figuras=[self._figura(xs, ys, x, y)],
            avisos=avisos,
            comparacao=ComparacaoPValores(
                titulo_esquerda="Pearson",
                titulo_direita="Spearman",
                hipoteses=list(ALTERNATIVAS_COR),
                linhas=[
                    (float(resultados[a].pvalue), float(spearman[a].pvalue))
                    for a in ALTERNATIVAS_COR.values()
                ],
            ),
        )

    @staticmethod
    def _figura(xs: np.ndarray, ys: np.ndarray, x: str, y: str) -> Figura:
        inclinacao, intercepto = np.polyfit(xs, ys, 1)
        x0, x1 = float(xs.min()), float(xs.max())
        reta = (
            "Reta de mínimos quadrados",
            x0,
            float(intercepto + inclinacao * x0),
            x1,
            float(intercepto + inclinacao * x1),
            "destaque",
        )
        return dispersao(xs, ys, f"'{y}' × '{x}'", x, y, [reta])


class TesteCorrelacaoSpearman(_Correlacao):
    """Correlação de postos de Spearman entre duas colunas numéricas (H₀: ρₛ = 0).

    Postos médios nos empates; ρₛ = correlação de Pearson dos postos. p pela aproximação t
    (`scipy.stats.spearmanr`; = `cor.test(method = "spearman", exact = FALSE)` do R). IC pela
    z de Fisher com a variância de Bonett & Wright (2000): (1 + ρₛ²/2)/(n − 3). Força pelas
    faixas de Cohen. Aviso de empates. Sem card. Gráficos: dispersão dos valores e dos postos
    (lista "Escala"). Decisão: p ≤ α.
    """

    id = "correlacao_spearman"
    nome = "Correlação de Spearman"
    grupo = "Relação"
    simbolo = "ρₛ"

    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)
        x, y = params["x"], params["y"]
        alfa = float(params.get("alfa", 0.05))
        alternativa = ALTERNATIVAS_COR[params.get("alternativa", ALTERNATIVA_PADRAO_COR)]
        xs, ys, descartadas = self.pares(df, x, y)
        n = len(xs)

        resultado = stats.spearmanr(xs, ys, alternative=alternativa)
        rho, p_valor = float(resultado.statistic), float(resultado.pvalue)
        if n > 3:
            ic = ic_fisher(rho, n, 1 - alfa, alternativa, (1 + rho**2 / 2) / (n - 3))
        else:
            ic = (math.nan, math.nan)
        postos_x, postos_y = stats.rankdata(xs), stats.rankdata(ys)
        empates = len(np.unique(xs)) < n or len(np.unique(ys)) < n

        estatisticas = {
            "n": float(n),
            "n_descartadas": float(descartadas),
            "rho": rho,
            "p_valor": p_valor,
            "ic_inferior": ic[0],
            "ic_superior": ic[1],
        }
        avisos = []
        if descartadas:
            avisos.append(f"{descartadas} linha(s) com X ou Y ausente foram descartadas.")
        if empates:
            avisos.append("Há empates nos valores: postos médios; o p-valor usa a aproximação t.")
        if n <= 3:
            avisos.append(AVISO_IC_N3)
        resumo = [
            ("Observações (n)", f"{n}"),
            ("Coeficiente de Spearman (ρₛ)", formatar_numero(rho, 4)),
            ("Força", descrever(rho).capitalize()),
            ("p-valor (aproximação t)", formatar_p_valor(p_valor)),
            (self._rotulo_ic(alfa, alternativa).replace("ρ", "ρₛ"), self._formatar_ic(*ic)),
        ]
        figuras = GrupoFiguras(
            "Escala",
            {
                VALORES: dispersao(xs, ys, f"'{y}' × '{x}'", x, y),
                POSTOS: dispersao(
                    postos_x,
                    postos_y,
                    f"Postos de '{y}' × postos de '{x}'",
                    f"Posto de '{x}'",
                    f"Posto de '{y}'",
                ),
            },
            VALORES,
        )
        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=self._interpretacao(rho, p_valor, alfa, alternativa, x, y),
            tabelas={"Resumo": pd.DataFrame(resumo, columns=["Medida", "Valor"])},
            figuras=[figuras],
            avisos=avisos,
        )
