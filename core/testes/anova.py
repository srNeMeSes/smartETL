"""Análise de variância (ANOVA)."""

import itertools
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from core.base import ComparacaoPValores, ErroValidacao, ParametroSpec, ResultadoTeste, TesteBase
from core.figuras import boxplot
from core.interpretacao import (
    REJEITA_H0,
    decidir,
    formatar_numero,
    formatar_p_valor,
    interpretar,
)
from core.testes.medias import TesteT2Amostras
from core.tipos import rotulo_nivel
from core.validacao import erro_alfa, erro_opcao, erros_coluna

CLASSICA = "Clássica (variâncias iguais)"
WELCH = "Welch (variâncias diferentes)"
VARIANTES_ANOVA = (CLASSICA, WELCH)
MAX_GRUPOS_ANOVA = 20
ALFA_LEVENE = 0.05
HIPOTESE_CARD = "algum μᵢ ≠ μⱼ"
TABELA_TUKEY = "Comparações múltiplas (Tukey HSD)"


def somas_de_quadrados(amostras: list[np.ndarray]) -> tuple[float, float]:
    """(SQ entre grupos, SQ dentro dos grupos) da decomposição da ANOVA de 1 fator."""
    media_geral = np.concatenate(amostras).mean()
    sq_entre = float(sum(len(x) * (x.mean() - media_geral) ** 2 for x in amostras))
    sq_dentro = float(sum(((x - x.mean()) ** 2).sum() for x in amostras))
    return sq_entre, sq_dentro


class TesteAnova1Fator(TesteBase):
    """ANOVA de 1 fator para k ≥ 2 grupos independentes (H₀: μ₁ = μ₂ = … = μₖ).

    Entrada: coluna numérica + fator com 2 a 20 níveis e ao menos 2 observações por grupo;
    linhas incompletas descartadas (aviso). Variante clássica (padrão, `scipy.stats.f_oneway`):
    F = QM entre / QM dentro com (k − 1, N − k) gl. Variante Welch (`f_oneway(equal_var=False)`):
    F de Welch com gl do denominador fracionário, exige variância > 0 em cada grupo. A tabela
    ANOVA (SQ, gl, QM, F, p) e os efeitos η² = SQE/SQT e ω² = (SQE − (k − 1)·QMD)/(SQT + QMD)
    vêm sempre da decomposição clássica; na variante Welch o teste (F, gl e p) é o de Welch.
    Pressuposto de homogeneidade: teste de Levene centrado na mediana (Brown-Forsythe,
    `scipy.stats.levene`), só como aviso quando p < 0,05. Pós-teste de Tukey HSD
    (`scipy.stats.tukey_hsd`, Tukey-Kramer com n desiguais) opcional, desligado por padrão e
    exibido só quando H₀ é rejeitada. Card: ANOVA × Kruskal-Wallis, só a linha bilateral (os dois
    testes não têm alternativa unilateral). Decisão: p ≤ α.
    """

    id = "anova_1fator"
    nome = "ANOVA (1 fator)"
    grupo = "ANOVA"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec("coluna", "Variável", "coluna_numerica"),
            ParametroSpec("grupo", "Fator (2 ou mais níveis)", "coluna_categorica"),
            ParametroSpec(
                "variante", "Variante", "opcao", padrao=CLASSICA, opcoes=list(VARIANTES_ANOVA)
            ),
            ParametroSpec("tukey", "Comparações múltiplas (Tukey HSD)", "booleano", padrao=False),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    def comparacao_inicial(self) -> ComparacaoPValores:
        return ComparacaoPValores(
            titulo_esquerda="ANOVA",
            titulo_direita="Kruskal-Wallis",
            hipoteses=[HIPOTESE_CARD],
            linhas=[(None, None)],
        )

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        coluna, grupo = params.get("coluna"), params.get("grupo")
        variante = params.get("variante", CLASSICA)
        erros = erros_coluna(df, coluna) + erros_coluna(df, grupo, "o fator", numerica=False)
        if not erros and coluna == grupo:
            erros.append("A variável e o fator devem ser colunas diferentes.")
        erros += erro_opcao(variante, VARIANTES_ANOVA, "Escolha uma variante válida.")
        if not erros:
            erros += self._erros_dados(df, coluna, grupo, variante)
        erros += erro_alfa(params.get("alfa", 0.05))
        return erros

    @staticmethod
    def _erros_dados(df: pd.DataFrame, coluna: str, grupo: str, variante: str) -> list[str]:
        niveis, amostras, _ = TesteT2Amostras.separar(df, coluna, grupo)
        k = len(niveis)
        if not 2 <= k <= MAX_GRUPOS_ANOVA:
            return [
                f"O fator '{grupo}' deve ter de 2 a {MAX_GRUPOS_ANOVA} níveis com dados válidos "
                f"(tem {k})."
            ]
        pequenos = [
            f"'{rotulo_nivel(nivel)}' ({len(x)})"
            for nivel, x in zip(niveis, amostras, strict=True)
            if len(x) < 2
        ]
        if pequenos:
            return [
                "Cada grupo precisa de ao menos 2 observações válidas; grupos com menos: "
                + ", ".join(pequenos)
                + "."
            ]
        if all(np.ptp(x) == 0 for x in amostras):
            return [
                f"A variância de '{coluna}' dentro dos grupos é zero (todos os valores de cada "
                "grupo são iguais): a ANOVA não pode ser calculada."
            ]
        constantes = [
            f"'{rotulo_nivel(nivel)}'"
            for nivel, x in zip(niveis, amostras, strict=True)
            if np.ptp(x) == 0
        ]
        if variante == WELCH and constantes:
            return [
                "A variante Welch exige variância maior que zero em cada grupo; grupos com todos "
                "os valores iguais: " + ", ".join(constantes) + ". Use a variante clássica."
            ]
        return []

    # ---------------- Execução ----------------
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)

        coluna, grupo = params["coluna"], params["grupo"]
        alfa = float(params.get("alfa", 0.05))
        welch = params.get("variante", CLASSICA) == WELCH
        pedir_tukey = bool(params.get("tukey", False))

        niveis, amostras, descartadas = TesteT2Amostras.separar(df, coluna, grupo)
        rotulos = [rotulo_nivel(n) for n in niveis]
        tamanhos = [len(x) for x in amostras]
        k, total = len(amostras), sum(tamanhos)

        sq_entre, sq_dentro = somas_de_quadrados(amostras)
        sq_total = sq_entre + sq_dentro
        gl_entre, gl_dentro = k - 1, total - k
        qm_entre, qm_dentro = sq_entre / gl_entre, sq_dentro / gl_dentro
        classica = stats.f_oneway(*amostras)
        f_classico, p_classico = float(classica.statistic), float(classica.pvalue)
        eta2 = sq_entre / sq_total
        omega2 = (sq_entre - gl_entre * qm_dentro) / (sq_total + qm_dentro)

        if welch:
            resultado = stats.f_oneway(*amostras, equal_var=False)
            f, p_valor = float(resultado.statistic), float(resultado.pvalue)
            gl2 = self._gl_welch(amostras)
        else:
            f, p_valor, gl2 = f_classico, p_classico, float(gl_dentro)

        estatisticas = {
            "k": float(k),
            "n": float(total),
            "n_descartadas": float(descartadas),
            "f": f,
            "gl1": float(gl_entre),
            "gl2": gl2,
            "p_valor": p_valor,
            "sq_entre": sq_entre,
            "sq_dentro": sq_dentro,
            "sq_total": sq_total,
            "qm_entre": qm_entre,
            "qm_dentro": qm_dentro,
            "f_classico": f_classico,
            "p_classico": p_classico,
            "eta2": eta2,
            "omega2": omega2,
        }

        interpretacao = interpretar(
            p_valor,
            alfa,
            h0=f"a média de '{coluna}' é a mesma nos {k} grupos de '{grupo}'",
            h1="ao menos uma média difere das demais",
            conclusao_rejeita=(
                f"Há evidência estatística de que a média de '{coluna}' difere entre os grupos "
                f"de '{grupo}'."
            ),
            conclusao_nao_rejeita=(
                f"Não há evidência suficiente de que a média de '{coluna}' difira entre os "
                f"grupos de '{grupo}'."
            ),
        )
        rejeita = decidir(p_valor, alfa) == REJEITA_H0

        avisos = []
        if descartadas:
            avisos.append(f"{descartadas} linha(s) com valor ou grupo ausente foram descartadas.")
        p_levene = self._levene(amostras)
        estatisticas["p_levene"] = p_levene
        if not math.isnan(p_levene) and p_levene < ALFA_LEVENE:
            sugestao = (
                "a variante Welch é a adequada."
                if welch
                else "considere a variante Welch, que não assume variâncias iguais."
            )
            avisos.append(
                f"O teste de Levene (centrado na mediana) indica variâncias diferentes entre os "
                f"grupos (p = {formatar_p_valor(p_levene)}): {sugestao}"
            )

        tabelas = {
            "Resumo": pd.DataFrame(self._resumo(estatisticas, welch), columns=["Medida", "Valor"]),
            "Tabela ANOVA": self._tabela_anova(estatisticas),
            "Grupos": self._tabela_grupos(rotulos, amostras, alfa),
        }

        if pedir_tukey and rejeita:
            tabelas[TABELA_TUKEY] = self._tukey(rotulos, amostras, alfa)
            estatisticas["comparacoes"] = float(k * (k - 1) // 2)
            if welch:
                avisos.append(
                    "O Tukey HSD assume variâncias iguais; com a variante Welch, interprete as "
                    "comparações com cautela."
                )
        elif pedir_tukey:
            avisos.append(
                "Comparações múltiplas (Tukey HSD) não exibidas: H₀ não foi rejeitada, então não "
                "há diferença a localizar entre os grupos."
            )

        p_kw = float(stats.kruskal(*amostras).pvalue) if np.ptp(np.concatenate(amostras)) else None

        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=interpretacao,
            tabelas=tabelas,
            figuras=[
                boxplot(
                    list(zip(rotulos, amostras, strict=True)), f"'{coluna}' por '{grupo}'", coluna
                )
            ],
            avisos=avisos,
            comparacao=ComparacaoPValores(
                titulo_esquerda="ANOVA",
                titulo_direita="Kruskal-Wallis",
                hipoteses=[HIPOTESE_CARD],
                linhas=[(p_valor, p_kw)],
            ),
        )

    # ---------------- Auxiliares ----------------
    @staticmethod
    def _gl_welch(amostras: list[np.ndarray]) -> float:
        """gl do denominador do F de Welch: (k² − 1) / (3·Σ (1 − wᵢ/W)²/(nᵢ − 1)), wᵢ = nᵢ/sᵢ²."""
        k = len(amostras)
        pesos = np.array([len(x) / x.var(ddof=1) for x in amostras])
        termo = sum(
            (1 - w / pesos.sum()) ** 2 / (len(x) - 1) for w, x in zip(pesos, amostras, strict=True)
        )
        return float((k**2 - 1) / (3 * termo))

    @staticmethod
    def _levene(amostras: list[np.ndarray]) -> float:
        """p do Levene centrado na mediana; NaN quando não calculável (desvios todos nulos)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p = float(stats.levene(*amostras, center="median").pvalue)
        return p

    @staticmethod
    def _resumo(e: dict[str, float], welch: bool) -> list[tuple[str, str]]:
        if welch:
            teste = [
                ("Variante", WELCH),
                ("F de Welch", formatar_numero(e["f"])),
                (
                    "Graus de liberdade (numerador; denominador)",
                    f"{e['gl1']:.0f}; {formatar_numero(e['gl2'], 2)}",
                ),
                ("p-valor (Welch)", formatar_p_valor(e["p_valor"])),
            ]
        else:
            teste = [
                ("Variante", CLASSICA),
                ("Estatística F", formatar_numero(e["f"])),
                ("Graus de liberdade (numerador; denominador)", f"{e['gl1']:.0f}; {e['gl2']:.0f}"),
                ("p-valor", formatar_p_valor(e["p_valor"])),
            ]
        return [
            ("Grupos (k)", f"{e['k']:.0f}"),
            ("Observações (N)", f"{e['n']:.0f}"),
            *teste,
            ("η² (eta quadrado)", formatar_numero(e["eta2"])),
            ("ω² (ômega quadrado)", formatar_numero(e["omega2"])),
            ("p-valor do Levene (mediana)", formatar_p_valor(e["p_levene"])),
        ]

    @staticmethod
    def _tabela_anova(e: dict[str, float]) -> pd.DataFrame:
        gl_total = e["n"] - 1
        return pd.DataFrame(
            [
                (
                    "Entre grupos",
                    formatar_numero(e["sq_entre"]),
                    f"{e['gl1']:.0f}",
                    formatar_numero(e["qm_entre"]),
                    formatar_numero(e["f_classico"]),
                    formatar_p_valor(e["p_classico"]),
                ),
                (
                    "Dentro dos grupos",
                    formatar_numero(e["sq_dentro"]),
                    f"{e['n'] - e['k']:.0f}",
                    formatar_numero(e["qm_dentro"]),
                    "",
                    "",
                ),
                ("Total", formatar_numero(e["sq_total"]), f"{gl_total:.0f}", "", "", ""),
            ],
            columns=["Fonte", "SQ", "gl", "QM", "F", "p"],
        )

    @staticmethod
    def _tabela_grupos(rotulos: list[str], amostras: list[np.ndarray], alfa: float) -> pd.DataFrame:
        confianca = f"{(1 - alfa) * 100:.0f}%"
        linhas = []
        for rotulo, x in zip(rotulos, amostras, strict=True):
            media, desvio = float(x.mean()), float(x.std(ddof=1))
            margem = float(stats.t.ppf(1 - alfa / 2, len(x) - 1)) * desvio / math.sqrt(len(x))
            linhas.append(
                (
                    rotulo,
                    len(x),
                    formatar_numero(media),
                    formatar_numero(desvio),
                    f"[{formatar_numero(media - margem)}; {formatar_numero(media + margem)}]",
                )
            )
        return pd.DataFrame(
            linhas, columns=["Grupo", "n", "Média", "Desvio padrão", f"IC {confianca} da média"]
        )

    @staticmethod
    def _tukey(rotulos: list[str], amostras: list[np.ndarray], alfa: float) -> pd.DataFrame:
        resultado = stats.tukey_hsd(*amostras)
        ic = resultado.confidence_interval(confidence_level=1 - alfa)
        confianca = f"{(1 - alfa) * 100:.0f}%"
        linhas = []
        for i, j in itertools.combinations(range(len(amostras)), 2):
            p = float(resultado.pvalue[i, j])
            linhas.append(
                (
                    f"'{rotulos[i]}' × '{rotulos[j]}'",
                    formatar_numero(float(resultado.statistic[i, j])),
                    f"[{formatar_numero(float(ic.low[i, j]))}; "
                    f"{formatar_numero(float(ic.high[i, j]))}]",
                    formatar_p_valor(p),
                    "Sim" if p <= alfa else "Não",
                )
            )
        return pd.DataFrame(
            linhas,
            columns=[
                "Comparação",
                "Diferença de médias",
                f"IC {confianca} (Tukey)",
                "p ajustado (Tukey)",
                "Diferem?",
            ],
        )
