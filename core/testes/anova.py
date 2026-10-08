"""Análise de variância (ANOVA)."""

import itertools
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from core.base import ComparacaoPValores, ErroValidacao, ParametroSpec, ResultadoTeste, TesteBase
from core.diagnosticos import aviso_normalidade, normalidade_amostra, rotulo_normalidade
from core.figuras import barras_agrupadas, boxplot
from core.interpretacao import (
    REJEITA_H0,
    decidir,
    formatar_alfa,
    formatar_numero,
    formatar_p_valor,
    interpretar,
)
from core.testes.medias import TesteT2Amostras
from core.tipos import ordenar_niveis, rotulo_nivel
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
        # Normalidade dos resíduos: cada valor menos a média do seu grupo.
        normal = normalidade_amostra(np.concatenate([x - x.mean() for x in amostras]))
        estatisticas["p_normalidade"] = normal.p_valor if normal else math.nan
        aviso = aviso_normalidade(
            normal, "os resíduos da ANOVA", "o Kruskal-Wallis", "a ANOVA", plural=True
        )
        if aviso:
            avisos.append(aviso)

        tabelas = {
            "Resumo": pd.DataFrame(
                self._resumo(estatisticas, welch, rotulo_normalidade(normal, "resíduos")),
                columns=["Medida", "Valor"],
            ),
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
    def _resumo(e: dict[str, float], welch: bool, rotulo_normal: str) -> list[tuple[str, str]]:
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
            (rotulo_normal, formatar_p_valor(e["p_normalidade"])),
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


# ---------------------------------------------------------------------------
# ANOVA de 2 fatores
# ---------------------------------------------------------------------------
TIPO_II = "Tipo II"
TIPO_III = "Tipo III"
TIPOS_SQ = (TIPO_II, TIPO_III)
TABELA_MEDIAS = "Médias por combinação"
_MAX_CELULAS_LISTADAS = 5


def _texto_p(p: float) -> str:
    """Texto "p = 0,012" ou "p < 0,001"."""
    texto = formatar_p_valor(p)
    return f"p {texto}" if texto.startswith("<") else f"p = {texto}"


class TesteAnova2Fatores(TesteBase):
    """ANOVA de 2 fatores (efeitos principais de A e B e, opcionalmente, a interação A × B).

    Entrada: coluna numérica + dois fatores diferentes, cada um com 2 a 20 níveis; toda combinação
    de níveis precisa de dados (≥ 1 observação; ≥ 2 com a interação, para haver variação dentro
    das combinações). Linhas incompletas descartadas (aviso). Modelo linear com codificação por
    soma (`statsmodels` `ols` + `anova_lm`); somas de quadrados Tipo II (padrão) ou Tipo III — em
    desenho balanceado coincidem; no desbalanceado há aviso. Tabela por fonte (SQ, gl, QM, F, p,
    η² parcial = SQ/(SQ + SQ resíduo)). Decisão principal: com a interação, a do termo A × B;
    sem ela, rejeita H₀ se algum efeito principal tiver p ≤ α (p-valor exibido = o menor dos
    dois). A interpretação comenta cada efeito. Levene (mediana) entre as combinações só como
    aviso quando p < 0,05 (exige ≥ 2 observações por combinação). Sem pós-teste e sem card.
    """

    id = "anova_2fator"
    nome = "ANOVA (2 fatores)"
    grupo = "ANOVA"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec("coluna", "Variável", "coluna_numerica"),
            ParametroSpec("fator_a", "Fator A", "coluna_categorica"),
            ParametroSpec("fator_b", "Fator B", "coluna_categorica"),
            ParametroSpec("interacao", "Incluir interação A × B", "booleano", padrao=True),
            ParametroSpec(
                "tipo_sq", "Soma de quadrados", "opcao", padrao=TIPO_II, opcoes=list(TIPOS_SQ)
            ),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    @staticmethod
    def preparar(
        df: pd.DataFrame, coluna: str, fator_a: str, fator_b: str
    ) -> tuple[pd.DataFrame, list, list, int]:
        """(dados com y, a, b — códigos dos níveis —, níveis de A, níveis de B, descartadas)."""
        validas = df[[coluna, fator_a, fator_b]].dropna()
        niveis_a = ordenar_niveis(list(pd.unique(validas[fator_a])))
        niveis_b = ordenar_niveis(list(pd.unique(validas[fator_b])))
        dados = pd.DataFrame(
            {
                "y": validas[coluna].astype(float).to_numpy(),
                "a": pd.Categorical(validas[fator_a], categories=niveis_a).codes,
                "b": pd.Categorical(validas[fator_b], categories=niveis_b).codes,
            }
        )
        return dados, niveis_a, niveis_b, len(df) - len(validas)

    @staticmethod
    def contagens(dados: pd.DataFrame, na: int, nb: int) -> np.ndarray:
        """Matriz na × nb com o número de observações de cada combinação."""
        tabela = np.zeros((na, nb), dtype=int)
        np.add.at(tabela, (dados["a"].to_numpy(), dados["b"].to_numpy()), 1)
        return tabela

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        coluna = params.get("coluna")
        fator_a, fator_b = params.get("fator_a"), params.get("fator_b")
        erros = (
            erros_coluna(df, coluna)
            + erros_coluna(df, fator_a, "o fator A", numerica=False)
            + erros_coluna(df, fator_b, "o fator B", numerica=False)
        )
        if not erros and len({coluna, fator_a, fator_b}) < 3:
            erros.append("A variável e os dois fatores devem ser colunas diferentes.")
        erros += erro_opcao(
            params.get("tipo_sq", TIPO_II), TIPOS_SQ, "Escolha o tipo de soma de quadrados."
        )
        if not erros:
            erros += self._erros_dados(
                df, coluna, fator_a, fator_b, bool(params.get("interacao", True))
            )
        erros += erro_alfa(params.get("alfa", 0.05))
        return erros

    def _erros_dados(
        self, df: pd.DataFrame, coluna: str, fator_a: str, fator_b: str, interacao: bool
    ) -> list[str]:
        dados, niveis_a, niveis_b, _ = self.preparar(df, coluna, fator_a, fator_b)
        erros = [
            f"O fator '{nome}' deve ter de 2 a {MAX_GRUPOS_ANOVA} níveis com dados válidos "
            f"(tem {len(niveis)})."
            for nome, niveis in ((fator_a, niveis_a), (fator_b, niveis_b))
            if not 2 <= len(niveis) <= MAX_GRUPOS_ANOVA
        ]
        if erros:
            return erros
        na, nb = len(niveis_a), len(niveis_b)
        contagens = self.contagens(dados, na, nb)
        minimo = 2 if interacao else 1
        faltam = [
            f"'{rotulo_nivel(niveis_a[i])}' × '{rotulo_nivel(niveis_b[j])}' ({contagens[i, j]})"
            for i, j in zip(*np.nonzero(contagens < minimo), strict=True)
        ]
        if faltam:
            lista = ", ".join(faltam[:_MAX_CELULAS_LISTADAS])
            if len(faltam) > _MAX_CELULAS_LISTADAS:
                lista += f" e mais {len(faltam) - _MAX_CELULAS_LISTADAS}"
            if interacao:
                return [
                    "Com a interação, cada combinação dos níveis precisa de ao menos 2 "
                    f"observações; com menos: {lista}. Desmarque a interação ou reúna mais dados."
                ]
            return [
                f"Toda combinação dos níveis precisa de ao menos 1 observação; vazias: {lista}."
            ]
        parametros = na * nb if interacao else na + nb - 1
        if len(dados) <= parametros:
            return [
                f"Observações insuficientes: o modelo tem {parametros} parâmetros e há "
                f"{len(dados)} linhas completas; não sobram graus de liberdade para o resíduo."
            ]
        y = dados["y"].to_numpy()
        sq_total = float(((y - y.mean()) ** 2).sum())
        if self._sq_residuo(dados, na, nb, interacao) <= 1e-12 * max(1.0, sq_total):
            return [
                f"A variância residual de '{coluna}' é zero (o modelo reproduz todos os valores): "
                "a ANOVA não pode ser calculada."
            ]
        return []

    @staticmethod
    def _sq_residuo(dados: pd.DataFrame, na: int, nb: int, interacao: bool) -> float:
        """SQ do resíduo por mínimos quadrados (para detectar ajuste perfeito)."""
        a = np.eye(na)[dados["a"].to_numpy()]
        b = np.eye(nb)[dados["b"].to_numpy()]
        if interacao:
            x = np.einsum("ni,nj->nij", a, b).reshape(len(dados), -1)
        else:
            x = np.column_stack([a, b[:, 1:]])
        y = dados["y"].to_numpy()
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        return float(((y - x @ coef) ** 2).sum())

    # ---------------- Execução ----------------
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)

        import statsmodels.formula.api as smf  # import tardio (abre o app mais rápido)
        from statsmodels.stats.anova import anova_lm

        coluna, fator_a, fator_b = params["coluna"], params["fator_a"], params["fator_b"]
        alfa = float(params.get("alfa", 0.05))
        interacao = bool(params.get("interacao", True))
        tipo_sq = params.get("tipo_sq", TIPO_II)

        dados, niveis_a, niveis_b, descartadas = self.preparar(df, coluna, fator_a, fator_b)
        rotulos_a = [rotulo_nivel(n) for n in niveis_a]
        rotulos_b = [rotulo_nivel(n) for n in niveis_b]
        contagens = self.contagens(dados, len(niveis_a), len(niveis_b))
        balanceado = bool((contagens == contagens.flat[0]).all())

        formula = "y ~ C(a, Sum) * C(b, Sum)" if interacao else "y ~ C(a, Sum) + C(b, Sum)"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo = smf.ols(formula, data=dados).fit()
            tabela = anova_lm(modelo, typ=2 if tipo_sq == TIPO_II else 3)

        sq_res = float(tabela.loc["Residual", "sum_sq"])
        gl_res = float(tabela.loc["Residual", "df"])
        estatisticas: dict[str, float] = {
            "n": float(len(dados)),
            "n_descartadas": float(descartadas),
            "niveis_a": float(len(niveis_a)),
            "niveis_b": float(len(niveis_b)),
            "balanceado": float(balanceado),
            "sq_residuo": sq_res,
            "gl_residuo": gl_res,
            "qm_residuo": sq_res / gl_res,
            "r2": float(modelo.rsquared),
        }
        termos = {
            "C(a, Sum)": ("a", f"'{fator_a}'"),
            "C(b, Sum)": ("b", f"'{fator_b}'"),
            "C(a, Sum):C(b, Sum)": ("ab", f"'{fator_a}' × '{fator_b}'"),
        }
        linhas_anova = []
        for termo, (chave, rotulo) in termos.items():
            if termo not in tabela.index:
                continue
            sq, gl = float(tabela.loc[termo, "sum_sq"]), float(tabela.loc[termo, "df"])
            f, p = float(tabela.loc[termo, "F"]), float(tabela.loc[termo, "PR(>F)"])
            eta2p = sq / (sq + sq_res)
            estatisticas |= {
                f"sq_{chave}": sq,
                f"gl_{chave}": gl,
                f"f_{chave}": f,
                f"p_{chave}": p,
                f"eta2p_{chave}": eta2p,
            }
            linhas_anova.append(
                (
                    rotulo,
                    formatar_numero(sq),
                    f"{gl:.0f}",
                    formatar_numero(sq / gl),
                    formatar_numero(f),
                    formatar_p_valor(p),
                    formatar_numero(eta2p),
                )
            )
        linhas_anova.append(
            (
                "Resíduo",
                formatar_numero(sq_res),
                f"{gl_res:.0f}",
                formatar_numero(sq_res / gl_res),
                "",
                "",
                "",
            )
        )

        if interacao:
            p_valor = estatisticas["p_ab"]
        else:
            p_valor = min(estatisticas["p_a"], estatisticas["p_b"])
        estatisticas["p_valor"] = p_valor

        avisos = self._avisos(dados, contagens, descartadas, balanceado, tipo_sq, estatisticas)
        normal = normalidade_amostra(np.asarray(modelo.resid))
        estatisticas["p_normalidade"] = normal.p_valor if normal else math.nan
        aviso = aviso_normalidade(
            normal,
            "os resíduos da ANOVA",
            f"uma transformação de '{coluna}' (ex.: logaritmo)",
            "a ANOVA",
            plural=True,
        )
        if aviso:
            avisos.append(aviso)
        resumo = [
            ("Variável", f"'{coluna}'"),
            ("Fator A", f"'{fator_a}' ({len(niveis_a)} níveis)"),
            ("Fator B", f"'{fator_b}' ({len(niveis_b)} níveis)"),
            ("Observações (N)", f"{len(dados)}"),
            ("Interação A × B", "Incluída" if interacao else "Não incluída"),
            ("Soma de quadrados", tipo_sq),
            ("Desenho", "Balanceado" if balanceado else "Desbalanceado"),
            ("R² do modelo", formatar_numero(estatisticas["r2"])),
            ("p-valor do Levene (mediana)", formatar_p_valor(estatisticas["p_levene"])),
            (
                rotulo_normalidade(normal, "resíduos"),
                formatar_p_valor(estatisticas["p_normalidade"]),
            ),
        ]
        medias = self._medias(dados, contagens)
        tabela_medias = pd.DataFrame(
            [
                (
                    rotulos_a[i],
                    rotulos_b[j],
                    int(contagens[i, j]),
                    formatar_numero(m),
                    formatar_numero(s),
                )
                for (i, j), (m, s) in medias.items()
            ],
            columns=[f"{fator_a} (A)", f"{fator_b} (B)", "n", "Média", "Desvio padrão"],
        )
        tabelas = {
            "Resumo": pd.DataFrame(resumo, columns=["Medida", "Valor"]),
            "Tabela ANOVA": pd.DataFrame(
                linhas_anova, columns=["Fonte", "SQ", "gl", "QM", "F", "p", "η² parcial"]
            ),
            TABELA_MEDIAS: tabela_medias,
        }
        figura = barras_agrupadas(
            [
                (rotulos_a[i], [medias[(i, j)][0] for j in range(len(niveis_b))])
                for i in range(len(niveis_a))
            ],
            rotulos_b,
            f"Média de '{coluna}' por '{fator_a}' e '{fator_b}'",
            f"Média de '{coluna}'",
        )

        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=self._interpretacao(
                estatisticas, alfa, coluna, fator_a, fator_b, interacao
            ),
            tabelas=tabelas,
            figuras=[figura],
            avisos=avisos,
        )

    # ---------------- Auxiliares ----------------
    def _avisos(
        self,
        dados: pd.DataFrame,
        contagens: np.ndarray,
        descartadas: int,
        balanceado: bool,
        tipo_sq: str,
        estatisticas: dict[str, float],
    ) -> list[str]:
        """Avisos não bloqueantes; grava o p do Levene em `estatisticas`."""
        avisos = []
        if descartadas:
            avisos.append(f"{descartadas} linha(s) com valor ou fator ausente foram descartadas.")
        if not balanceado:
            avisos.append(
                "Desenho desbalanceado (combinações com números diferentes de observações): as "
                "somas de quadrados dependem do tipo escolhido. O Tipo II testa cada efeito "
                "principal ajustado pelo outro (adequado quando não há interação); o Tipo III "
                f"ajusta cada efeito por todos os demais. Resultado com o {tipo_sq}."
            )
        p_levene = self._levene_celulas(dados, contagens)
        estatisticas["p_levene"] = p_levene
        if not math.isnan(p_levene) and p_levene < ALFA_LEVENE:
            avisos.append(
                "O teste de Levene (centrado na mediana) indica variâncias diferentes entre as "
                f"combinações dos níveis (p = {formatar_p_valor(p_levene)}): os p-valores da "
                "ANOVA podem ficar imprecisos, sobretudo com combinações de tamanhos diferentes."
            )
        return avisos

    @staticmethod
    def _medias(
        dados: pd.DataFrame, contagens: np.ndarray
    ) -> dict[tuple[int, int], tuple[float, float | None]]:
        """(média, desvio padrão amostral ou None se n = 1) de cada combinação, em ordem."""
        resultado = {}
        for i, j in itertools.product(range(contagens.shape[0]), range(contagens.shape[1])):
            y = dados.loc[(dados["a"] == i) & (dados["b"] == j), "y"].to_numpy()
            resultado[(i, j)] = (float(y.mean()), float(y.std(ddof=1)) if len(y) > 1 else None)
        return resultado

    @staticmethod
    def _levene_celulas(dados: pd.DataFrame, contagens: np.ndarray) -> float:
        """p do Levene (mediana) entre as combinações; NaN se alguma tem menos de 2 observações."""
        if (contagens < 2).any():
            return math.nan
        celulas = [y.to_numpy() for _, y in dados.groupby(["a", "b"])["y"]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(stats.levene(*celulas, center="median").pvalue)

    @staticmethod
    def _interpretacao(
        e: dict[str, float], alfa: float, coluna: str, fator_a: str, fator_b: str, interacao: bool
    ) -> str:
        def rejeita(chave: str) -> bool:
            return decidir(e[f"p_{chave}"], alfa) == REJEITA_H0

        def frase(chave: str, efeito: str, h0: str) -> str:
            acao = "rejeita-se H₀" if rejeita(chave) else "não se rejeita H₀"
            return f"{efeito[0].upper()}{efeito[1:]}: {_texto_p(e[f'p_{chave}'])}, {acao} ({h0})."

        partes = [f"Com α = {formatar_alfa(alfa)}:"]
        if interacao:
            partes.append(
                frase(
                    "ab",
                    f"interação '{fator_a}' × '{fator_b}'",
                    f"o efeito de '{fator_a}' sobre '{coluna}' não depende de '{fator_b}'",
                )
            )
        for chave, fator in (("a", fator_a), ("b", fator_b)):
            partes.append(
                frase(
                    chave,
                    f"efeito de '{fator}'",
                    f"a média de '{coluna}' é a mesma em todos os níveis de '{fator}'",
                )
            )
        if interacao and rejeita("ab"):
            partes.append(
                f"Há evidência estatística de interação: o efeito de '{fator_a}' sobre "
                f"'{coluna}' depende do nível de '{fator_b}'. Interprete os efeitos principais "
                "com cautela e compare as médias por combinação."
            )
            return " ".join(partes)
        significativos = [f for chave, f in (("a", fator_a), ("b", fator_b)) if rejeita(chave)]
        if significativos:
            fatores = " e de ".join(f"'{f}'" for f in significativos)
            partes.append(
                f"Há evidência estatística de que a média de '{coluna}' difere entre os níveis "
                f"de {fatores}."
            )
        else:
            partes.append(
                f"Não há evidência suficiente de efeito de '{fator_a}' ou de '{fator_b}' sobre a "
                f"média de '{coluna}'."
            )
        return " ".join(partes)
