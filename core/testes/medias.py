"""Testes de médias."""

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from core.base import (
    ComparacaoPValores,
    ErroValidacao,
    ParametroSpec,
    ResultadoTeste,
    TesteBase,
)
from core.figuras import boxplot, histograma
from core.interpretacao import decidir, formatar_numero, formatar_p_valor, interpretar
from core.validacao import converter_numero, erro_alfa, erro_numero, erro_opcao, erros_coluna

# Rótulo exibido na UI → valor de `alternative` do scipy. A ordem é a das linhas do card.
ALTERNATIVAS = {"μ ≠ μ₀": "two-sided", "μ > μ₀": "greater", "μ < μ₀": "less"}
ALTERNATIVA_PADRAO = "μ ≠ μ₀"
_SIMBOLO = {"two-sided": "≠", "greater": ">", "less": "<"}
_TEXTO = {"two-sided": "diferente de", "greater": "maior que", "less": "menor que"}
N_PEQUENO = 30


class TesteT1Amostra(TesteBase):
    """Teste t de Student para uma amostra (H₀: μ = μ₀).

    Wrapper de `scipy.stats.ttest_1samp`, com variância amostral (ddof=1) e gl = n − 1.
    Equivalente não paramétrico no card: Wilcoxon dos postos sinalizados de x − μ₀
    (`scipy.stats.wilcoxon`, zero_method="wilcox": diferenças nulas são descartadas;
    sem correção de continuidade; method="auto": exato para amostras pequenas sem empates).
    Valores ausentes são ignorados (com aviso). Decisão: rejeita H₀ quando p ≤ α.
    """

    id = "teste_t_1am"
    nome = "Teste t (uma amostra)"
    grupo = "Médias"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec("coluna", "Variável", "coluna_numerica"),
            ParametroSpec("mu0", "Média Hipotética", "numero"),
            ParametroSpec(
                "alternativa",
                "Hipótese alternativa (H₁)",
                "opcao",
                padrao=ALTERNATIVA_PADRAO,
                opcoes=list(ALTERNATIVAS),
            ),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    def comparacao_inicial(self) -> ComparacaoPValores:
        return ComparacaoPValores(
            titulo_esquerda="t Student",
            titulo_direita="Wilcoxon",
            hipoteses=list(ALTERNATIVAS),
            linhas=[(None, None)] * len(ALTERNATIVAS),
        )

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        coluna = params.get("coluna")
        erros = erros_coluna(df, coluna)
        if not erros:
            x = df[coluna].dropna()
            if len(x) < 2:
                erros.append(
                    f"São necessárias ao menos 2 observações válidas em '{coluna}' (há {len(x)})."
                )
            elif x.nunique() == 1:
                erros.append(
                    f"Todos os valores de '{coluna}' são iguais (variância zero): "
                    "o teste t não pode ser calculado."
                )
        erros += erro_numero(
            params.get("mu0"), "Informe um número válido para a média hipotética (μ₀)."
        )
        erros += erro_opcao(
            params.get("alternativa", ALTERNATIVA_PADRAO),
            ALTERNATIVAS,
            "Escolha uma hipótese alternativa válida.",
        )
        erros += erro_alfa(params.get("alfa", 0.05))
        return erros

    # ---------------- Execução ----------------
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)

        coluna = params["coluna"]
        mu0 = converter_numero(params["mu0"])
        alfa = float(params.get("alfa", 0.05))
        rotulo_alt = params.get("alternativa", ALTERNATIVA_PADRAO)
        alternativa = ALTERNATIVAS[rotulo_alt]

        serie = df[coluna]
        x = serie.dropna().astype(float).to_numpy()
        n, n_ausentes = len(x), int(serie.isna().sum())

        testes_t = {alt: stats.ttest_1samp(x, mu0, alternative=alt) for alt in _SIMBOLO}
        t = testes_t[alternativa]
        p_valor = float(t.pvalue)
        ic = t.confidence_interval(confidence_level=1 - alfa)

        media = float(x.mean())
        desvio = float(x.std(ddof=1))
        erro_padrao = desvio / math.sqrt(n)
        d_cohen = (media - mu0) / desvio

        p_wilcoxon, avisos_w = self._wilcoxon(x, mu0)

        estatisticas = {
            "n": float(n),
            "n_ausentes": float(n_ausentes),
            "media": media,
            "desvio_padrao": desvio,
            "erro_padrao": erro_padrao,
            "mu0": mu0,
            "diferenca": media - mu0,
            "t": float(t.statistic),
            "gl": float(t.df),
            "p_valor": p_valor,
            "ic_inferior": float(ic.low),
            "ic_superior": float(ic.high),
            "d_cohen": d_cohen,
        }
        if p_wilcoxon[alternativa] is not None:
            estatisticas["p_wilcoxon"] = p_wilcoxon[alternativa]

        mu0_txt = formatar_numero(mu0, 4).rstrip("0").rstrip(",")
        simbolo, texto = _SIMBOLO[alternativa], _TEXTO[alternativa]
        interpretacao = interpretar(
            p_valor,
            alfa,
            h0=f"μ = {mu0_txt}",
            h1=f"μ {simbolo} {mu0_txt}",
            conclusao_rejeita=(
                f"Há evidência estatística de que a média de '{coluna}' é {texto} {mu0_txt}."
            ),
            conclusao_nao_rejeita=(
                f"Não há evidência suficiente de que a média de '{coluna}' seja {texto} {mu0_txt}."
            ),
        )

        avisos = []
        if n_ausentes:
            avisos.append(f"{n_ausentes} valor(es) ausente(s) em '{coluna}' foram ignorados.")
        if n < N_PEQUENO:
            avisos.append(
                f"Amostra pequena (n = {n}): o teste t supõe distribuição aproximadamente normal "
                "da variável. Compare com o Wilcoxon no card."
            )
        avisos += avisos_w

        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=interpretacao,
            tabelas={"Resumo": self._tabela_resumo(estatisticas, alfa, alternativa, p_wilcoxon)},
            figuras=[
                histograma(
                    x,
                    f"Distribuição de '{coluna}'",
                    coluna,
                    [
                        (f"Média amostral (x̄ = {formatar_numero(media, 2)})", media, "destaque"),
                        (f"Média hipotética (μ₀ = {mu0_txt})", mu0, "tracejado"),
                    ],
                )
            ],
            avisos=avisos,
            comparacao=ComparacaoPValores(
                titulo_esquerda="t Student",
                titulo_direita="Wilcoxon",
                hipoteses=list(ALTERNATIVAS),
                linhas=[(float(testes_t[a].pvalue), p_wilcoxon[a]) for a in ALTERNATIVAS.values()],
            ),
        )

    @staticmethod
    def _wilcoxon(x: np.ndarray, mu0: float) -> tuple[dict[str, float | None], list[str]]:
        """p-valores do Wilcoxon para as três alternativas e avisos sobre zeros."""
        diferencas = x - mu0
        zeros = int((diferencas == 0).sum())
        if zeros == len(diferencas):
            return dict.fromkeys(_SIMBOLO), [
                "Wilcoxon não calculado: todas as observações são iguais a μ₀."
            ]
        p = {
            alt: float(stats.wilcoxon(diferencas, zero_method="wilcox", alternative=alt).pvalue)
            for alt in _SIMBOLO
        }
        avisos = []
        if zeros:
            avisos.append(
                f"No Wilcoxon, {zeros} observação(ões) igual(is) a μ₀ foram descartadas "
                "(diferença nula)."
            )
        return p, avisos

    @staticmethod
    def _tabela_resumo(
        e: dict[str, float], alfa: float, alternativa: str, p_wilcoxon: dict[str, float | None]
    ) -> pd.DataFrame:
        confianca = f"{(1 - alfa) * 100:.0f}%"
        tipo_ic = "" if alternativa == "two-sided" else " (unilateral)"
        ic = f"[{formatar_numero(e['ic_inferior'])}; {formatar_numero(e['ic_superior'])}]"
        linhas = [
            ("Observações (n)", f"{int(e['n'])}"),
            ("Média amostral (x̄)", formatar_numero(e["media"])),
            ("Desvio padrão (s)", formatar_numero(e["desvio_padrao"])),
            ("Erro padrão", formatar_numero(e["erro_padrao"])),
            ("Média hipotética (μ₀)", formatar_numero(e["mu0"])),
            ("Diferença (x̄ − μ₀)", formatar_numero(e["diferenca"])),
            ("Estatística t", formatar_numero(e["t"])),
            ("Graus de liberdade", f"{int(e['gl'])}"),
            ("p-valor (t)", formatar_p_valor(e["p_valor"])),
            (f"IC {confianca} para μ{tipo_ic}", ic),
            ("d de Cohen", formatar_numero(e["d_cohen"])),
            ("p-valor (Wilcoxon)", formatar_p_valor(p_wilcoxon[alternativa])),
        ]
        return pd.DataFrame(linhas, columns=["Medida", "Valor"])


# ---------------------------------------------------------------------------
# Teste t (duas amostras)
# ---------------------------------------------------------------------------
ALTERNATIVAS_2 = {"μ₁ ≠ μ₂": "two-sided", "μ₁ > μ₂": "greater", "μ₁ < μ₂": "less"}
ALTERNATIVA_PADRAO_2 = "μ₁ ≠ μ₂"
VARIANCIAS = {"Diferentes (Welch)": False, "Iguais (pooled)": True}  # → equal_var do scipy
VARIANCIA_PADRAO = "Diferentes (Welch)"
_TEXTO_2 = {"two-sided": "diferente da", "greater": "maior que a", "less": "menor que a"}
RAZAO_VARIANCIAS_AVISO = 4.0


def rotulo_nivel(valor: object) -> str:
    """Texto de um nível de grupo (1.0 → "1")."""
    if isinstance(valor, float) and valor.is_integer():
        return str(int(valor))
    return str(valor)


def ordenar_niveis(niveis: list) -> list:
    """Ordem crescente (numérica ou alfabética); tipos misturados caem para ordem textual."""
    try:
        return sorted(niveis)
    except TypeError:
        return sorted(niveis, key=str)


class TesteT2Amostras(TesteBase):
    """Teste t para duas amostras independentes (H₀: μ₁ = μ₂).

    Entrada: uma coluna numérica e uma coluna de grupo com exatamente 2 níveis. O grupo 1 é o
    primeiro nível em ordem crescente (numérica ou alfabética) e o grupo 2 o segundo; linhas
    com valor ou grupo ausente são descartadas (com aviso).
    Wrapper de `scipy.stats.ttest_ind`: Welch (equal_var=False, padrão de livro-texto, gl de
    Welch–Satterthwaite) ou variâncias iguais (pooled, gl = n₁ + n₂ − 2). O IC é da diferença
    μ₁ − μ₂. d de Cohen = (x̄₁ − x̄₂) / s combinado (pooled) nas duas variantes.
    Equivalente não paramétrico no card: Mann-Whitney U (`scipy.stats.mannwhitneyu`,
    method="auto": exato para amostras pequenas sem empates, senão aproximação normal com
    correção de continuidade). Decisão: rejeita H₀ quando p ≤ α.
    """

    id = "teste_t_2am"
    nome = "Teste t (duas amostras)"
    grupo = "Médias"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec("coluna", "Variável", "coluna_numerica"),
            ParametroSpec("grupo", "Grupo (2 níveis)", "coluna_binaria"),
            ParametroSpec(
                "variancias",
                "Variâncias",
                "opcao",
                padrao=VARIANCIA_PADRAO,
                opcoes=list(VARIANCIAS),
            ),
            ParametroSpec(
                "alternativa",
                "Hipótese alternativa (H₁)",
                "opcao",
                padrao=ALTERNATIVA_PADRAO_2,
                opcoes=list(ALTERNATIVAS_2),
            ),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    def comparacao_inicial(self) -> ComparacaoPValores:
        return ComparacaoPValores(
            titulo_esquerda="t Student",
            titulo_direita="Mann-Whitney",
            hipoteses=list(ALTERNATIVAS_2),
            linhas=[(None, None)] * len(ALTERNATIVAS_2),
        )

    @staticmethod
    def separar(df: pd.DataFrame, coluna: str, grupo: str) -> tuple[list, list[np.ndarray], int]:
        """(níveis em ordem, [x₁, x₂, ...], linhas descartadas por valor/grupo ausente)."""
        validas = df[[coluna, grupo]].dropna()
        niveis = ordenar_niveis(list(pd.unique(validas[grupo])))
        amostras = [
            validas.loc[validas[grupo] == nivel, coluna].astype(float).to_numpy()
            for nivel in niveis
        ]
        return niveis, amostras, len(df) - len(validas)

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        coluna, grupo = params.get("coluna"), params.get("grupo")
        erros = erros_coluna(df, coluna)
        erros_grupo = erros_coluna(df, grupo, "o grupo", numerica=False)
        erros += erros_grupo
        if not erros and coluna == grupo:
            erros.append("A variável e o grupo devem ser colunas diferentes.")
        if not erros:
            niveis, amostras, _ = self.separar(df, coluna, grupo)
            if len(niveis) != 2:
                erros.append(
                    f"A coluna de grupo '{grupo}' deve ter exatamente 2 níveis com dados válidos "
                    f"(tem {len(niveis)})."
                )
            else:
                for nivel, x in zip(niveis, amostras, strict=True):
                    if len(x) < 2:
                        erros.append(
                            f"O grupo '{rotulo_nivel(nivel)}' tem {len(x)} observação(ões) "
                            "válida(s); são necessárias ao menos 2 em cada grupo."
                        )
                if not erros and all(np.ptp(x) == 0 for x in amostras):
                    erros.append(
                        "Os dois grupos têm todos os valores iguais (variância zero): "
                        "o teste t não pode ser calculado."
                    )
        erros += erro_opcao(
            params.get("variancias", VARIANCIA_PADRAO), VARIANCIAS, "Escolha a opção de variâncias."
        )
        erros += erro_opcao(
            params.get("alternativa", ALTERNATIVA_PADRAO_2),
            ALTERNATIVAS_2,
            "Escolha uma hipótese alternativa válida.",
        )
        erros += erro_alfa(params.get("alfa", 0.05))
        return erros

    # ---------------- Execução ----------------
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)

        coluna, grupo = params["coluna"], params["grupo"]
        alfa = float(params.get("alfa", 0.05))
        rotulo_var = params.get("variancias", VARIANCIA_PADRAO)
        iguais = VARIANCIAS[rotulo_var]
        alternativa = ALTERNATIVAS_2[params.get("alternativa", ALTERNATIVA_PADRAO_2)]

        niveis, (x1, x2), descartadas = self.separar(df, coluna, grupo)
        g1, g2 = (rotulo_nivel(n) for n in niveis)
        n1, n2 = len(x1), len(x2)

        with warnings.catch_warnings():
            # Grupo constante: o scipy avisa perda de precisão; o usuário recebe aviso em pt-BR.
            warnings.filterwarnings("ignore", message="Precision loss", category=RuntimeWarning)
            testes_t = {
                alt: stats.ttest_ind(x1, x2, equal_var=iguais, alternative=alt) for alt in _SIMBOLO
            }
        t = testes_t[alternativa]
        p_valor = float(t.pvalue)
        ic = t.confidence_interval(confidence_level=1 - alfa)

        media1, media2 = float(x1.mean()), float(x2.mean())
        dp1, dp2 = float(x1.std(ddof=1)), float(x2.std(ddof=1))
        dp_combinado = math.sqrt(((n1 - 1) * dp1**2 + (n2 - 1) * dp2**2) / (n1 + n2 - 2))
        d_cohen = (media1 - media2) / dp_combinado

        testes_mw = {alt: stats.mannwhitneyu(x1, x2, alternative=alt) for alt in _SIMBOLO}
        p_mw = {alt: float(r.pvalue) for alt, r in testes_mw.items()}

        estatisticas = {
            "n1": float(n1),
            "n2": float(n2),
            "n_descartadas": float(descartadas),
            "media1": media1,
            "media2": media2,
            "dp1": dp1,
            "dp2": dp2,
            "diferenca": media1 - media2,
            "dp_combinado": dp_combinado,
            "t": float(t.statistic),
            "gl": float(t.df),
            "p_valor": p_valor,
            "ic_inferior": float(ic.low),
            "ic_superior": float(ic.high),
            "d_cohen": d_cohen,
            "u": float(testes_mw[alternativa].statistic),
            "p_mann_whitney": p_mw[alternativa],
        }

        simbolo, texto = _SIMBOLO[alternativa], _TEXTO_2[alternativa]
        no_g2 = f"média no grupo '{g2}'"
        interpretacao = interpretar(
            p_valor,
            alfa,
            h0="μ₁ = μ₂",
            h1=f"μ₁ {simbolo} μ₂",
            conclusao_rejeita=(
                f"Há evidência estatística de que a média de '{coluna}' no grupo '{g1}' "
                f"é {texto} {no_g2}."
            ),
            conclusao_nao_rejeita=(
                f"Não há evidência suficiente de que a média de '{coluna}' no grupo '{g1}' "
                f"seja {texto} {no_g2}."
            ),
        )

        avisos = []
        if descartadas:
            avisos.append(f"{descartadas} linha(s) com valor ou grupo ausente foram descartadas.")
        pequenos = [f"'{g}' (n = {n})" for g, n in ((g1, n1), (g2, n2)) if n < N_PEQUENO]
        if pequenos:
            avisos.append(
                f"Amostra pequena em {' e '.join(pequenos)}: o teste t supõe distribuição "
                "aproximadamente normal em cada grupo. Compare com o Mann-Whitney no card."
            )
        for g, dp in ((g1, dp1), (g2, dp2)):
            if dp == 0:
                avisos.append(
                    f"Todos os valores do grupo '{g}' são iguais (variância zero): interprete o "
                    "resultado com cautela."
                )
        razao = max(dp1, dp2) ** 2 / min(dp1, dp2) ** 2 if min(dp1, dp2) > 0 else math.inf
        if iguais and razao > RAZAO_VARIANCIAS_AVISO:
            avisos.append(
                f"A maior variância é {formatar_numero(razao, 1)} vezes a menor: a suposição de "
                "variâncias iguais é duvidosa. Prefira a opção Welch."
            )

        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=interpretacao,
            tabelas={
                "Resumo": self._tabela_resumo(
                    estatisticas, (g1, g2), rotulo_var, alfa, alternativa, iguais
                )
            },
            figuras=[boxplot([(g1, x1), (g2, x2)], f"'{coluna}' por '{grupo}'", coluna)],
            avisos=avisos,
            comparacao=ComparacaoPValores(
                titulo_esquerda="t Student",
                titulo_direita="Mann-Whitney",
                hipoteses=list(ALTERNATIVAS_2),
                linhas=[(float(testes_t[a].pvalue), p_mw[a]) for a in ALTERNATIVAS_2.values()],
            ),
        )

    @staticmethod
    def _tabela_resumo(
        e: dict[str, float],
        grupos: tuple[str, str],
        rotulo_var: str,
        alfa: float,
        alternativa: str,
        iguais: bool,
    ) -> pd.DataFrame:
        confianca = f"{(1 - alfa) * 100:.0f}%"
        tipo_ic = "" if alternativa == "two-sided" else " (unilateral)"
        ic = f"[{formatar_numero(e['ic_inferior'])}; {formatar_numero(e['ic_superior'])}]"
        gl = f"{int(e['gl'])}" if iguais else formatar_numero(e["gl"], 2)
        g1, g2 = grupos
        linhas = [
            ("Grupo 1", f"'{g1}' (n = {int(e['n1'])})"),
            ("Grupo 2", f"'{g2}' (n = {int(e['n2'])})"),
            ("Média do grupo 1 (x̄₁)", formatar_numero(e["media1"])),
            ("Média do grupo 2 (x̄₂)", formatar_numero(e["media2"])),
            ("Desvio padrão do grupo 1", formatar_numero(e["dp1"])),
            ("Desvio padrão do grupo 2", formatar_numero(e["dp2"])),
            ("Diferença (x̄₁ − x̄₂)", formatar_numero(e["diferenca"])),
            ("Variâncias", rotulo_var),
            ("Estatística t", formatar_numero(e["t"])),
            ("Graus de liberdade", gl),
            ("p-valor (t)", formatar_p_valor(e["p_valor"])),
            (f"IC {confianca} para μ₁ − μ₂{tipo_ic}", ic),
            ("d de Cohen", formatar_numero(e["d_cohen"])),
            ("U de Mann-Whitney", formatar_numero(e["u"], 1)),
            ("p-valor (Mann-Whitney)", formatar_p_valor(e["p_mann_whitney"])),
        ]
        return pd.DataFrame(linhas, columns=["Medida", "Valor"])
