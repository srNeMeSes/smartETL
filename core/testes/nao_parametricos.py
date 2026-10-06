"""Testes não paramétricos."""

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

from core.base import ErroValidacao, ParametroSpec, ResultadoTeste, TesteBase
from core.figuras import boxplot, histograma
from core.interpretacao import (
    REJEITA_H0,
    decidir,
    formatar_numero,
    formatar_p_valor,
    interpretar,
)
from core.testes.medias import TesteT2Amostras
from core.tipos import rotulo_nivel
from core.validacao import converter_numero, erro_alfa, erro_numero, erro_opcao, erros_coluna

UMA_AMOSTRA = "Uma amostra"
PAREADO = "Pareado"
MODOS_SINAL = (UMA_AMOSTRA, PAREADO)
ALTERNATIVAS_SINAL = {"M ≠ M₀": "two-sided", "M > M₀": "greater", "M < M₀": "less"}
ALTERNATIVA_PADRAO_SINAL = "M ≠ M₀"
_SIMBOLO = {"two-sided": "≠", "greater": ">", "less": "<"}
_TEXTO = {"two-sided": "diferente de", "greater": "maior que", "less": "menor que"}


def _numero_curto(valor: float) -> str:
    """Número em pt-BR sem zeros à direita ("5", "2,5")."""
    return formatar_numero(valor, 4).rstrip("0").rstrip(",")


def ic_mediana_exato(
    valores: np.ndarray, alfa: float, alternativa: str
) -> tuple[float, float, float]:
    """IC exato para a mediana por estatísticas de ordem: (inferior, superior, confiança obtida).

    Com B ~ Bin(N, 1/2), k é o maior inteiro ≥ 1 com P(B ≤ k − 1) ≤ cauda (α/2 no bilateral,
    α no unilateral); o IC é [x₍ₖ₎, x₍N−k+1₎] (ou só um dos limites no unilateral). Como B é
    discreta, a confiança obtida é ≥ 1 − α. Se nenhum k servir (N pequeno), o limite fica
    infinito.
    """
    x = np.sort(np.asarray(valores, dtype=float))
    n = len(x)
    cauda = alfa / 2 if alternativa == "two-sided" else alfa
    k = 0
    for j in range(1, n // 2 + 2):
        if stats.binom.cdf(j - 1, n, 0.5) <= cauda:
            k = j
    if k == 0:
        return -math.inf, math.inf, 1.0
    perda = float(stats.binom.cdf(k - 1, n, 0.5))
    if alternativa == "greater":
        return float(x[k - 1]), math.inf, 1 - perda
    if alternativa == "less":
        return -math.inf, float(x[n - k]), 1 - perda
    return float(x[k - 1]), float(x[n - k]), 1 - 2 * perda


class TesteSinal(TesteBase):
    """Teste do sinal para a mediana (H₀: M = M₀).

    Modos ("Tipo de teste"): Uma amostra (uma coluna numérica e a mediana hipotética M₀) ou
    Pareado (duas colunas numéricas na mesma linha; testa a mediana das diferenças
    d = medida 1 − medida 2 contra M₀, em geral 0). Linhas com valor ausente são descartadas
    (aviso). Empates com M₀ (diferença zero) são descartados e reduzem o n, com aviso (padrão de
    livro-texto). Sob H₀ o número de sinais + segue Bin(n, 1/2): p exato via
    `scipy.stats.binomtest`. IC exato da mediana por estatísticas de ordem (todas as
    observações válidas), com a confiança efetivamente obtida. Sem card. Decisão: p ≤ α.
    """

    id = "teste_sinal"
    nome = "Teste do sinal"
    grupo = "Não paramétricos"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec(
                "modo", "Tipo de teste", "opcao", padrao=UMA_AMOSTRA, opcoes=list(MODOS_SINAL)
            ),
            ParametroSpec("coluna1", "Variável (ou medida 1)", "coluna_numerica"),
            ParametroSpec("coluna2", "Medida 2 (pareado)", "coluna_numerica", obrigatorio=False),
            ParametroSpec("m0", "Mediana hipotética (M₀)", "numero", padrao=0),
            ParametroSpec(
                "alternativa",
                "Hipótese alternativa (H₁)",
                "opcao",
                padrao=ALTERNATIVA_PADRAO_SINAL,
                opcoes=list(ALTERNATIVAS_SINAL),
            ),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    @staticmethod
    def valores(df: pd.DataFrame, params: dict) -> tuple[np.ndarray, int]:
        """(valores analisados — x ou as diferenças d —, linhas descartadas por ausência)."""
        c1 = params["coluna1"]
        if params.get("modo", UMA_AMOSTRA) == PAREADO:
            completos = df[[c1, params["coluna2"]]].dropna().astype(float)
            d = (completos[c1] - completos[params["coluna2"]]).to_numpy()
            return d, len(df) - len(completos)
        x = df[c1].dropna().astype(float).to_numpy()
        return x, len(df) - len(x)

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        modo = params.get("modo", UMA_AMOSTRA)
        erros = erro_opcao(modo, MODOS_SINAL, "Escolha o tipo de teste.")
        c1, c2 = params.get("coluna1"), params.get("coluna2")
        rotulo1 = "a medida 1" if modo == PAREADO else "a variável"
        erros += erros_coluna(df, c1, rotulo1)
        if modo == PAREADO:
            erros += erros_coluna(df, c2, "a medida 2")
            if not erros and c1 == c2:
                erros.append("As duas medidas devem ser colunas diferentes.")
        erros += erro_numero(
            params.get("m0"), "Informe um número válido para a mediana hipotética (M₀)."
        )
        if not erros:
            valores, _ = self.valores(df, params)
            if len(valores) == 0:
                erros.append("Não há observações válidas para o teste.")
            elif not (valores != converter_numero(params["m0"])).any():
                erros.append(
                    "Todas as observações são iguais a M₀: não há sinais + nem − e o teste não "
                    "pode ser calculado."
                )
        erros += erro_opcao(
            params.get("alternativa", ALTERNATIVA_PADRAO_SINAL),
            ALTERNATIVAS_SINAL,
            "Escolha uma hipótese alternativa válida.",
        )
        erros += erro_alfa(params.get("alfa", 0.05))
        return erros

    # ---------------- Execução ----------------
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)

        modo = params.get("modo", UMA_AMOSTRA)
        pareado = modo == PAREADO
        c1, c2 = params["coluna1"], params.get("coluna2")
        m0 = converter_numero(params["m0"])
        alfa = float(params.get("alfa", 0.05))
        alternativa = ALTERNATIVAS_SINAL[params.get("alternativa", ALTERNATIVA_PADRAO_SINAL)]

        valores, descartadas = self.valores(df, params)
        positivos = int((valores > m0).sum())
        negativos = int((valores < m0).sum())
        empates = int((valores == m0).sum())
        n = positivos + negativos
        p_valor = float(stats.binomtest(positivos, n, 0.5, alternative=alternativa).pvalue)
        mediana = float(np.median(valores))
        ic_inf, ic_sup, confianca_obtida = ic_mediana_exato(valores, alfa, alternativa)

        estatisticas = {
            "n_validos": float(len(valores)),
            "n": float(n),
            "positivos": float(positivos),
            "negativos": float(negativos),
            "empates": float(empates),
            "n_descartadas": float(descartadas),
            "m0": m0,
            "mediana": mediana,
            "p_valor": p_valor,
            "ic_inferior": ic_inf,
            "ic_superior": ic_sup,
            "confianca_obtida": confianca_obtida,
        }

        m0_txt = _numero_curto(m0)
        alvo = f"a mediana das diferenças '{c1}' − '{c2}'" if pareado else f"a mediana de '{c1}'"
        simbolo, texto = _SIMBOLO[alternativa], _TEXTO[alternativa]
        interpretacao = interpretar(
            p_valor,
            alfa,
            h0=f"M = {m0_txt}",
            h1=f"M {simbolo} {m0_txt}",
            conclusao_rejeita=f"Há evidência estatística de que {alvo} é {texto} {m0_txt}.",
            conclusao_nao_rejeita=(
                f"Não há evidência suficiente de que {alvo} seja {texto} {m0_txt}."
            ),
        )

        avisos = []
        if descartadas:
            onde = f"'{c1}' ou '{c2}'" if pareado else f"'{c1}'"
            avisos.append(f"{descartadas} linha(s) com valor ausente em {onde} foram descartadas.")
        if empates:
            avisos.append(
                f"{empates} observação(ões) igual(is) a M₀ = {m0_txt} (sem sinal) foram "
                f"descartadas; o teste usa n = {n}."
            )
        if math.isinf(ic_inf) and math.isinf(ic_sup):
            avisos.append(
                "Poucas observações para um IC exato da mediana neste nível de confiança."
            )

        nivel = f"{(1 - alfa) * 100:.0f}%"
        tipo_ic = "" if alternativa == "two-sided" else " (unilateral)"
        nome_mediana = "Mediana das diferenças" if pareado else "Mediana amostral"
        resumo = [
            ("Tipo de teste", modo),
            ("Observações válidas", f"{len(valores)}"),
            ("Sinais + (acima de M₀)", f"{positivos}"),
            ("Sinais − (abaixo de M₀)", f"{negativos}"),
            ("Empates com M₀ (descartados)", f"{empates}"),
            ("n usado no teste (+ e −)", f"{n}"),
            ("Mediana hipotética (M₀)", formatar_numero(m0)),
            (nome_mediana, formatar_numero(mediana)),
            ("p-valor exato (binomial)", formatar_p_valor(p_valor)),
            (
                f"IC {nivel} exato para a mediana{tipo_ic}",
                f"[{formatar_numero(ic_inf)}; {formatar_numero(ic_sup)}]",
            ),
            ("Confiança obtida do IC", f"{formatar_numero(confianca_obtida * 100, 1)}%"),
        ]
        rotulo = f"{c1} − {c2}" if pareado else c1
        titulo = f"Diferenças '{c1}' − '{c2}'" if pareado else f"Distribuição de '{c1}'"

        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=interpretacao,
            tabelas={"Resumo": pd.DataFrame(resumo, columns=["Medida", "Valor"])},
            figuras=[
                histograma(
                    valores,
                    titulo,
                    rotulo,
                    [
                        (f"{nome_mediana} ({formatar_numero(mediana, 2)})", mediana, "destaque"),
                        (f"Mediana hipotética (M₀ = {m0_txt})", m0, "tracejado"),
                    ],
                )
            ],
            avisos=avisos,
        )


# ---------------------------------------------------------------------------
# Wilcoxon (postos sinalizados)
# ---------------------------------------------------------------------------
LIMITE_EXATO_WILCOXON = 50  # n até este valor, sem empates: p e IC exatos


def distribuicao_postos_sinalizados(n: int) -> np.ndarray:
    """Contagem de subconjuntos de {1..n} por soma: distribuição exata de W⁺ sob H₀ (× 2ⁿ)."""
    contagens = np.zeros(n * (n + 1) // 2 + 1, dtype=float)
    contagens[0] = 1
    for posto in range(1, n + 1):
        contagens[posto:] = contagens[posto:] + contagens[: len(contagens) - posto].copy()
    return contagens


def _quantil_postos(p: float, n: int) -> int:
    """Menor w com P(W⁺ ≤ w) ≥ p (como `qsignrank` do R)."""
    acumulada = np.cumsum(distribuicao_postos_sinalizados(n)) / 2**n
    return int(np.searchsorted(acumulada, p - 1e-12))


def hodges_lehmann(d: np.ndarray, alfa: float, alternativa: str, exato: bool) -> tuple:
    """(pseudomediana, IC inferior, IC superior) sobre as médias de Walsh de `d` (já sem zeros).

    Exato: quantis da distribuição de W⁺ (método do `wilcox.test` do R). Aproximado: posto
    k = n(n+1)/4 − z·√(n(n+1)(2n+1)/24) para escolher as estatísticas de ordem.
    """
    n = len(d)
    i, j = np.triu_indices(n)
    walsh = np.sort((d[i] + d[j]) / 2)
    estimativa = float(np.median(walsh))
    m = len(walsh)  # = n(n+1)/2
    cauda = alfa / 2 if alternativa == "two-sided" else alfa
    if exato:
        qu = max(_quantil_postos(cauda, n), 1)
    else:
        z = float(stats.norm.ppf(1 - cauda))
        qu = max(int(np.floor(n * (n + 1) / 4 - z * math.sqrt(n * (n + 1) * (2 * n + 1) / 24))), 1)
    qu = min(qu, m)
    baixo, alto = float(walsh[qu - 1]), float(walsh[m - qu])
    if alternativa == "greater":
        return estimativa, baixo, math.inf
    if alternativa == "less":
        return estimativa, -math.inf, alto
    return estimativa, baixo, alto


class TesteWilcoxon(TesteBase):
    """Teste de Wilcoxon dos postos sinalizados (H₀: mediana = M₀, distribuição simétrica).

    Modos ("Tipo de teste"): Uma amostra (coluna + M₀) ou Pareado (diferenças
    d = medida 1 − medida 2 contra M₀, padrão 0); linhas incompletas descartadas (aviso).
    Diferenças nulas descartadas (`zero_method="wilcox"`, aviso); empates em |d| recebem posto
    médio. p-valor via `scipy.stats.wilcoxon`, exato quando n ≤ 50 e não há empates, senão
    aproximação normal com correção de empates e sem correção de continuidade (o Resumo informa
    o método). W⁺/W⁻ = somas dos postos positivos/negativos. Pseudomediana de Hodges-Lehmann
    (mediana das médias de Walsh, sem as diferenças nulas, + M₀) com IC — exato pelos quantis
    de W⁺ como o `wilcox.test` do R, ou pela aproximação normal. Efeito r = z/√n, com z da
    aproximação normal. Sem card. Decisão: p ≤ α.
    """

    id = "wilcoxon"
    nome = "Wilcoxon"
    grupo = "Não paramétricos"

    def parametros(self) -> list[ParametroSpec]:
        return TesteSinal().parametros()

    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        return TesteSinal().validar(df, params)

    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)

        modo = params.get("modo", UMA_AMOSTRA)
        pareado = modo == PAREADO
        c1, c2 = params["coluna1"], params.get("coluna2")
        m0 = converter_numero(params["m0"])
        alfa = float(params.get("alfa", 0.05))
        alternativa = ALTERNATIVAS_SINAL[params.get("alternativa", ALTERNATIVA_PADRAO_SINAL)]

        valores, descartadas = TesteSinal.valores(df, params)
        diferencas = valores - m0
        nao_nulas = diferencas[diferencas != 0]
        zeros = len(diferencas) - len(nao_nulas)
        n = len(nao_nulas)
        postos = stats.rankdata(np.abs(nao_nulas))
        w_mais = float(postos[nao_nulas > 0].sum())
        w_menos = float(postos[nao_nulas < 0].sum())
        tem_empates = len(np.unique(np.abs(nao_nulas))) < n
        exato = n <= LIMITE_EXATO_WILCOXON and not tem_empates
        metodo = "exact" if exato else "approx"

        p_valor = float(
            stats.wilcoxon(
                nao_nulas, zero_method="wilcox", alternative=alternativa, method=metodo
            ).pvalue
        )
        _, contagem_empates = np.unique(postos, return_counts=True)
        variancia = n * (n + 1) * (2 * n + 1) / 24 - (
            (contagem_empates**3 - contagem_empates).sum() / 48
        )
        z = (w_mais - n * (n + 1) / 4) / math.sqrt(variancia) if variancia > 0 else 0.0
        r_efeito = z / math.sqrt(n)
        hl, ic_inf, ic_sup = hodges_lehmann(nao_nulas, alfa, alternativa, exato)
        pseudomediana = hl + m0
        mediana = float(np.median(valores))

        estatisticas = {
            "n_validos": float(len(valores)),
            "n": float(n),
            "zeros": float(zeros),
            "n_descartadas": float(descartadas),
            "w_mais": w_mais,
            "w_menos": w_menos,
            "usou_exato": float(exato),
            "p_valor": p_valor,
            "z": z,
            "r": r_efeito,
            "m0": m0,
            "mediana": mediana,
            "pseudomediana": pseudomediana,
            "ic_inferior": ic_inf + m0,
            "ic_superior": ic_sup + m0,
        }

        m0_txt = _numero_curto(m0)
        alvo = f"a mediana das diferenças '{c1}' − '{c2}'" if pareado else f"a mediana de '{c1}'"
        simbolo, texto = _SIMBOLO[alternativa], _TEXTO[alternativa]
        interpretacao = interpretar(
            p_valor,
            alfa,
            h0=f"M = {m0_txt}",
            h1=f"M {simbolo} {m0_txt}",
            conclusao_rejeita=f"Há evidência estatística de que {alvo} é {texto} {m0_txt}.",
            conclusao_nao_rejeita=(
                f"Não há evidência suficiente de que {alvo} seja {texto} {m0_txt}."
            ),
        )

        avisos = []
        if descartadas:
            onde = f"'{c1}' ou '{c2}'" if pareado else f"'{c1}'"
            avisos.append(f"{descartadas} linha(s) com valor ausente em {onde} foram descartadas.")
        if zeros:
            avisos.append(
                f"{zeros} diferença(s) nula(s) (valor igual a M₀ = {m0_txt}) foram descartadas; "
                f"o teste usa n = {n}."
            )
        if tem_empates:
            avisos.append(
                "Há empates nos valores absolutos das diferenças: postos médios e p-valor pela "
                "aproximação normal com correção de empates."
            )
        avisos.append(
            "O Wilcoxon supõe distribuição das diferenças aproximadamente simétrica em torno da "
            "mediana; se houver forte assimetria, prefira o Teste do sinal."
        )

        nivel = f"{(1 - alfa) * 100:.0f}%"
        tipo_ic = "" if alternativa == "two-sided" else " (unilateral)"
        nome_mediana = "Mediana das diferenças" if pareado else "Mediana amostral"
        metodo_txt = (
            "Exato" if exato else "Aproximação normal (correção de empates, sem continuidade)"
        )
        resumo = [
            ("Tipo de teste", modo),
            ("Observações válidas", f"{len(valores)}"),
            ("Diferenças nulas (descartadas)", f"{zeros}"),
            ("n usado no teste", f"{n}"),
            ("W⁺ (soma dos postos positivos)", formatar_numero(w_mais, 1)),
            ("W⁻ (soma dos postos negativos)", formatar_numero(w_menos, 1)),
            ("Método do p-valor", metodo_txt),
            ("p-valor", formatar_p_valor(p_valor)),
            ("Mediana hipotética (M₀)", formatar_numero(m0)),
            (nome_mediana, formatar_numero(mediana)),
            ("Pseudomediana (Hodges-Lehmann)", formatar_numero(pseudomediana)),
            (
                f"IC {nivel} para a pseudomediana{tipo_ic}",
                f"[{formatar_numero(ic_inf + m0)}; {formatar_numero(ic_sup + m0)}]",
            ),
            ("Tamanho de efeito r = z/√n", formatar_numero(r_efeito)),
        ]
        rotulo = f"{c1} − {c2}" if pareado else c1
        titulo = f"Diferenças '{c1}' − '{c2}'" if pareado else f"Distribuição de '{c1}'"

        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=interpretacao,
            tabelas={"Resumo": pd.DataFrame(resumo, columns=["Medida", "Valor"])},
            figuras=[
                histograma(
                    valores,
                    titulo,
                    rotulo,
                    [
                        (
                            f"Pseudomediana ({formatar_numero(pseudomediana, 2)})",
                            pseudomediana,
                            "destaque",
                        ),
                        (f"Mediana hipotética (M₀ = {m0_txt})", m0, "tracejado"),
                    ],
                )
            ],
            avisos=avisos,
        )


# ---------------------------------------------------------------------------
# Mann-Whitney U
# ---------------------------------------------------------------------------
ALTERNATIVAS_MW = {"G₁ ≠ G₂": "two-sided", "G₁ > G₂": "greater", "G₁ < G₂": "less"}
ALTERNATIVA_PADRAO_MW = "G₁ ≠ G₂"
LIMITE_EXATO_MW = 8  # como o method="auto" do scipy: exato se min(n₁, n₂) ≤ 8 e sem empates


def distribuicao_u(n1: int, n2: int) -> np.ndarray:
    """Contagens de U = 0..n₁n₂ sob H₀: coeficientes de ∏ₖ (1 − q^(n₂+k)) / (1 − qᵏ), k = 1..n₁."""
    poli = np.zeros(n1 * n2 + 1)
    poli[0] = 1.0
    for k in range(1, n1 + 1):
        multiplicado = poli.copy()
        multiplicado[n2 + k :] -= poli[: len(poli) - (n2 + k)]
        for i in range(k, len(multiplicado)):  # divide por (1 − qᵏ): soma acumulada de passo k
            multiplicado[i] += multiplicado[i - k]
        poli = multiplicado
    return poli


def _quantil_u(p: float, n1: int, n2: int) -> int:
    """Menor u com P(U ≤ u) ≥ p (como `qwilcox` do R)."""
    contagens = distribuicao_u(n1, n2)
    acumulada = np.cumsum(contagens) / contagens.sum()
    return int(np.searchsorted(acumulada, p - 1e-12))


def deslocamento_hodges_lehmann(
    x1: np.ndarray, x2: np.ndarray, alfa: float, alternativa: str, exato: bool
) -> tuple[float, float, float]:
    """(estimativa, IC inferior, IC superior) do deslocamento grupo 1 − grupo 2.

    Estimativa: mediana das n₁n₂ diferenças x₁ᵢ − x₂ⱼ. IC exato pelos quantis de U (método do
    `wilcox.test` do R) ou pela aproximação normal k = n₁n₂/2 − z·√(n₁n₂(N+1)/12).
    """
    n1, n2 = len(x1), len(x2)
    dif = np.sort((x1[:, None] - x2[None, :]).ravel())
    m = len(dif)
    estimativa = float(np.median(dif))
    cauda = alfa / 2 if alternativa == "two-sided" else alfa
    if exato:
        qu = max(_quantil_u(cauda, n1, n2), 1)
    else:
        z = float(stats.norm.ppf(1 - cauda))
        qu = max(int(np.floor(n1 * n2 / 2 - z * math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12))), 1)
    qu = min(qu, m)
    baixo, alto = float(dif[qu - 1]), float(dif[m - qu])
    if alternativa == "greater":
        return estimativa, baixo, math.inf
    if alternativa == "less":
        return estimativa, -math.inf, alto
    return estimativa, baixo, alto


class TesteMannWhitney(TesteBase):
    """Teste de Mann-Whitney U (Wilcoxon da soma dos postos) para dois grupos independentes.

    H₀: as distribuições da variável nos dois grupos são iguais; "G₁ > G₂": a variável tende a
    ser maior no grupo 1. Entrada: coluna numérica + coluna de grupo com exatamente 2 níveis
    (grupo 1 = primeiro nível em ordem crescente); linhas incompletas descartadas (aviso).
    Wrapper de `scipy.stats.mannwhitneyu`: exato quando min(n₁, n₂) ≤ 8 e não há empates (regra
    do method="auto"), senão aproximação normal com correção de empates e de continuidade; o
    Resumo informa o método. U₁ = R₁ − n₁(n₁+1)/2 e U₂ = n₁n₂ − U₁. Efeitos: probabilidade de
    superioridade U₁/(n₁n₂) = P(X₁ > X₂) + ½·P(X₁ = X₂) e r = z/√N (z sem continuidade).
    Deslocamento de Hodges-Lehmann com IC (exato como o `wilcox.test` do R, ou aproximado). Sem
    card. Decisão: p ≤ α.
    """

    id = "mann_whitney"
    nome = "Mann-Whitney U"
    grupo = "Não paramétricos"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec("coluna", "Variável", "coluna_numerica"),
            ParametroSpec("grupo", "Grupo (2 níveis)", "coluna_binaria"),
            ParametroSpec(
                "alternativa",
                "Hipótese alternativa (H₁)",
                "opcao",
                padrao=ALTERNATIVA_PADRAO_MW,
                opcoes=list(ALTERNATIVAS_MW),
            ),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        coluna, grupo = params.get("coluna"), params.get("grupo")
        erros = erros_coluna(df, coluna) + erros_coluna(df, grupo, "o grupo", numerica=False)
        if not erros and coluna == grupo:
            erros.append("A variável e o grupo devem ser colunas diferentes.")
        if not erros:
            niveis, amostras, _ = TesteT2Amostras.separar(df, coluna, grupo)
            if len(niveis) != 2:
                erros.append(
                    f"A coluna de grupo '{grupo}' deve ter exatamente 2 níveis com dados válidos "
                    f"(tem {len(niveis)})."
                )
            elif np.ptp(np.concatenate(amostras)) == 0:
                erros.append(
                    f"Todos os valores de '{coluna}' são iguais nos dois grupos: não há ordem "
                    "para comparar."
                )
        erros += erro_opcao(
            params.get("alternativa", ALTERNATIVA_PADRAO_MW),
            ALTERNATIVAS_MW,
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
        alternativa = ALTERNATIVAS_MW[params.get("alternativa", ALTERNATIVA_PADRAO_MW)]

        niveis, (x1, x2), descartadas = TesteT2Amostras.separar(df, coluna, grupo)
        g1, g2 = (rotulo_nivel(n) for n in niveis)
        n1, n2 = len(x1), len(x2)
        total = n1 + n2

        postos = stats.rankdata(np.concatenate([x1, x2]))
        r1, r2 = float(postos[:n1].sum()), float(postos[n1:].sum())
        u1 = r1 - n1 * (n1 + 1) / 2
        u2 = n1 * n2 - u1
        _, contagem_empates = np.unique(postos, return_counts=True)
        tem_empates = bool((contagem_empates > 1).any())
        exato = min(n1, n2) <= LIMITE_EXATO_MW and not tem_empates

        p_valor = float(
            stats.mannwhitneyu(
                x1, x2, alternative=alternativa, method="exact" if exato else "asymptotic"
            ).pvalue
        )
        correcao = (contagem_empates**3 - contagem_empates).sum() / (total * (total - 1))
        variancia = n1 * n2 / 12 * ((total + 1) - correcao)
        z = (u1 - n1 * n2 / 2) / math.sqrt(variancia)
        hl, ic_inf, ic_sup = deslocamento_hodges_lehmann(x1, x2, alfa, alternativa, exato)

        estatisticas = {
            "n1": float(n1),
            "n2": float(n2),
            "n_descartadas": float(descartadas),
            "soma_postos1": r1,
            "soma_postos2": r2,
            "u1": u1,
            "u2": u2,
            "usou_exato": float(exato),
            "p_valor": p_valor,
            "z": z,
            "r": z / math.sqrt(total),
            "prob_superioridade": u1 / (n1 * n2),
            "mediana1": float(np.median(x1)),
            "mediana2": float(np.median(x2)),
            "deslocamento_hl": hl,
            "ic_inferior": ic_inf,
            "ic_superior": ic_sup,
        }

        simbolo = _SIMBOLO[alternativa]
        if alternativa == "two-sided":
            rejeita = f"Há evidência estatística de que '{coluna}' difere entre '{g1}' e '{g2}'."
            nao = f"Não há evidência suficiente de que '{coluna}' difira entre '{g1}' e '{g2}'."
        else:
            maior, menor = (g1, g2) if alternativa == "greater" else (g2, g1)
            rejeita = (
                f"Há evidência estatística de que '{coluna}' tende a ser maior no grupo "
                f"'{maior}' do que no grupo '{menor}'."
            )
            nao = (
                f"Não há evidência suficiente de que '{coluna}' tenda a ser maior no grupo "
                f"'{maior}' do que no grupo '{menor}'."
            )
        interpretacao = interpretar(
            p_valor,
            alfa,
            h0=f"a distribuição de '{coluna}' é a mesma em '{g1}' e '{g2}'",
            h1=f"G₁ {simbolo} G₂",
            conclusao_rejeita=rejeita,
            conclusao_nao_rejeita=nao,
        )

        avisos = []
        if descartadas:
            avisos.append(f"{descartadas} linha(s) com valor ou grupo ausente foram descartadas.")
        if tem_empates:
            avisos.append(
                "Há empates entre os valores: postos médios e p-valor pela aproximação normal com "
                "correção de empates."
            )

        nivel = f"{(1 - alfa) * 100:.0f}%"
        tipo_ic = "" if alternativa == "two-sided" else " (unilateral)"
        metodo_txt = "Exato" if exato else "Aproximação normal (empates e continuidade)"
        resumo = [
            ("Grupo 1", f"'{g1}' (n = {n1})"),
            ("Grupo 2", f"'{g2}' (n = {n2})"),
            ("Soma dos postos do grupo 1 (R₁)", formatar_numero(r1, 1)),
            ("Soma dos postos do grupo 2 (R₂)", formatar_numero(r2, 1)),
            ("U₁", formatar_numero(u1, 1)),
            ("U₂", formatar_numero(u2, 1)),
            ("Método do p-valor", metodo_txt),
            ("p-valor", formatar_p_valor(p_valor)),
            ("Mediana do grupo 1", formatar_numero(estatisticas["mediana1"])),
            ("Mediana do grupo 2", formatar_numero(estatisticas["mediana2"])),
            ("Deslocamento de Hodges-Lehmann (G₁ − G₂)", formatar_numero(hl)),
            (
                f"IC {nivel} para o deslocamento{tipo_ic}",
                f"[{formatar_numero(ic_inf)}; {formatar_numero(ic_sup)}]",
            ),
            (
                "Probabilidade de superioridade U₁/(n₁n₂)",
                formatar_numero(estatisticas["prob_superioridade"]),
            ),
            ("Tamanho de efeito r = z/√N", formatar_numero(estatisticas["r"])),
        ]

        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=interpretacao,
            tabelas={"Resumo": pd.DataFrame(resumo, columns=["Medida", "Valor"])},
            figuras=[boxplot([(g1, x1), (g2, x2)], f"'{coluna}' por '{grupo}'", coluna)],
            avisos=avisos,
        )


# ---------------------------------------------------------------------------
# Kruskal-Wallis
# ---------------------------------------------------------------------------
MAX_GRUPOS_KW = 20
MIN_POR_GRUPO_APROXIMACAO = 5


def ajuste_holm(p_valores: list[float]) -> list[float]:
    """p-valores ajustados por Holm (passo a passo descendente), na ordem original."""
    m = len(p_valores)
    ordem = sorted(range(m), key=lambda i: p_valores[i])
    ajustados = [0.0] * m
    maximo = 0.0
    for posicao, i in enumerate(ordem):
        maximo = max(maximo, min(1.0, (m - posicao) * p_valores[i]))
        ajustados[i] = maximo
    return ajustados


def dunn(
    postos_medios: list[float], tamanhos: list[int], n_total: int, soma_empates: float
) -> list[tuple[int, int, float, float, float]]:
    """Pós-teste de Dunn: (i, j, diferença de postos médios, z, p bilateral) para cada par.

    z = (R̄ᵢ − R̄ⱼ) / √{[N(N+1)/12 − Σ(t³ − t)/(12(N − 1))]·(1/nᵢ + 1/nⱼ)}.
    """
    base = n_total * (n_total + 1) / 12 - soma_empates / (12 * (n_total - 1))
    pares = []
    for i in range(len(tamanhos)):
        for j in range(i + 1, len(tamanhos)):
            diferenca = postos_medios[i] - postos_medios[j]
            z = diferenca / math.sqrt(base * (1 / tamanhos[i] + 1 / tamanhos[j]))
            pares.append((i, j, diferenca, z, float(2 * stats.norm.sf(abs(z)))))
    return pares


class TesteKruskalWallis(TesteBase):
    """Teste de Kruskal-Wallis para k ≥ 2 grupos independentes (H₀: mesma distribuição).

    Entrada: coluna numérica + coluna de grupo com 2 a 20 níveis e ao menos 2 observações por
    grupo; linhas incompletas descartadas (aviso). H com correção de empates e p pela
    aproximação qui-quadrado com k − 1 gl (`scipy.stats.kruskal`); aviso quando algum grupo
    tem menos de 5 observações. Efeito: ε² = H/(N − 1). Pós-teste de Dunn opcional (desligado
    por padrão), exibido só quando H₀ é rejeitada, com p ajustado por Holm. Sem card. Decisão:
    p ≤ α.
    """

    id = "kruskal_wallis"
    nome = "Kruskal-Wallis"
    grupo = "Não paramétricos"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec("coluna", "Variável", "coluna_numerica"),
            ParametroSpec("grupo", "Grupo (2 ou mais níveis)", "coluna_categorica"),
            ParametroSpec("dunn", "Comparações múltiplas (Dunn)", "booleano", padrao=False),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        coluna, grupo = params.get("coluna"), params.get("grupo")
        erros = erros_coluna(df, coluna) + erros_coluna(df, grupo, "o grupo", numerica=False)
        if not erros and coluna == grupo:
            erros.append("A variável e o grupo devem ser colunas diferentes.")
        if not erros:
            niveis, amostras, _ = TesteT2Amostras.separar(df, coluna, grupo)
            k = len(niveis)
            if not 2 <= k <= MAX_GRUPOS_KW:
                erros.append(
                    f"A coluna de grupo '{grupo}' deve ter de 2 a {MAX_GRUPOS_KW} níveis com "
                    f"dados válidos (tem {k})."
                )
            else:
                pequenos = [
                    f"'{rotulo_nivel(nivel)}' ({len(x)})"
                    for nivel, x in zip(niveis, amostras, strict=True)
                    if len(x) < 2
                ]
                if pequenos:
                    erros.append(
                        "Cada grupo precisa de ao menos 2 observações válidas; grupos com menos: "
                        + ", ".join(pequenos)
                        + "."
                    )
                elif np.ptp(np.concatenate(amostras)) == 0:
                    erros.append(
                        f"Todos os valores de '{coluna}' são iguais: não há ordem para comparar."
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
        pedir_dunn = bool(params.get("dunn", False))

        niveis, amostras, descartadas = TesteT2Amostras.separar(df, coluna, grupo)
        rotulos = [rotulo_nivel(n) for n in niveis]
        tamanhos = [len(x) for x in amostras]
        k, total = len(amostras), sum(tamanhos)

        resultado = stats.kruskal(*amostras)
        h, p_valor = float(resultado.statistic), float(resultado.pvalue)
        postos = stats.rankdata(np.concatenate(amostras))
        limites = np.cumsum([0, *tamanhos])
        postos_medios = [float(postos[a:b].mean()) for a, b in itertools.pairwise(limites)]
        _, contagem = np.unique(postos, return_counts=True)
        soma_empates = float((contagem**3 - contagem).sum())
        epsilon2 = h / (total - 1)

        estatisticas = {
            "k": float(k),
            "n": float(total),
            "n_descartadas": float(descartadas),
            "h": h,
            "gl": float(k - 1),
            "p_valor": p_valor,
            "epsilon2": epsilon2,
        }

        interpretacao = interpretar(
            p_valor,
            alfa,
            h0=f"a distribuição de '{coluna}' é a mesma nos {k} grupos de '{grupo}'",
            h1="ao menos um grupo difere dos demais",
            conclusao_rejeita=(
                f"Há evidência estatística de que '{coluna}' difere entre os grupos de '{grupo}'."
            ),
            conclusao_nao_rejeita=(
                f"Não há evidência suficiente de que '{coluna}' difira entre os grupos de "
                f"'{grupo}'."
            ),
        )
        rejeita = decidir(p_valor, alfa) == REJEITA_H0

        avisos = []
        if descartadas:
            avisos.append(f"{descartadas} linha(s) com valor ou grupo ausente foram descartadas.")
        pequenos = [
            f"'{r}' (n = {n})"
            for r, n in zip(rotulos, tamanhos, strict=True)
            if n < MIN_POR_GRUPO_APROXIMACAO
        ]
        if pequenos:
            avisos.append(
                f"Grupos com menos de {MIN_POR_GRUPO_APROXIMACAO} observações "
                f"({', '.join(pequenos)}): a aproximação qui-quadrado do p-valor é fraca."
            )
        if soma_empates:
            avisos.append(
                "Há empates entre os valores: H foi corrigido para empates (postos médios)."
            )

        tabela_grupos = pd.DataFrame(
            {
                "Grupo": rotulos,
                "n": tamanhos,
                "Mediana": [formatar_numero(float(np.median(x))) for x in amostras],
                "Posto médio": [formatar_numero(r, 2) for r in postos_medios],
            }
        )
        resumo = [
            ("Grupos (k)", f"{k}"),
            ("Observações (N)", f"{total}"),
            ("Estatística H (corrigida para empates)", formatar_numero(h)),
            ("Graus de liberdade", f"{k - 1}"),
            ("p-valor (qui-quadrado)", formatar_p_valor(p_valor)),
            ("Tamanho de efeito ε² = H/(N − 1)", formatar_numero(epsilon2)),
        ]
        tabelas = {
            "Resumo": pd.DataFrame(resumo, columns=["Medida", "Valor"]),
            "Grupos": tabela_grupos,
        }

        if pedir_dunn and rejeita:
            pares = dunn(postos_medios, tamanhos, total, soma_empates)
            ajustados = ajuste_holm([p for *_, p in pares])
            estatisticas["comparacoes"] = float(len(pares))
            tabelas["Comparações múltiplas (Dunn, Holm)"] = pd.DataFrame(
                [
                    (
                        f"'{rotulos[i]}' × '{rotulos[j]}'",
                        formatar_numero(dif, 2),
                        formatar_numero(z, 3),
                        formatar_p_valor(p),
                        formatar_p_valor(p_aj),
                        "Sim" if p_aj <= alfa else "Não",
                    )
                    for (i, j, dif, z, p), p_aj in zip(pares, ajustados, strict=True)
                ],
                columns=[
                    "Comparação",
                    "Diferença de postos médios",
                    "z",
                    "p",
                    "p ajustado (Holm)",
                    "Diferem?",
                ],
            )
        elif pedir_dunn:
            avisos.append(
                "Comparações múltiplas (Dunn) não exibidas: H₀ não foi rejeitada, então não há "
                "diferença a localizar entre os grupos."
            )

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
        )


# ---------------------------------------------------------------------------
# Friedman
# ---------------------------------------------------------------------------
MIN_BLOCOS_APROXIMACAO = 10


class TesteFriedman(TesteBase):
    """Teste de Friedman para k ≥ 3 medidas repetidas (H₀: as k medidas têm a mesma distribuição).

    Entrada: 3 ou mais colunas numéricas na mesma linha; cada linha é um bloco (ex.: um aluno em
    k provas). Linhas com valor ausente em alguma coluna escolhida são descartadas (aviso).
    Postos dentro de cada linha (postos médios nos empates); estatística com correção de empates
    e p pela qui-quadrado com k − 1 gl (`scipy.stats.friedmanchisquare`); aviso com menos de 10
    blocos. Efeito: W de Kendall = Q/(n(k − 1)). Comparações múltiplas opcionais (desligadas),
    exibidas só quando H₀ é rejeitada: Wilcoxon pareado entre cada par de colunas (mesmas
    opções do teste de Wilcoxon) com p ajustado por Holm. Sem card. Decisão: p ≤ α.
    """

    id = "friedman"
    nome = "Friedman"
    grupo = "Não paramétricos"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec("colunas", "Medidas repetidas (3 ou mais colunas)", "multi_coluna"),
            ParametroSpec(
                "comparacoes", "Comparações múltiplas (Wilcoxon + Holm)", "booleano", padrao=False
            ),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    @staticmethod
    def blocos(df: pd.DataFrame, colunas: list[str]) -> tuple[np.ndarray, int]:
        """(matriz n × k com as linhas completas, linhas descartadas)."""
        completos = df[colunas].dropna().astype(float)
        return completos.to_numpy(), len(df) - len(completos)

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        colunas = list(params.get("colunas") or [])
        erros: list[str] = []
        if len(colunas) < 3:
            erros.append(f"Selecione ao menos 3 colunas (há {len(colunas)}).")
        elif len(set(colunas)) != len(colunas):
            erros.append("As colunas escolhidas devem ser diferentes.")
        else:
            for coluna in colunas:
                erros += erros_coluna(df, coluna)
        if not erros:
            dados, _ = self.blocos(df, colunas)
            if len(dados) < 2:
                erros.append(
                    f"São necessárias ao menos 2 linhas completas nas colunas escolhidas "
                    f"(há {len(dados)})."
                )
            elif (np.ptp(dados, axis=1) == 0).all():
                erros.append(
                    "Em todas as linhas as medidas são iguais entre si: não há ordem para comparar."
                )
        erros += erro_alfa(params.get("alfa", 0.05))
        return erros

    # ---------------- Execução ----------------
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)

        colunas = list(params["colunas"])
        alfa = float(params.get("alfa", 0.05))
        pedir_comparacoes = bool(params.get("comparacoes", False))

        dados, descartadas = self.blocos(df, colunas)
        n, k = dados.shape
        resultado = stats.friedmanchisquare(*dados.T)
        q, p_valor = float(resultado.statistic), float(resultado.pvalue)
        postos = np.apply_along_axis(stats.rankdata, 1, dados)
        postos_medios = postos.mean(axis=0)
        tem_empates = any(len(np.unique(linha)) < k for linha in dados)
        w_kendall = q / (n * (k - 1))

        estatisticas = {
            "k": float(k),
            "n": float(n),
            "n_descartadas": float(descartadas),
            "q": q,
            "gl": float(k - 1),
            "p_valor": p_valor,
            "w_kendall": w_kendall,
        }

        interpretacao = interpretar(
            p_valor,
            alfa,
            h0=f"as {k} medidas têm a mesma distribuição",
            h1="ao menos uma medida difere das demais",
            conclusao_rejeita=(
                "Há evidência estatística de diferença entre as medidas "
                + ", ".join(f"'{c}'" for c in colunas)
                + "."
            ),
            conclusao_nao_rejeita=(
                "Não há evidência suficiente de diferença entre as medidas "
                + ", ".join(f"'{c}'" for c in colunas)
                + "."
            ),
        )
        rejeita = decidir(p_valor, alfa) == REJEITA_H0

        avisos = []
        if descartadas:
            avisos.append(
                f"{descartadas} linha(s) com valor ausente em alguma das colunas foram descartadas."
            )
        if n < MIN_BLOCOS_APROXIMACAO:
            avisos.append(
                f"Poucas linhas completas (n = {n} < {MIN_BLOCOS_APROXIMACAO}): a aproximação "
                "qui-quadrado do p-valor é fraca."
            )
        if tem_empates:
            avisos.append(
                "Há empates dentro de linhas: postos médios e estatística corrigida para empates."
            )

        tabelas = {
            "Resumo": pd.DataFrame(
                [
                    ("Medidas (k)", f"{k}"),
                    ("Linhas completas (n)", f"{n}"),
                    ("Estatística de Friedman (corrigida para empates)", formatar_numero(q)),
                    ("Graus de liberdade", f"{k - 1}"),
                    ("p-valor (qui-quadrado)", formatar_p_valor(p_valor)),
                    ("W de Kendall", formatar_numero(w_kendall)),
                ],
                columns=["Medida", "Valor"],
            ),
            "Medidas": pd.DataFrame(
                {
                    "Coluna": colunas,
                    "Mediana": [formatar_numero(float(np.median(dados[:, j]))) for j in range(k)],
                    "Posto médio": [formatar_numero(float(r), 2) for r in postos_medios],
                }
            ),
        }

        if pedir_comparacoes and rejeita:
            pares = []
            for i, j in itertools.combinations(range(k), 2):
                d = dados[:, i] - dados[:, j]
                nao_nulas = d[d != 0]
                if len(nao_nulas) == 0:
                    p_par = 1.0
                else:
                    sem_empates = len(np.unique(np.abs(nao_nulas))) == len(nao_nulas)
                    metodo = (
                        "exact"
                        if len(nao_nulas) <= LIMITE_EXATO_WILCOXON and sem_empates
                        else "approx"
                    )
                    p_par = float(stats.wilcoxon(nao_nulas, method=metodo).pvalue)
                pares.append((i, j, float(np.median(d)), p_par))
            ajustados = ajuste_holm([p for *_, p in pares])
            estatisticas["comparacoes"] = float(len(pares))
            tabelas["Comparações múltiplas (Wilcoxon, Holm)"] = pd.DataFrame(
                [
                    (
                        f"'{colunas[i]}' × '{colunas[j]}'",
                        formatar_numero(mediana_dif, 2),
                        formatar_p_valor(p),
                        formatar_p_valor(p_aj),
                        "Sim" if p_aj <= alfa else "Não",
                    )
                    for (i, j, mediana_dif, p), p_aj in zip(pares, ajustados, strict=True)
                ],
                columns=[
                    "Comparação",
                    "Mediana das diferenças",
                    "p",
                    "p ajustado (Holm)",
                    "Diferem?",
                ],
            )
        elif pedir_comparacoes:
            avisos.append(
                "Comparações múltiplas não exibidas: H₀ não foi rejeitada, então não há diferença "
                "a localizar entre as medidas."
            )

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
                    [(c, dados[:, j]) for j, c in enumerate(colunas)],
                    "Medidas repetidas: " + ", ".join(f"'{c}'" for c in colunas),
                    "valor",
                )
            ],
            avisos=avisos,
        )
