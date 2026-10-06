"""Testes não paramétricos."""

import math

import numpy as np
import pandas as pd
from scipy import stats

from core.base import ErroValidacao, ParametroSpec, ResultadoTeste, TesteBase
from core.figuras import histograma
from core.interpretacao import decidir, formatar_numero, formatar_p_valor, interpretar
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
