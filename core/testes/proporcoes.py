"""Testes de proporções."""

import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_ztest,
)

from core.base import ComparacaoPValores, ErroValidacao, ParametroSpec, ResultadoTeste, TesteBase
from core.figuras import barras
from core.interpretacao import decidir, formatar_numero, formatar_p_valor, interpretar
from core.tipos import niveis_coluna, rotulo_nivel
from core.validacao import converter_numero, erro_alfa, erro_opcao, erros_coluna

ALTERNATIVAS_P = {"p ≠ p₀": "two-sided", "p > p₀": "greater", "p < p₀": "less"}
ALTERNATIVA_PADRAO_P = "p ≠ p₀"
_STATSMODELS = {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}
_SIMBOLO = {"two-sided": "≠", "greater": ">", "less": "<"}
_TEXTO = {"two-sided": "diferente de", "greater": "maior que", "less": "menor que"}
MINIMO_ESPERADO = 5  # regra prática para a aproximação normal: n·p₀ ≥ 5 e n·(1 − p₀) ≥ 5


def _texto_proporcao(p: float) -> str:
    return formatar_numero(p, 4).rstrip("0").rstrip(",")


def _percentual(p: float) -> str:
    return f"{formatar_numero(p * 100, 1)}%"


def ic_wilson(sucessos: int, n: int, alfa: float, alternativa: str) -> tuple[float, float]:
    """IC de Wilson (escore) para p, coerente com o teste Z que usa p₀ no erro padrão.

    Bilateral com confiança 1 − α; unilateral: o limite do IC bilateral de confiança 1 − 2α e o
    outro extremo em 0 ou 1.
    """
    if alternativa == "two-sided":
        baixo, alto = proportion_confint(sucessos, n, alpha=alfa, method="wilson")
        return float(baixo), float(alto)
    baixo, alto = proportion_confint(sucessos, n, alpha=2 * alfa, method="wilson")
    return (float(baixo), 1.0) if alternativa == "greater" else (0.0, float(alto))


class TesteZ1Prop(TesteBase):
    """Teste Z para uma proporção (H₀: p = p₀).

    Entrada: uma coluna binária (exatamente 2 valores distintos) e o valor que conta como
    "sucesso". Valores ausentes são ignorados (com aviso).
    Estatística z = (p̂ − p₀) / √(p₀(1 − p₀)/n): erro padrão calculado com p₀ (teste de escore,
    padrão de livro-texto, que mantém o nível α sob H₀), via
    `statsmodels.stats.proportion.proportions_ztest(..., prop_var=p₀)`.
    IC para p: Wilson (o intervalo que inverte esse mesmo teste). Tamanho de efeito: h de Cohen
    = 2·arcsen(√p̂) − 2·arcsen(√p₀).
    Equivalente no card: teste binomial exato (`scipy.stats.binomtest`; bilateral pelo método
    das probabilidades menores ou iguais à observada). Aviso quando n·p₀ ou n·(1 − p₀) < 5.
    Decisão: rejeita H₀ quando p ≤ α.
    """

    id = "teste_z_1prop"
    nome = "Teste Z (uma proporção)"
    grupo = "Proporções"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec("coluna", "Variável (binária)", "coluna_binaria"),
            ParametroSpec("sucesso", "Valor que conta como sucesso", "nivel", depende_de="coluna"),
            ParametroSpec("p0", "Proporção hipotética (p₀)", "numero", padrao=0.5),
            ParametroSpec(
                "alternativa",
                "Hipótese alternativa (H₁)",
                "opcao",
                padrao=ALTERNATIVA_PADRAO_P,
                opcoes=list(ALTERNATIVAS_P),
            ),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    def comparacao_inicial(self) -> ComparacaoPValores:
        return ComparacaoPValores(
            titulo_esquerda="Teste Z",
            titulo_direita="Binomial exato",
            hipoteses=list(ALTERNATIVAS_P),
            linhas=[(None, None)] * len(ALTERNATIVAS_P),
        )

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        coluna, sucesso = params.get("coluna"), params.get("sucesso")
        erros = erros_coluna(df, coluna, numerica=False)
        if not erros:
            niveis = niveis_coluna(df[coluna])
            if len(niveis) != 2:
                erros.append(
                    f"A coluna '{coluna}' deve ter exatamente 2 valores distintos "
                    f"(tem {len(niveis)})."
                )
            elif not sucesso:
                erros.append("Escolha o valor que conta como sucesso.")
            elif str(sucesso) not in niveis:
                erros.append(f"O valor '{sucesso}' não aparece na coluna '{coluna}'.")
        try:
            p0 = converter_numero(params.get("p0"))
            if not 0 < p0 < 1:
                raise ValueError
        except (TypeError, ValueError):
            erros.append("Informe a proporção hipotética (p₀) entre 0 e 1 (ex.: 0,5).")
        erros += erro_opcao(
            params.get("alternativa", ALTERNATIVA_PADRAO_P),
            ALTERNATIVAS_P,
            "Escolha uma hipótese alternativa válida.",
        )
        erros += erro_alfa(params.get("alfa", 0.05))
        return erros

    # ---------------- Execução ----------------
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)

        coluna, sucesso = params["coluna"], str(params["sucesso"])
        p0 = converter_numero(params["p0"])
        alfa = float(params.get("alfa", 0.05))
        alternativa = ALTERNATIVAS_P[params.get("alternativa", ALTERNATIVA_PADRAO_P)]

        serie = df[coluna]
        validos = serie.dropna().map(rotulo_nivel)
        n, n_ausentes = len(validos), int(serie.isna().sum())
        k = int((validos == sucesso).sum())
        fracasso = next(nivel for nivel in niveis_coluna(serie) if nivel != sucesso)
        p_hat = k / n

        testes_z = {
            alt: proportions_ztest(k, n, value=p0, alternative=_STATSMODELS[alt], prop_var=p0)
            for alt in _SIMBOLO
        }
        z, p_valor = (float(v) for v in testes_z[alternativa])
        p_binomial = {
            alt: float(stats.binomtest(k, n, p0, alternative=alt).pvalue) for alt in _SIMBOLO
        }
        ic = ic_wilson(k, n, alfa, alternativa)
        h_cohen = 2 * math.asin(math.sqrt(p_hat)) - 2 * math.asin(math.sqrt(p0))

        estatisticas = {
            "n": float(n),
            "n_ausentes": float(n_ausentes),
            "sucessos": float(k),
            "p_hat": p_hat,
            "p0": p0,
            "erro_padrao": math.sqrt(p0 * (1 - p0) / n),
            "z": z,
            "p_valor": p_valor,
            "ic_inferior": ic[0],
            "ic_superior": ic[1],
            "h_cohen": h_cohen,
            "p_binomial": p_binomial[alternativa],
        }

        p0_txt = _texto_proporcao(p0)
        simbolo, texto = _SIMBOLO[alternativa], _TEXTO[alternativa]
        de_quem = f"a proporção de '{sucesso}' em '{coluna}'"
        interpretacao = interpretar(
            p_valor,
            alfa,
            h0=f"p = {p0_txt}",
            h1=f"p {simbolo} {p0_txt}",
            conclusao_rejeita=f"Há evidência estatística de que {de_quem} é {texto} {p0_txt}.",
            conclusao_nao_rejeita=(
                f"Não há evidência suficiente de que {de_quem} seja {texto} {p0_txt}."
            ),
        )

        avisos = []
        if n_ausentes:
            avisos.append(f"{n_ausentes} valor(es) ausente(s) em '{coluna}' foram ignorados.")
        esperados = (n * p0, n * (1 - p0))
        if min(esperados) < MINIMO_ESPERADO:
            avisos.append(
                f"n·p₀ = {formatar_numero(esperados[0], 1)} e n·(1 − p₀) = "
                f"{formatar_numero(esperados[1], 1)}: com algum deles abaixo de "
                f"{MINIMO_ESPERADO}, a aproximação normal é fraca. Prefira o binomial exato "
                "do card."
            )

        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=interpretacao,
            tabelas={"Resumo": self._tabela_resumo(estatisticas, sucesso, alfa, alternativa)},
            figuras=[
                barras(
                    [(f"{sucesso} (sucesso)", p_hat), (fracasso, 1 - p_hat)],
                    f"Proporções em '{coluna}'",
                    "Proporção",
                    maximo=1.0,
                    referencias=[(f"Proporção hipotética (p₀ = {p0_txt})", p0, "tracejado")],
                    percentual=True,
                )
            ],
            avisos=avisos,
            comparacao=ComparacaoPValores(
                titulo_esquerda="Teste Z",
                titulo_direita="Binomial exato",
                hipoteses=list(ALTERNATIVAS_P),
                linhas=[(float(testes_z[a][1]), p_binomial[a]) for a in ALTERNATIVAS_P.values()],
            ),
        )

    @staticmethod
    def _tabela_resumo(
        e: dict[str, float], sucesso: str, alfa: float, alternativa: str
    ) -> pd.DataFrame:
        confianca = f"{(1 - alfa) * 100:.0f}%"
        tipo_ic = "" if alternativa == "two-sided" else " (unilateral)"
        ic = f"[{formatar_numero(e['ic_inferior'])}; {formatar_numero(e['ic_superior'])}]"
        linhas = [
            ("Observações (n)", f"{int(e['n'])}"),
            (f"Sucessos ('{sucesso}')", f"{int(e['sucessos'])}"),
            (
                "Proporção amostral (p̂)",
                f"{formatar_numero(e['p_hat'])} ({_percentual(e['p_hat'])})",
            ),
            ("Proporção hipotética (p₀)", formatar_numero(e["p0"])),
            ("Erro padrão (com p₀)", formatar_numero(e["erro_padrao"])),
            ("Estatística z", formatar_numero(e["z"])),
            ("p-valor (Z)", formatar_p_valor(e["p_valor"])),
            (f"IC {confianca} de Wilson para p{tipo_ic}", ic),
            ("h de Cohen", formatar_numero(e["h_cohen"])),
            ("p-valor (binomial exato)", formatar_p_valor(e["p_binomial"])),
        ]
        return pd.DataFrame(linhas, columns=["Medida", "Valor"])


# ---------------------------------------------------------------------------
# Teste Z (duas proporções)
# ---------------------------------------------------------------------------
ALTERNATIVAS_2P = {"p₁ ≠ p₂": "two-sided", "p₁ > p₂": "greater", "p₁ < p₂": "less"}
ALTERNATIVA_PADRAO_2P = "p₁ ≠ p₂"
_TEXTO_2 = {"two-sided": "diferente da", "greater": "maior que a", "less": "menor que a"}


def ic_diferenca_wald(
    k1: int, n1: int, k2: int, n2: int, alfa: float, alternativa: str
) -> tuple[float, float]:
    """IC de Wald (erro padrão não combinado) para p₁ − p₂; unilateral como em `ic_wilson`."""
    nivel = alfa if alternativa == "two-sided" else 2 * alfa
    baixo, alto = (
        float(v)
        for v in confint_proportions_2indep(
            k1, n1, k2, n2, method="wald", compare="diff", alpha=nivel
        )
    )
    if alternativa == "greater":
        return baixo, 1.0
    if alternativa == "less":
        return -1.0, alto
    return baixo, alto


class TesteZ2Prop(TesteBase):
    """Teste Z para duas proporções independentes (H₀: p₁ = p₂).

    Entrada: uma coluna binária (resposta), o valor que conta como "sucesso" e uma coluna de
    grupo com exatamente 2 níveis; grupo 1 = primeiro nível em ordem crescente. Linhas com
    resposta ou grupo ausente são descartadas (com aviso).
    Estatística z com a proporção combinada (pooled) no erro padrão, padrão de livro-texto sob
    H₀: p₁ = p₂ (`proportions_ztest` com duas amostras). IC da diferença p₁ − p₂: Wald com erro
    padrão não combinado (`confint_proportions_2indep(method="wald")`).
    Tamanhos de efeito: diferença de proporções, h de Cohen e razão de chances (odds ratio) da
    tabela 2×2 com IC de Woolf (log; unilateral quando H₁ é); indefinida com célula zero (aviso).
    Equivalente no card: teste exato de Fisher (`scipy.stats.fisher_exact`) na tabela
    [[sucessos₁, fracassos₁], [sucessos₂, fracassos₂]]; "maior" ⇔ OR > 1 ⇔ p₁ > p₂.
    Aviso quando alguma frequência esperada sob H₀ é menor que 5. Decisão: rejeita H₀ se p ≤ α.
    """

    id = "teste_z_2prop"
    nome = "Teste Z (duas proporções)"
    grupo = "Proporções"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec("coluna", "Resposta (binária)", "coluna_binaria"),
            ParametroSpec("sucesso", "Valor que conta como sucesso", "nivel", depende_de="coluna"),
            ParametroSpec("grupo", "Grupo (2 níveis)", "coluna_binaria"),
            ParametroSpec(
                "alternativa",
                "Hipótese alternativa (H₁)",
                "opcao",
                padrao=ALTERNATIVA_PADRAO_2P,
                opcoes=list(ALTERNATIVAS_2P),
            ),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    def comparacao_inicial(self) -> ComparacaoPValores:
        return ComparacaoPValores(
            titulo_esquerda="Teste Z",
            titulo_direita="Fisher exato",
            hipoteses=list(ALTERNATIVAS_2P),
            linhas=[(None, None)] * len(ALTERNATIVAS_2P),
        )

    @staticmethod
    def tabela_2x2(
        df: pd.DataFrame, coluna: str, sucesso: str, grupo: str
    ) -> tuple[list[str], list[list[int]], int]:
        """(rótulos dos grupos, [[s₁, f₁], [s₂, f₂]], linhas descartadas)."""
        validas = df[[coluna, grupo]].dropna()
        resposta = validas[coluna].map(rotulo_nivel)
        rotulos_grupo = validas[grupo].map(rotulo_nivel)
        grupos = niveis_coluna(validas[grupo])
        tabela = []
        for g in grupos:
            r = resposta[rotulos_grupo == g]
            s = int((r == sucesso).sum())
            tabela.append([s, len(r) - s])
        return grupos, tabela, len(df) - len(validas)

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        coluna, sucesso, grupo = params.get("coluna"), params.get("sucesso"), params.get("grupo")
        erros = erros_coluna(df, coluna, "a resposta", numerica=False)
        erros += erros_coluna(df, grupo, "o grupo", numerica=False)
        if not erros and coluna == grupo:
            erros.append("A resposta e o grupo devem ser colunas diferentes.")
        if not erros:
            niveis = niveis_coluna(df[coluna])
            if len(niveis) != 2:
                erros.append(
                    f"A coluna '{coluna}' deve ter exatamente 2 valores distintos "
                    f"(tem {len(niveis)})."
                )
            elif not sucesso:
                erros.append("Escolha o valor que conta como sucesso.")
            elif str(sucesso) not in niveis:
                erros.append(f"O valor '{sucesso}' não aparece na coluna '{coluna}'.")
        if not erros:
            grupos, tabela, _ = self.tabela_2x2(df, coluna, str(sucesso), grupo)
            if len(grupos) != 2:
                erros.append(
                    f"A coluna de grupo '{grupo}' deve ter exatamente 2 níveis com dados válidos "
                    f"(tem {len(grupos)})."
                )
            else:
                sucessos = tabela[0][0] + tabela[1][0]
                total = sum(map(sum, tabela))
                if sucessos in (0, total):
                    tipo = "fracasso" if sucessos == 0 else "sucesso"
                    erros.append(
                        f"Todas as observações válidas são {tipo}: a proporção combinada é "
                        f"{'0' if sucessos == 0 else '1'} e o teste Z não pode ser calculado."
                    )
        erros += erro_opcao(
            params.get("alternativa", ALTERNATIVA_PADRAO_2P),
            ALTERNATIVAS_2P,
            "Escolha uma hipótese alternativa válida.",
        )
        erros += erro_alfa(params.get("alfa", 0.05))
        return erros

    # ---------------- Execução ----------------
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)

        coluna, sucesso, grupo = params["coluna"], str(params["sucesso"]), params["grupo"]
        alfa = float(params.get("alfa", 0.05))
        alternativa = ALTERNATIVAS_2P[params.get("alternativa", ALTERNATIVA_PADRAO_2P)]

        (g1, g2), tabela, descartadas = self.tabela_2x2(df, coluna, sucesso, grupo)
        (k1, f1), (k2, f2) = tabela
        n1, n2 = k1 + f1, k2 + f2
        p1, p2 = k1 / n1, k2 / n2
        p_comb = (k1 + k2) / (n1 + n2)

        testes_z = {
            alt: proportions_ztest([k1, k2], [n1, n2], alternative=_STATSMODELS[alt])
            for alt in _SIMBOLO
        }
        z, p_valor = (float(v) for v in testes_z[alternativa])
        p_fisher = {
            alt: float(stats.fisher_exact(tabela, alternative=alt).pvalue) for alt in _SIMBOLO
        }
        ic = ic_diferenca_wald(k1, n1, k2, n2, alfa, alternativa)
        h_cohen = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))

        avisos = []
        if 0 in (k1, f1, k2, f2):
            odds, odds_ic = math.nan, (math.nan, math.nan)
            avisos.append(
                "A tabela 2×2 tem uma célula com zero: a razão de chances (odds ratio) não é "
                "definida."
            )
        else:
            tabela_sm = Table2x2(np.array(tabela, dtype=float))
            odds = float(tabela_sm.oddsratio)
            odds_ic = self._ic_odds_ratio(tabela_sm, alfa, alternativa)

        estatisticas = {
            "n1": float(n1),
            "n2": float(n2),
            "sucessos1": float(k1),
            "sucessos2": float(k2),
            "n_descartadas": float(descartadas),
            "p1": p1,
            "p2": p2,
            "p_combinada": p_comb,
            "diferenca": p1 - p2,
            "erro_padrao": math.sqrt(p_comb * (1 - p_comb) * (1 / n1 + 1 / n2)),
            "z": z,
            "p_valor": p_valor,
            "ic_inferior": ic[0],
            "ic_superior": ic[1],
            "h_cohen": h_cohen,
            "odds_ratio": odds,
            "or_ic_inferior": odds_ic[0],
            "or_ic_superior": odds_ic[1],
            "p_fisher": p_fisher[alternativa],
        }

        simbolo, texto = _SIMBOLO[alternativa], _TEXTO_2[alternativa]
        de_quem = f"a proporção de '{sucesso}' em '{coluna}' no grupo '{g1}'"
        no_g2 = f"proporção no grupo '{g2}'"
        interpretacao = interpretar(
            p_valor,
            alfa,
            h0="p₁ = p₂",
            h1=f"p₁ {simbolo} p₂",
            conclusao_rejeita=f"Há evidência estatística de que {de_quem} é {texto} {no_g2}.",
            conclusao_nao_rejeita=(
                f"Não há evidência suficiente de que {de_quem} seja {texto} {no_g2}."
            ),
        )

        if descartadas:
            avisos.insert(
                0, f"{descartadas} linha(s) com resposta ou grupo ausente foram descartadas."
            )
        esperadas = [n * p for n in (n1, n2) for p in (p_comb, 1 - p_comb)]
        if min(esperadas) < MINIMO_ESPERADO:
            avisos.append(
                f"Há frequência esperada menor que {MINIMO_ESPERADO} (mínima = "
                f"{formatar_numero(min(esperadas), 1)}): a aproximação normal é fraca. Prefira o "
                "Fisher exato do card."
            )

        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=interpretacao,
            tabelas={
                "Resumo": self._tabela_resumo(estatisticas, (g1, g2), sucesso, alfa, alternativa),
                "Tabela 2×2": pd.DataFrame(
                    [[g1, k1, f1, n1], [g2, k2, f2, n2]],
                    columns=["Grupo", f"Sucesso ('{sucesso}')", "Fracasso", "Total"],
                ),
            },
            figuras=[
                barras(
                    [(f"{g1} (n = {n1})", p1), (f"{g2} (n = {n2})", p2)],
                    f"Proporção de '{sucesso}' em '{coluna}' por '{grupo}'",
                    "Proporção",
                    maximo=1.0,
                    referencias=[
                        (f"Proporção combinada ({_percentual(p_comb)})", p_comb, "tracejado")
                    ],
                    percentual=True,
                )
            ],
            avisos=avisos,
            comparacao=ComparacaoPValores(
                titulo_esquerda="Teste Z",
                titulo_direita="Fisher exato",
                hipoteses=list(ALTERNATIVAS_2P),
                linhas=[(float(testes_z[a][1]), p_fisher[a]) for a in ALTERNATIVAS_2P.values()],
            ),
        )

    @staticmethod
    def _ic_odds_ratio(tabela_sm: Table2x2, alfa: float, alternativa: str) -> tuple[float, float]:
        """IC de Woolf da odds ratio; unilateral quando H₁ é (como os demais ICs do teste):
        "p₁ > p₂" ⇔ OR > 1 → [limite inferior com 1 − α; +∞); "p₁ < p₂" → (0; superior]."""
        if alternativa == "two-sided":
            baixo, alto = tabela_sm.oddsratio_confint(alpha=alfa)
            return float(baixo), float(alto)
        baixo, alto = tabela_sm.oddsratio_confint(alpha=2 * alfa)
        return (float(baixo), math.inf) if alternativa == "greater" else (0.0, float(alto))

    @staticmethod
    def _tabela_resumo(
        e: dict[str, float], grupos: tuple[str, str], sucesso: str, alfa: float, alternativa: str
    ) -> pd.DataFrame:
        confianca = f"{(1 - alfa) * 100:.0f}%"
        tipo_ic = "" if alternativa == "two-sided" else " (unilateral)"
        ic = f"[{formatar_numero(e['ic_inferior'])}; {formatar_numero(e['ic_superior'])}]"
        if math.isnan(e["odds_ratio"]):
            odds, odds_ic = "— (célula zero)", "—"
        else:
            odds = formatar_numero(e["odds_ratio"])
            odds_ic = (
                f"[{formatar_numero(e['or_ic_inferior'])}; {formatar_numero(e['or_ic_superior'])}]"
            )
        g1, g2 = grupos
        linhas = [
            ("Grupo 1", f"'{g1}' (n = {int(e['n1'])})"),
            ("Grupo 2", f"'{g2}' (n = {int(e['n2'])})"),
            (
                f"Proporção de '{sucesso}' no grupo 1 (p̂₁)",
                f"{formatar_numero(e['p1'])} ({_percentual(e['p1'])})",
            ),
            (
                f"Proporção de '{sucesso}' no grupo 2 (p̂₂)",
                f"{formatar_numero(e['p2'])} ({_percentual(e['p2'])})",
            ),
            ("Proporção combinada (p̂)", formatar_numero(e["p_combinada"])),
            ("Diferença (p̂₁ − p̂₂)", formatar_numero(e["diferenca"])),
            ("Erro padrão (combinado)", formatar_numero(e["erro_padrao"])),
            ("Estatística z", formatar_numero(e["z"])),
            ("p-valor (Z)", formatar_p_valor(e["p_valor"])),
            (f"IC {confianca} para p₁ − p₂{tipo_ic}", ic),
            ("h de Cohen", formatar_numero(e["h_cohen"])),
            ("Odds ratio (razão de chances)", odds),
            (f"IC {confianca} para a odds ratio (Woolf){tipo_ic}", odds_ic),
            ("p-valor (Fisher exato)", formatar_p_valor(e["p_fisher"])),
        ]
        return pd.DataFrame(linhas, columns=["Medida", "Valor"])
