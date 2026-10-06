"""Testes de proporções."""

import math

import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

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
