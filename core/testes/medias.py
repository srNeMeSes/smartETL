"""Testes de médias."""

import math

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
from core.figuras import histograma
from core.interpretacao import decidir, formatar_numero, formatar_p_valor, interpretar
from core.tipos import e_numerica
from core.validacao import converter_numero

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
        erros: list[str] = []
        coluna = params.get("coluna")
        if not coluna:
            erros.append("Selecione a variável.")
        elif coluna not in df.columns:
            erros.append(f"A coluna '{coluna}' não existe no arquivo.")
        elif not e_numerica(df[coluna]):
            erros.append(f"A coluna '{coluna}' não é numérica.")
        else:
            x = df[coluna].dropna().astype(float)
            if np.isinf(x).any():
                erros.append(f"A coluna '{coluna}' contém valores infinitos.")
            elif len(x) < 2:
                erros.append(
                    f"São necessárias ao menos 2 observações válidas em '{coluna}' (há {len(x)})."
                )
            elif x.nunique() == 1:
                erros.append(
                    f"Todos os valores de '{coluna}' são iguais (variância zero): "
                    "o teste t não pode ser calculado."
                )

        try:
            converter_numero(params.get("mu0"))
        except (TypeError, ValueError):
            erros.append("Informe um número válido para a média hipotética (μ₀).")

        if params.get("alternativa", ALTERNATIVA_PADRAO) not in ALTERNATIVAS:
            erros.append("Escolha uma hipótese alternativa válida.")

        try:
            alfa = float(params.get("alfa", 0.05))
            if not 0 < alfa < 1:
                raise ValueError
        except (TypeError, ValueError):
            erros.append("O nível de significância (α) deve estar entre 0 e 1.")
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
