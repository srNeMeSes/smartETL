"""Regressão linear (especificação em docs/regressao_linear.md)."""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from core import diagnosticos as dg
from core.base import (
    ErroValidacao,
    Figura,
    GrupoFiguras,
    ParametroSpec,
    ResultadoTeste,
    Secao,
    TesteBase,
)
from core.figuras import boxplot, cascata, dispersao, qq_normal
from core.interpretacao import (
    REJEITA_H0,
    decidir,
    formatar_numero,
    formatar_p_valor,
    interpretar,
)
from core.tipos import e_numerica, ordenar_niveis, rotulo_nivel
from core.validacao import erro_alfa, erro_opcao, erros_coluna

INTERCEPTO = "(Intercepto)"
MIN_POR_NIVEL = 5


@dataclass(frozen=True)
class VariavelModelo:
    """Preditor original do modelo e as colunas que ele gera na matriz X."""

    nome: str
    categorica: bool
    colunas: list[int]  # índices em X (o intercepto é a coluna 0)
    niveis: list[str] = field(default_factory=list)  # categórica: todos, inclusive a referência
    referencia: str | None = None
    contagens: dict[str, int] = field(default_factory=dict)


@dataclass
class DadosModelo:
    """y e matriz X (com intercepto) prontos para o ajuste, após descartar linhas incompletas."""

    nome_y: str
    y: np.ndarray
    x: np.ndarray
    colunas: list[str]  # nome de cada coluna de X ("(Intercepto)", "x", "cargo[Gerente]"...)
    variaveis: list[VariavelModelo]
    linhas: pd.DataFrame  # dados originais das linhas usadas (y e preditores)
    descartadas: int


def e_categorica(serie: pd.Series) -> bool:
    """Preditor categórico = coluna não numérica (texto, booleana); numéricas entram como número."""
    return not e_numerica(serie)


def niveis_ordenados(serie: pd.Series) -> list[str]:
    """Níveis de uma coluna categórica como texto, em ordem crescente."""
    return [rotulo_nivel(v) for v in ordenar_niveis(list(pd.unique(serie.dropna())))]


def nivel_referencia_padrao(serie: pd.Series) -> str | None:
    """Nível mais frequente (empate: o primeiro em ordem crescente)."""
    textos = serie.dropna().map(rotulo_nivel)
    if textos.empty:
        return None
    contagens = textos.value_counts()
    maximo = contagens.max()
    return next(n for n in niveis_ordenados(serie) if contagens.get(n, 0) == maximo)


def preparar_dados(
    df: pd.DataFrame, nome_y: str, preditores: list[str], referencias: dict[str, str] | None = None
) -> DadosModelo:
    """Monta y e X (intercepto + numéricas + k − 1 dummies por categórica).

    Linhas com valor ausente em y ou em algum preditor são descartadas. `referencias` dá o nível
    de referência de cada categórica (ausente ou inválido → o mais frequente). A ordem das
    dummies segue a ordem crescente dos níveis, sem a referência.
    """
    referencias = referencias or {}
    linhas = df[[nome_y, *preditores]].dropna()
    descartadas = len(df) - len(linhas)
    colunas = [INTERCEPTO]
    blocos = [np.ones(len(linhas))]
    variaveis = []
    for nome in preditores:
        serie = linhas[nome]
        if e_categorica(serie):
            textos = serie.map(rotulo_nivel)
            niveis = niveis_ordenados(serie)
            referencia = referencias.get(nome)
            if referencia not in niveis:
                referencia = nivel_referencia_padrao(serie)
            indices = []
            for nivel in niveis:
                if nivel == referencia:
                    continue
                indices.append(len(colunas))
                colunas.append(f"{nome}[{nivel}]")
                blocos.append((textos == nivel).to_numpy(dtype=float))
            contagens = {n: int((textos == n).sum()) for n in niveis}
            variaveis.append(VariavelModelo(nome, True, indices, niveis, referencia, contagens))
        else:
            variaveis.append(VariavelModelo(nome, False, [len(colunas)]))
            colunas.append(nome)
            blocos.append(serie.to_numpy(dtype=float))
    return DadosModelo(
        nome_y=nome_y,
        y=linhas[nome_y].to_numpy(dtype=float),
        x=np.column_stack(blocos),
        colunas=colunas,
        variaveis=variaveis,
        linhas=linhas,
        descartadas=descartadas,
    )


def variaveis_colineares(dados: DadosModelo) -> list[list[str]]:
    """Grupos de preditores em colinearidade perfeita (vazio se X tem posto completo).

    Cada vetor do núcleo de X (valores singulares ≈ 0) dá uma combinação linear exata; os
    preditores originais com peso não nulo nela formam um grupo. "(Intercepto)" aparece quando a
    combinação envolve uma constante (ex.: preditor constante, dummies que somam 1).
    """
    x = dados.x
    escala = np.where(np.abs(x).max(axis=0) > 0, np.abs(x).max(axis=0), 1.0)
    _, valores, vt = np.linalg.svd(x / escala, full_matrices=True)
    tolerancia = max(x.shape) * np.finfo(float).eps * (valores[0] if len(valores) else 1.0) * 1e3
    nulos = [vt[i] for i in range(x.shape[1]) if i >= len(valores) or valores[i] <= tolerancia]
    dono = {0: INTERCEPTO}
    for variavel in dados.variaveis:
        for indice in variavel.colunas:
            dono[indice] = variavel.nome
    grupos: list[list[str]] = []
    for vetor in nulos:
        pesos = np.abs(vetor) / np.abs(vetor).max()
        nomes: list[str] = []
        for indice in np.nonzero(pesos > 1e-6)[0]:
            if dono[int(indice)] not in nomes:
                nomes.append(dono[int(indice)])
        if nomes not in grupos:
            grupos.append(nomes)
    return grupos


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------
def formatar_coeficiente(valor: float) -> str:
    """Pelo menos 4 algarismos significativos e 2 casas (3.135,82; 35,41; 3,878; 0,001013)."""
    if valor == 0 or not math.isfinite(valor):
        return formatar_numero(valor, 2)
    casas = min(max(2, 3 - math.floor(math.log10(abs(valor)))), 8)
    return formatar_numero(valor, casas)


def _confianca(texto: str) -> float:
    return float(texto.rstrip("%").replace(",", ".")) / 100


# ---------------------------------------------------------------------------
# Simulação
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CampoSimulacao:
    """Entrada da aba Simulação: numérica (campo + slider) ou categórica (lista)."""

    nome: str
    categorica: bool
    inicial: float | str
    minimo: float | None = None
    maximo: float | None = None
    niveis: tuple[str, ...] = ()
    referencia: str | None = None


@dataclass(frozen=True)
class TermoEquacao:
    texto: str  # ex.: "3.135,82", " + 35,41·(Experiência)", " − 3,878·(wt)"
    variavel: str | None  # preditor original do termo (None no intercepto)


@dataclass(frozen=True)
class Previsao:
    valor: float
    ic_inferior: float  # IC da média prevista
    ic_superior: float
    ip_inferior: float  # intervalo de predição (nova observação)
    ip_superior: float
    confianca: float
    contribuicoes: list[tuple[str, float]]  # (preditor original, coeficiente × valor)
    extrapolacoes: list[str]  # avisos de valores fora da faixa observada
    figura: Figura  # cascata do intercepto até o valor previsto


class SimuladorRegressao:
    """Previsões do modelo ajustado para valores escolhidos (sem Flet; a UI só exibe).

    IC da média: ŷ ± t·√(x₀ᵀ V x₀); intervalo de predição: ŷ ± t·√(x₀ᵀ V x₀ + σ̂²), com V a
    covariância dos coeficientes, σ̂² = QM do resíduo e t com n − p − 1 gl (o `predict` do R).
    """

    def __init__(
        self,
        dados: DadosModelo,
        coeficientes: np.ndarray,
        covariancia: np.ndarray,
        sigma2: float,
        gl_residuo: float,
        confianca: float,
    ):
        self.nome_y = dados.nome_y
        self.confianca = confianca
        self._dados = dados
        self._beta = np.asarray(coeficientes, dtype=float)
        self._cov = np.asarray(covariancia, dtype=float)
        self._sigma2 = float(sigma2)
        self._t = float(stats.t.ppf(1 - (1 - confianca) / 2, gl_residuo))
        self.campos = [self._campo(v) for v in dados.variaveis]

    def _campo(self, variavel: VariavelModelo) -> CampoSimulacao:
        if variavel.categorica:
            return CampoSimulacao(
                variavel.nome,
                True,
                variavel.referencia,
                niveis=tuple(variavel.niveis),
                referencia=variavel.referencia,
            )
        valores = self._dados.linhas[variavel.nome].astype(float)
        return CampoSimulacao(
            variavel.nome, False, float(valores.mean()), float(valores.min()), float(valores.max())
        )

    def valores_iniciais(self) -> dict[str, float | str]:
        return {c.nome: c.inicial for c in self.campos}

    def _rotulo_coluna(self, indice: int) -> str:
        coluna = self._dados.colunas[indice]
        variavel = next(v for v in self._dados.variaveis if indice in v.colunas)
        if not variavel.categorica:
            return variavel.nome
        nivel = coluna[len(variavel.nome) + 1 : -1]
        repetido = sum(nivel in v.niveis for v in self._dados.variaveis if v.categorica) > 1
        return f"{variavel.nome}: {nivel}" if repetido else nivel

    def equacao(self) -> list[TermoEquacao]:
        """Termos da equação "y = b₀ + b₁·(x₁) + …", cada um ligado ao seu preditor."""
        termos = [TermoEquacao(f"{self.nome_y} = {formatar_coeficiente(self._beta[0])}", None)]
        for variavel in self._dados.variaveis:
            for indice in variavel.colunas:
                b = self._beta[indice]
                sinal = "−" if b < 0 else "+"
                texto = f" {sinal} {formatar_coeficiente(abs(b))}·({self._rotulo_coluna(indice)})"
                termos.append(TermoEquacao(texto, variavel.nome))
        return termos

    def linha_x(self, valores: dict[str, float | str]) -> np.ndarray:
        """Vetor x₀ (intercepto + preditores; categóricas viram as dummies)."""
        x0 = np.zeros(len(self._dados.colunas))
        x0[0] = 1.0
        for variavel in self._dados.variaveis:
            valor = valores[variavel.nome]
            if variavel.categorica:
                coluna = f"{variavel.nome}[{valor}]"
                if coluna in self._dados.colunas:
                    x0[self._dados.colunas.index(coluna)] = 1.0
            else:
                x0[variavel.colunas[0]] = float(valor)
        return x0

    def prever(self, valores: dict[str, float | str]) -> Previsao:
        x0 = self.linha_x(valores)
        valor = float(x0 @ self._beta)
        var_media = float(x0 @ self._cov @ x0)
        ep_media = math.sqrt(max(var_media, 0.0))
        ep_pred = math.sqrt(max(var_media, 0.0) + self._sigma2)
        contribuicoes = [
            (v.nome, float(sum(x0[i] * self._beta[i] for i in v.colunas)))
            for v in self._dados.variaveis
        ]
        extrapolacoes = []
        for campo in self.campos:
            if campo.categorica:
                continue
            x = float(valores[campo.nome])
            if x < campo.minimo or x > campo.maximo:
                extrapolacoes.append(
                    f"'{campo.nome}' = {formatar_coeficiente(x)} está fora da faixa observada "
                    f"[{formatar_coeficiente(campo.minimo)}; {formatar_coeficiente(campo.maximo)}]"
                    ": a previsão é uma extrapolação."
                )
        figura = cascata(
            ("Intercepto", float(self._beta[0])),
            contribuicoes,
            f"Previsto ({self.nome_y})",
            self.nome_y,
            "Contribuição de cada termo",
        )
        return Previsao(
            valor=valor,
            ic_inferior=valor - self._t * ep_media,
            ic_superior=valor + self._t * ep_media,
            ip_inferior=valor - self._t * ep_pred,
            ip_superior=valor + self._t * ep_pred,
            confianca=self.confianca,
            contribuicoes=contribuicoes,
            extrapolacoes=extrapolacoes,
            figura=figura,
        )


# ---------------------------------------------------------------------------
# Teste
# ---------------------------------------------------------------------------
VALORES_AJUSTADOS = "Valores ajustados"
CONFIANCAS = ("90%", "95%", "99%")
NOTA_DW = "Regra prática; os limites formais dependem de n e do número de preditores."
AVISO_ORDEM = (
    "Só é interpretável se as linhas tiverem ordem significativa (tempo, sequência de coleta)."
)
NOTA_N_GRANDE = (
    "Com n grande, o teste rejeita desvios irrelevantes da normalidade; consulte o gráfico Q-Q "
    "(aba Visualização)."
)
DICA_RMSE = (
    "RMSE = erro padrão dos resíduos = √(SQE / (n − p − 1)), em que SQE é a soma dos quadrados "
    "dos resíduos, n o número de observações e p o número de coeficientes dos preditores."
)
COLUNAS_TESTE = ["Teste", "Estatística", "gl", "p-valor", "Interpretação"]


class TesteRegressaoLinear(TesteBase):
    """Regressão linear por mínimos quadrados (`statsmodels.api.OLS`), com pressupostos.

    Especificação completa em docs/regressao_linear.md. Resumo das escolhas:
    - y numérica; preditores numéricos (entram como número) ou não numéricos (dummies k − 1,
      referência escolhida no formulário; padrão: o nível mais frequente); linhas incompletas
      descartadas (aviso); exige n > p + 1 e posto completo (informa os preditores colineares).
    - Pressupostos, sempre com o α do formulário: Breusch-Pagan studentizado (Koenker,
      `het_breuschpagan(robust=True)`, = `bptest`), Goldfeld-Quandt e Harrison-McCabe como o
      lmtest (core/diagnosticos.py; ordenação escolhida, padrão: valores ajustados),
      Durbin-Watson (faixa interna) e Breusch-Godfrey de ordem 1 (= `bgtest`), VIF/GVIF como o
      `car::vif` (classificado por GVIF^(1/(2·gl)))²; a faixa é aplicada ao valor exibido com 2
      casas, para nunca contradizer a tela), Shapiro-Wilk (n ≤ 5000) ou Lilliefors.
    - Teste F global: H₀ = todos os coeficientes dos preditores são zero. Decisão: p ≤ α.
    """

    id = "regres_linear"
    nome = "Regressão Linear"
    grupo = "Regressão"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec(
                "y",
                "Variável dependente (y)",
                "coluna_numerica",
                ajuda=(
                    "Só colunas numéricas aparecem aqui: a regressão linear modela um valor "
                    "numérico (para uma resposta sim/não, use a Regressão Logística)."
                ),
            ),
            ParametroSpec("preditores", "Preditores (X)", "preditores"),
            ParametroSpec(
                "referencias",
                "Nível de referência",
                "niveis_referencia",
                obrigatorio=False,
                depende_de="preditores",
            ),
            ParametroSpec(
                "ordenar_por",
                "Ordenação dos dados",
                "ordenacao",
                padrao=VALORES_AJUSTADOS,
                opcoes=[VALORES_AJUSTADOS],
                ajuda=(
                    "Ordem usada pelos testes de Goldfeld-Quandt e Harrison-McCabe (verificam se "
                    "a variância dos resíduos cresce ao longo dela)."
                ),
            ),
            ParametroSpec(
                "confianca", "Nível de confiança", "opcao", padrao="95%", opcoes=list(CONFIANCAS)
            ),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        nome_y = params.get("y")
        preditores = list(params.get("preditores") or [])
        erros = erros_coluna(df, nome_y, "a variável dependente (y)")
        if not preditores:
            erros.append("Selecione ao menos um preditor.")
        for nome in preditores:
            if nome not in df.columns:
                erros.append(f"A coluna '{nome}' não existe no arquivo.")
        if nome_y in preditores:
            erros.append("A variável dependente não pode ser também um preditor.")
        if len(set(preditores)) != len(preditores):
            erros.append("Os preditores devem ser colunas diferentes.")
        ordenar = params.get("ordenar_por") or VALORES_AJUSTADOS
        if ordenar != VALORES_AJUSTADOS:
            erros += [
                e.replace("a variável", "a ordenação")
                for e in erros_coluna(df, ordenar, "a ordenação")
            ]
        erros += erro_opcao(
            params.get("confianca", "95%"), CONFIANCAS, "Escolha um nível de confiança válido."
        )
        erros += erro_alfa(params.get("alfa", 0.05))
        if erros:
            return erros
        return self._erros_modelo(df, nome_y, preditores, params.get("referencias"), ordenar)

    def _erros_modelo(
        self,
        df: pd.DataFrame,
        nome_y: str,
        preditores: list[str],
        referencias: dict | None,
        ordenar: str,
    ) -> list[str]:
        dados = preparar_dados(df, nome_y, preditores, referencias)
        n, k = dados.x.shape
        erros = [
            f"O preditor '{v.nome}' tem um único nível nas linhas completas ('{v.niveis[0]}'): "
            "não há o que comparar."
            for v in dados.variaveis
            if v.categorica and len(v.niveis) < 2
        ]
        if erros:
            return erros
        if n <= k:
            return [
                f"Observações insuficientes: n = {n} linhas completas e o modelo tem {k} "
                f"coeficientes (intercepto + {k - 1}); é preciso n > p + 1."
            ]
        if np.ptp(dados.y) == 0:
            return [f"Todos os valores de '{nome_y}' são iguais: não há variação a explicar."]
        for grupo in variaveis_colineares(dados):
            nomes = [g for g in grupo if g != INTERCEPTO]
            if INTERCEPTO in grupo and len(nomes) == 1:
                erros.append(
                    f"O preditor '{nomes[0]}' é constante nas linhas usadas (colinearidade "
                    "perfeita com o intercepto): remova-o."
                )
            else:
                lista = ", ".join(f"'{g}'" for g in nomes)
                extra = " e o intercepto" if INTERCEPTO in grupo else ""
                erros.append(
                    f"Colinearidade perfeita entre {lista}{extra}: um deles é combinação exata "
                    "dos outros. Remova um dos preditores."
                )
        if not erros and ordenar != VALORES_AJUSTADOS:
            if df.loc[dados.linhas.index, ordenar].isna().any():
                erros.append(
                    f"A coluna de ordenação '{ordenar}' tem valores ausentes nas linhas do modelo."
                )
        return erros

    # ---------------- Execução ----------------
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)

        import statsmodels.api as sm  # import tardio (abre o app mais rápido)
        from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
        from statsmodels.stats.stattools import durbin_watson

        nome_y = params["y"]
        preditores = list(params["preditores"])
        alfa = float(params.get("alfa", 0.05))
        confianca = _confianca(params.get("confianca", "95%"))
        ordenar = params.get("ordenar_por") or VALORES_AJUSTADOS
        dados = preparar_dados(df, nome_y, preditores, params.get("referencias"))
        n, k = dados.x.shape

        ajuste = sm.OLS(dados.y, dados.x).fit()
        residuos = np.asarray(ajuste.resid)
        ajustados = np.asarray(ajuste.fittedvalues)
        cov = np.asarray(ajuste.cov_params())

        bp_lm, bp_p, _, _ = het_breuschpagan(residuos, dados.x, robust=True)
        z = ajustados if ordenar == VALORES_AJUSTADOS else df.loc[dados.linhas.index, ordenar]
        ordem = dg.ordem_estavel(np.asarray(z, dtype=float))
        gq = dg.goldfeld_quandt(dados.y, dados.x, ordem)
        hmc = dg.harrison_mccabe(residuos, ordem, k)
        dw = float(durbin_watson(residuos))
        bg = acorr_breusch_godfrey(ajuste, nlags=1, result_object=True)
        nome_normal, est_normal, p_normal = dg.normalidade(residuos)

        estatisticas = {
            "n": float(n),
            "n_descartadas": float(dados.descartadas),
            "r": math.sqrt(max(ajuste.rsquared, 0.0)),
            "r2": float(ajuste.rsquared),
            "r2_ajustado": float(ajuste.rsquared_adj),
            "rmse": math.sqrt(ajuste.mse_resid),
            "f": float(ajuste.fvalue),
            "gl1": float(ajuste.df_model),
            "gl2": float(ajuste.df_resid),
            "p_valor": float(ajuste.f_pvalue),
            "bp": float(bp_lm),
            "p_bp": float(bp_p),
            "dw": dw,
            "bg": float(bg.lm),
            "p_bg": float(bg.lmpval),
            "normalidade": est_normal,
            "p_normalidade": p_normal,
        }
        if gq is not None:
            estatisticas |= {"gq": gq.estatistica, "p_gq": gq.p_valor}
        if hmc is not None:
            estatisticas |= {"hmc": hmc.estatistica, "p_hmc": hmc.p_valor}

        rotulo_ordem = "valores ajustados" if ordenar == VALORES_AJUSTADOS else f"'{ordenar}'"
        secoes = [Secao("Pressupostos")]
        secoes.append(
            self._secao_heterocedasticidade(bp_lm, bp_p, k - 1, gq, hmc, alfa, rotulo_ordem)
        )
        secoes.append(self._secao_autocorrelacao(dw, float(bg.lm), float(bg.lmpval), alfa))
        secoes.append(self._secao_colinearidade(dados, cov))
        secoes.append(
            self._secao_normalidade(nome_normal, est_normal, p_normal, alfa, n > dg.LIMITE_SHAPIRO)
        )
        modelo, interpretacao = self._secao_modelo(estatisticas, alfa, nome_y, dados)
        secoes.append(modelo)
        secoes.append(self._secao_coeficientes(dados, ajuste, confianca))

        tabelas = {nome: t for s in secoes for nome, t in s.tabelas.items()}
        avisos = [a for s in secoes for a in s.avisos]
        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=estatisticas["p_valor"],
            alfa=alfa,
            decisao=decidir(estatisticas["p_valor"], alfa),
            interpretacao=interpretacao,
            tabelas=tabelas,
            figuras=self._figuras(dados, residuos, ajustados),
            avisos=avisos,
            secoes=secoes,
            simulacao=SimuladorRegressao(
                dados, ajuste.params, cov, ajuste.mse_resid, ajuste.df_resid, confianca
            ),
        )

    # ---------------- Seções da Análise ----------------
    @staticmethod
    def _linha_teste(nome: str, estatistica: str, gl: str, p: float, texto: str) -> tuple:
        return (nome, estatistica, gl, formatar_p_valor(p), texto)

    def _secao_heterocedasticidade(self, bp, p_bp, gl_bp, gq, hmc, alfa, rotulo_ordem) -> Secao:
        def rejeita(p: float) -> bool:
            return decidir(p, alfa) == REJEITA_H0

        linhas = [
            self._linha_teste(
                "Breusch-Pagan",
                formatar_numero(bp),
                f"{gl_bp}",
                p_bp,
                "A variância dos resíduos muda com os preditores (heterocedasticidade): os "
                "erros padrão e os p-valores dos coeficientes podem não ser confiáveis."
                if rejeita(p_bp)
                else "Sem evidência de que a variância dos resíduos mude com os preditores.",
            )
        ]
        crescente = f"A variância dos resíduos é maior na metade com os maiores {rotulo_ordem}."
        constante = f"Sem evidência de que a variância dos resíduos cresça com os {rotulo_ordem}."
        if gq is None:
            linhas.append(
                (
                    "Goldfeld-Quandt",
                    "—",
                    "—",
                    "—",
                    "Não calculado: observações insuficientes em cada metade dos dados.",
                )
            )
        else:
            linhas.append(
                self._linha_teste(
                    "Goldfeld-Quandt",
                    formatar_numero(gq.estatistica),
                    f"{gq.gl1}; {gq.gl2}",
                    gq.p_valor,
                    crescente if rejeita(gq.p_valor) else constante,
                )
            )
        if hmc is None:
            linhas.append(
                (
                    "Harrison-McCabe",
                    "—",
                    "—",
                    "—",
                    "Não calculado: observações insuficientes para dividir os dados ao meio.",
                )
            )
        else:
            linhas.append(
                self._linha_teste(
                    "Harrison-McCabe",
                    formatar_numero(hmc.estatistica),
                    "—",
                    hmc.p_valor,
                    crescente if rejeita(hmc.p_valor) else constante,
                )
            )
        tabela = pd.DataFrame(linhas, columns=COLUNAS_TESTE)
        tabela.attrs["dicas"] = {
            "p-valor": "No Harrison-McCabe, o p-valor vem de 1000 simulações (semente fixa)."
        }
        avisos = []
        if gq is not None and gq.degenerado:
            avisos.append(
                "Goldfeld-Quandt: em uma das metades algum preditor não varia (ex.: todos os "
                "casos no mesmo nível de uma categórica); o resultado é pouco confiável."
            )
        return Secao(
            "Heterocedasticidade",
            nivel=2,
            tabelas={"Heterocedasticidade": tabela},
            notas=[
                f"Goldfeld-Quandt e Harrison-McCabe: dados ordenados pelos {rotulo_ordem} e "
                "divididos ao meio."
            ],
            avisos=avisos,
        )

    def _secao_autocorrelacao(self, dw: float, bg: float, p_bg: float, alfa: float) -> Secao:
        exibido = round(dw, 2)
        tabela_dw = pd.DataFrame(
            [(formatar_numero(exibido, 2), dg.classificar_dw(exibido))],
            columns=["DW", "Interpretação"],
        )
        texto = (
            "Há evidência de autocorrelação de 1ª ordem nos resíduos: observações vizinhas não "
            "são independentes."
            if decidir(p_bg, alfa) == REJEITA_H0
            else "Sem evidência de autocorrelação de 1ª ordem nos resíduos."
        )
        tabela_bg = pd.DataFrame(
            [self._linha_teste("Breusch-Godfrey (ordem 1)", formatar_numero(bg), "1", p_bg, texto)],
            columns=COLUNAS_TESTE,
        )
        return Secao(
            "Autocorrelação",
            nivel=2,
            tabelas={"Durbin-Watson": tabela_dw, "Breusch-Godfrey": tabela_bg},
            notas=[NOTA_DW],
            avisos=[AVISO_ORDEM],
        )

    def _secao_colinearidade(self, dados: DadosModelo, cov: np.ndarray) -> Secao:
        if len(dados.variaveis) < 2:
            return Secao(
                "Colinearidade",
                nivel=2,
                textos=["Com um único preditor não há colinearidade a avaliar."],
            )
        grupos = [[i - 1 for i in v.colunas] for v in dados.variaveis]
        valores = dg.gvif(cov[1:, 1:], grupos)
        generalizado = any(len(v.colunas) > 1 for v in dados.variaveis)
        linhas = []
        for variavel, g in zip(dados.variaveis, valores, strict=True):
            gl = len(variavel.colunas)
            ajustado = g ** (1 / (2 * gl))
            classificado = round(ajustado**2 if gl > 1 else g, 2)
            tolerancia = 1 / classificado
            linha = [variavel.nome, formatar_numero(g, 2)]
            if generalizado:
                linha += [f"{gl}", formatar_numero(ajustado, 2)]
            linha += [formatar_numero(tolerancia, 2), dg.classificar_vif(classificado)]
            linhas.append(linha)
        colunas = ["Preditor", "VIF"]
        if generalizado:
            colunas = ["Preditor", "VIF / GVIF", "gl", "GVIF^(1/(2·gl))"]
        tabela = pd.DataFrame(linhas, columns=[*colunas, "Tolerância", "Interpretação"])
        notas = []
        if generalizado:
            tabela.attrs["dicas"] = {
                "GVIF^(1/(2·gl))": "Categóricas com 3+ níveis usam o GVIF (Fox & Monette); a "
                "interpretação usa (GVIF^(1/(2·gl)))², comparável ao VIF.",
                "Tolerância": "Tolerância = 1/VIF (nas categóricas, 1/(GVIF^(1/(2·gl)))²).",
            }
            notas.append(
                "Categóricas com 3 ou mais níveis: GVIF (Fox & Monette) em uma linha por "
                "variável; a interpretação usa (GVIF^(1/(2·gl)))²."
            )
        else:
            tabela.attrs["dicas"] = {"Tolerância": "Tolerância = 1/VIF."}
        return Secao("Colinearidade", nivel=2, tabelas={"Colinearidade (VIF)": tabela}, notas=notas)

    def _secao_normalidade(self, nome, estatistica, p, alfa, grande) -> Secao:
        texto = (
            "Os resíduos se afastam da distribuição normal; com amostras pequenas, os IC e os "
            "p-valores podem ficar imprecisos."
            if decidir(p, alfa) == REJEITA_H0
            else "Sem evidência contra a normalidade dos resíduos."
        )
        tabela = pd.DataFrame(
            [(nome, formatar_numero(estatistica), formatar_p_valor(p), texto)],
            columns=["Teste", "Estatística", "p-valor", "Interpretação"],
        )
        return Secao(
            "Normalidade dos resíduos",
            nivel=2,
            tabelas={"Normalidade dos resíduos": tabela},
            notas=[NOTA_N_GRANDE] if grande else [],
        )

    def _secao_modelo(
        self, e: dict[str, float], alfa: float, nome_y: str, dados: DadosModelo
    ) -> tuple[Secao, str]:
        interpretacao = interpretar(
            e["p_valor"],
            alfa,
            h0=f"todos os coeficientes dos preditores são zero; o modelo não explica '{nome_y}'",
            h1="ao menos um coeficiente é diferente de zero",
            conclusao_rejeita=(
                f"Há evidência estatística de que o modelo explica parte da variação de "
                f"'{nome_y}' (R² = {formatar_numero(e['r2'], 4)})."
            ),
            conclusao_nao_rejeita=(
                f"Não há evidência suficiente de que os preditores expliquem '{nome_y}'."
            ),
        )
        tabela = pd.DataFrame(
            [
                (
                    formatar_numero(e["r"]),
                    formatar_numero(e["r2"]),
                    formatar_numero(e["r2_ajustado"]),
                    formatar_coeficiente(e["rmse"]),
                    formatar_numero(e["f"]),
                    f"{e['gl1']:.0f}",
                    f"{e['gl2']:.0f}",
                    formatar_p_valor(e["p_valor"]),
                    f"{e['n']:.0f}",
                )
            ],
            columns=["R", "R²", "R² ajustado", "RMSE", "F", "gl1", "gl2", "p-valor", "n"],
        )
        tabela.attrs["dicas"] = {"RMSE": DICA_RMSE}
        avisos = []
        if dados.descartadas:
            avisos.append(
                f"{dados.descartadas} linha(s) com valor ausente em y ou em algum preditor foram "
                "removidas."
            )
        for variavel in dados.variaveis:
            for nivel, contagem in variavel.contagens.items():
                if contagem < MIN_POR_NIVEL:
                    avisos.append(
                        f"O nível '{nivel}' de '{variavel.nome}' tem só {contagem} "
                        f"observação(ões) (< {MIN_POR_NIVEL}): o coeficiente dele é pouco preciso."
                    )
        secao = Secao(
            "Modelo",
            destaque=decidir(e["p_valor"], alfa).replace("H0", "H₀"),
            textos=[interpretacao],
            tabelas={"Modelo": tabela},
            avisos=avisos,
        )
        return secao, interpretacao

    def _secao_coeficientes(self, dados: DadosModelo, ajuste, confianca: float) -> Secao:
        ic = np.asarray(ajuste.conf_int(alpha=1 - confianca))
        rotulo = f"{confianca * 100:.0f}%"
        vazio = ("", "", "", "", "", "")
        traco = ("—",) * 6

        def numeros(i: int) -> tuple:
            return (
                formatar_coeficiente(ajuste.params[i]),
                formatar_coeficiente(ajuste.bse[i]),
                formatar_coeficiente(ic[i, 0]),
                formatar_coeficiente(ic[i, 1]),
                formatar_numero(ajuste.tvalues[i], 3),
                formatar_p_valor(ajuste.pvalues[i]),
            )

        linhas = [(INTERCEPTO, *numeros(0))]
        destaques = []
        recuo = "    "
        for variavel in dados.variaveis:
            if not variavel.categorica:
                linhas.append((variavel.nome, *numeros(variavel.colunas[0])))
                continue
            destaques.append(len(linhas))
            linhas.append((f"{variavel.nome} (ref.: {variavel.referencia})", *vazio))
            for nivel in variavel.niveis:
                if nivel == variavel.referencia:
                    continue
                indice = dados.colunas.index(f"{variavel.nome}[{nivel}]")
                linhas.append((recuo + nivel, *numeros(indice)))
            linhas.append((recuo + variavel.referencia, *traco))
        tabela = pd.DataFrame(
            linhas,
            columns=[
                "Preditor",
                "Estimativa",
                "EP",
                f"LI ({rotulo})",
                f"LS ({rotulo})",
                "t",
                "p-valor",
            ],
        )
        tabela.attrs["destaques"] = destaques
        tabela.attrs["dicas"] = {
            "EP": "Erro padrão da estimativa.",
            f"LI ({rotulo})": f"Limite inferior do intervalo de confiança de {rotulo}.",
            f"LS ({rotulo})": f"Limite superior do intervalo de confiança de {rotulo}.",
        }
        notas = []
        if destaques:
            notas.append(
                "Categóricas: cada nível é comparado com o nível de referência (linha com —)."
            )
        return Secao("Coeficientes", tabelas={"Coeficientes": tabela}, notas=notas)

    # ---------------- Visualização ----------------
    def _figuras(
        self, dados: DadosModelo, residuos: np.ndarray, ajustados: np.ndarray
    ) -> list[Figura | GrupoFiguras]:
        def linha_zero(xs: np.ndarray) -> tuple:
            return ("Resíduo = 0", float(xs.min()), 0.0, float(xs.max()), 0.0, "tracejado")

        titulo = "Resíduos"
        opcoes = {
            VALORES_AJUSTADOS: dispersao(
                ajustados,
                residuos,
                f"{titulo} × valores ajustados",
                "Valores ajustados",
                "Resíduo",
                [linha_zero(ajustados)],
            )
        }
        for variavel in dados.variaveis:
            serie = dados.linhas[variavel.nome]
            if variavel.categorica:
                textos = serie.map(rotulo_nivel).to_numpy()
                opcoes[variavel.nome] = boxplot(
                    [(nivel, residuos[textos == nivel]) for nivel in variavel.niveis],
                    f"{titulo} por '{variavel.nome}'",
                    "Resíduo",
                    referencias=[("Resíduo = 0", 0.0, "tracejado")],
                )
            else:
                xs = serie.to_numpy(dtype=float)
                opcoes[variavel.nome] = dispersao(
                    xs,
                    residuos,
                    f"{titulo} × '{variavel.nome}'",
                    variavel.nome,
                    "Resíduo",
                    [linha_zero(xs)],
                )
        return [
            GrupoFiguras("Eixo X", opcoes, VALORES_AJUSTADOS),
            qq_normal(residuos, "Gráfico Q-Q dos resíduos", "Resíduo"),
        ]
