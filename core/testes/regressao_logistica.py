"""Regressão logística (decisões em docs/regressao_logistica.md)."""

import math
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit

from core import diagnosticos as dg
from core.base import (
    ErroExecucao,
    ErroValidacao,
    Figura,
    GrupoFiguras,
    ParametroSpec,
    ResultadoTeste,
    Secao,
)
from core.figuras import boxplot, cascata, dispersao
from core.interpretacao import (
    REJEITA_H0,
    decidir,
    formatar_numero,
    formatar_p_valor,
    interpretar,
)
from core.testes.regressao import (
    CONFIANCAS,
    VALORES_AJUSTADOS,
    DadosModelo,
    Previsao,
    SimuladorRegressao,
    TesteRegressaoLinear,
    _confianca,
    avisos_dos_dados,
    formatar_coeficiente,
    linhas_coeficientes,
    preparar_dados,
    residuos_por_eixo,
)
from core.tipos import niveis_coluna, rotulo_nivel
from core.validacao import converter_numero, erro_alfa, erro_opcao, erros_coluna

LIMIAR_PADRAO = 0.5
EPV_MINIMO = 10
COLUNAS_DIAGNOSTICO = ["Preditor", "z", "p-valor", "Interpretação"]


def _percentual(p: float, casas: int = 1) -> str:
    return f"{formatar_numero(p * 100, casas)}%"


def binarizar(df: pd.DataFrame, nome_y: str, evento: str) -> pd.DataFrame:
    """Cópia de `df` com y = 1,0 no evento, 0,0 no outro valor e NaN onde faltar."""
    copia = df.copy()
    rotulos = df[nome_y].map(lambda v: None if pd.isna(v) else rotulo_nivel(v))
    copia[nome_y] = np.where(rotulos.isna(), np.nan, (rotulos == evento).astype(float))
    return copia


# ---------------------------------------------------------------------------
# Simulação
# ---------------------------------------------------------------------------
class SimuladorLogistico(SimuladorRegressao):
    """Probabilidade prevista do evento para valores escolhidos.

    η̂ = x₀ᵀβ̂ (log-odds); P = 1/(1 + e^(−η̂)); IC: 1/(1 + e^(−(η̂ ± z·√(x₀ᵀ V x₀)))), com z
    normal (como o Wald dos coeficientes e o `predict(type = "link", se.fit = TRUE)` do R).
    Sem intervalo de predição: a resposta é sim/não. Classe prevista: evento se P ≥ limiar.
    """

    def __init__(
        self,
        dados: DadosModelo,
        coeficientes: np.ndarray,
        covariancia: np.ndarray,
        confianca: float,
        evento: str,
        nao_evento: str,
        limiar: float,
    ):
        super().__init__(dados, coeficientes, covariancia, 0.0, math.inf, confianca)
        self.evento, self.nao_evento, self.limiar = evento, nao_evento, limiar

    @property
    def lado_esquerdo(self) -> str:
        return "logit(P)"

    @property
    def titulo_resultado(self) -> str:
        return f"Probabilidade prevista de '{self.nome_y}' = '{self.evento}'"

    @property
    def rodape_equacao(self) -> str:
        return f"P = P('{self.nome_y}' = '{self.evento}') = 1 / (1 + e^(−logit))"

    def prever(self, valores: dict[str, float | str]) -> Previsao:
        x0 = self.linha_x(valores)
        eta = float(x0 @ self._beta)
        ep = math.sqrt(max(float(x0 @ self._cov @ x0), 0.0))
        contribuicoes = self._contribuicoes(x0)
        prob = float(expit(eta))
        return Previsao(
            valor=prob,
            ic_inferior=float(expit(eta - self._t * ep)),
            ic_superior=float(expit(eta + self._t * ep)),
            ip_inferior=math.nan,
            ip_superior=math.nan,
            confianca=self.confianca,
            contribuicoes=contribuicoes,
            extrapolacoes=self._extrapolacoes(valores),
            figura=cascata(
                ("Intercepto", float(self._beta[0])),
                contribuicoes,
                "Logit previsto",
                "log-odds",
                "Contribuição de cada termo (log-odds)",
            ),
            classe=self.evento if prob >= self.limiar else self.nao_evento,
        )

    def textos_previsao(self, previsao: Previsao) -> tuple[str, list[tuple[str, str]], list[str]]:
        nivel = f"{previsao.confianca * 100:.0f}%"
        limiar = formatar_coeficiente(self.limiar).rstrip("0").rstrip(",")
        return (
            _percentual(previsao.valor),
            [
                (
                    f"IC {nivel} da probabilidade: [{_percentual(previsao.ic_inferior)}; "
                    f"{_percentual(previsao.ic_superior)}]",
                    "Onde deve estar a probabilidade do evento para casos com esses valores.",
                ),
                (
                    f"Classe prevista (limiar {limiar}): '{previsao.classe}'",
                    "Evento quando a probabilidade prevista é maior ou igual ao limiar.",
                ),
            ],
            [
                "Sem intervalo de predição: a resposta é sim/não (evento ou não evento).",
                f"Intervalo de {nivel}: o nível de confiança escolhido na aba Parâmetros.",
            ],
        )


# ---------------------------------------------------------------------------
# Teste
# ---------------------------------------------------------------------------
class TesteRegressaoLogistica(TesteRegressaoLinear):
    """Regressão logística binária por máxima verossimilhança (`statsmodels.api.Logit`).

    Decisões em docs/regressao_logistica.md. Resumo:
    - y com exatamente 2 valores + "Evento"; preditores e referências como na linear; linhas
      incompletas removidas; colinearidade perfeita e n > p + 1 bloqueiam; EPV < 10 e separação
      (completa ou quase) geram aviso.
    - IC de Wald (= `confint.default`); odds ratio = e^β com IC e^(LI), e^(LS).
    - Diagnósticos: VIF/GVIF (= `car::vif`), Box-Tidwell (x·ln x, um preditor positivo por vez,
      z de Wald), Hosmer-Lemeshow (= `hoslem.test`, 10 grupos).
    - Teste global: razão de verossimilhança contra o modelo nulo (χ² com p gl); decisão p ≤ α.
      Pseudo-R²: McFadden = 1 − LL/LL₀; Nagelkerke = Cox-Snell/(1 − e^(2·LL₀/n)).
    - Classificação pelo limiar do formulário: matriz de confusão, acurácia, sensibilidade,
      especificidade e AUC (= `pROC::auc`).
    """

    id = "regres_logit"
    nome = "Regressão Logística"
    grupo = "Regressão"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec(
                "y",
                "Variável dependente (y)",
                "coluna_binaria",
                ajuda=(
                    "Só colunas com exatamente 2 valores (ex.: sim/não): a regressão logística "
                    "modela a probabilidade de um deles."
                ),
            ),
            ParametroSpec(
                "evento", "Evento (o valor que conta como sucesso)", "nivel", depende_de="y"
            ),
            ParametroSpec("preditores", "Preditores (X)", "preditores", depende_de="y"),
            ParametroSpec(
                "referencias",
                "Nível de referência",
                "niveis_referencia",
                obrigatorio=False,
                depende_de="preditores",
            ),
            ParametroSpec(
                "limiar",
                "Limiar de classificação",
                "numero",
                padrao=LIMIAR_PADRAO,
                ajuda="Probabilidade a partir da qual um caso é classificado como o evento.",
            ),
            ParametroSpec(
                "confianca", "Nível de confiança", "opcao", padrao="95%", opcoes=list(CONFIANCAS)
            ),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        nome_y = params.get("y")
        evento = params.get("evento")
        preditores = list(params.get("preditores") or [])
        erros = erros_coluna(df, nome_y, "a variável dependente (y)", numerica=False)
        if not erros:
            niveis = niveis_coluna(df[nome_y])
            if len(niveis) != 2:
                erros.append(
                    f"A variável dependente '{nome_y}' deve ter exatamente 2 valores "
                    f"(tem {len(niveis)})."
                )
            elif evento not in niveis:
                erros.append("Escolha qual valor da variável dependente é o evento.")
        if not preditores:
            erros.append("Selecione ao menos um preditor.")
        for nome in preditores:
            if nome not in df.columns:
                erros.append(f"A coluna '{nome}' não existe no arquivo.")
        if nome_y in preditores:
            erros.append("A variável dependente não pode ser também um preditor.")
        if len(set(preditores)) != len(preditores):
            erros.append("Os preditores devem ser colunas diferentes.")
        try:
            limiar = converter_numero(params.get("limiar", LIMIAR_PADRAO))
            if not 0 < limiar < 1:
                raise ValueError
        except (TypeError, ValueError):
            erros.append("O limiar de classificação deve ser um número entre 0 e 1.")
        erros += erro_opcao(
            params.get("confianca", "95%"), CONFIANCAS, "Escolha um nível de confiança válido."
        )
        erros += erro_alfa(params.get("alfa", 0.05))
        if erros:
            return erros
        binario = binarizar(df, nome_y, evento)
        dados = preparar_dados(binario, nome_y, preditores, params.get("referencias"))
        if len(dados.y) and np.ptp(dados.y) == 0:
            classe = evento if dados.y[0] == 1 else "o outro valor"
            return [
                f"Nas linhas completas, '{nome_y}' só tem a classe '{classe}': é preciso haver "
                "casos das duas classes."
            ]
        return self._erros_modelo(
            binario, nome_y, preditores, params.get("referencias"), VALORES_AJUSTADOS
        )

    # ---------------- Execução ----------------
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)

        import statsmodels.api as sm  # import tardio (abre o app mais rápido)

        nome_y, evento = params["y"], params["evento"]
        nao_evento = next(n for n in niveis_coluna(df[nome_y]) if n != evento)
        preditores = list(params["preditores"])
        alfa = float(params.get("alfa", 0.05))
        confianca = _confianca(params.get("confianca", "95%"))
        limiar = converter_numero(params.get("limiar", LIMIAR_PADRAO))
        binario = binarizar(df, nome_y, evento)
        dados = preparar_dados(binario, nome_y, preditores, params.get("referencias"))
        n, k = dados.x.shape

        ajuste = self._ajustar(sm, dados.y, dados.x)
        cov = np.asarray(ajuste.cov_params())
        eta = dados.x @ np.asarray(ajuste.params)
        prob = expit(eta)
        y = dados.y

        llf, ll0 = float(ajuste.llf), float(ajuste.llnull)
        cox_snell = 1 - math.exp(2 * (ll0 - llf) / n)
        hl = dg.hosmer_lemeshow(y, prob)
        area = dg.auc(y, prob)
        previsto = prob >= limiar
        vp = int((previsto & (y == 1)).sum())
        fn = int((~previsto & (y == 1)).sum())
        fp = int((previsto & (y == 0)).sum())
        vn = int((~previsto & (y == 0)).sum())
        eventos = int(y.sum())

        estatisticas = {
            "n": float(n),
            "n_descartadas": float(dados.descartadas),
            "eventos": float(eventos),
            "loglik": llf,
            "loglik_nulo": ll0,
            "lr": float(ajuste.llr),
            "gl": float(ajuste.df_model),
            "p_valor": float(ajuste.llr_pvalue),
            "mcfadden": 1 - llf / ll0,
            "cox_snell": cox_snell,
            "nagelkerke": cox_snell / (1 - math.exp(2 * ll0 / n)),
            "aic": float(ajuste.aic),
            "bic": float(ajuste.bic),
            "auc": area,
            "vp": float(vp),
            "fn": float(fn),
            "fp": float(fp),
            "vn": float(vn),
            "acuracia": (vp + vn) / n,
            "sensibilidade": vp / (vp + fn) if vp + fn else math.nan,
            "especificidade": vn / (vn + fp) if vn + fp else math.nan,
            "epv": min(eventos, n - eventos) / (k - 1),
        }
        if hl is not None:
            estatisticas |= {"hl": hl.estatistica, "gl_hl": float(hl.gl), "p_hl": hl.p_valor}

        convergiu = bool(ajuste.mle_retvals.get("converged", True))
        estatisticas["convergiu"] = float(convergiu)
        separacao = dg.separacao(eta, y)
        rotulos = {"evento": evento, "nao_evento": nao_evento}
        secoes = [
            Secao("Pressupostos"),
            self._secao_colinearidade(dados, cov),
            self._secao_box_tidwell(sm, dados, alfa, estatisticas),
            self._secao_hosmer_lemeshow(hl, alfa, separacao),
            self._secao_amostra(estatisticas, separacao, k - 1, convergiu),
        ]
        modelo, interpretacao = self._secao_modelo_logit(estatisticas, alfa, nome_y, evento, dados)
        secoes += [
            modelo,
            self._secao_classificacao(estatisticas, limiar, rotulos),
            self._secao_coeficientes_logit(dados, ajuste, confianca),
        ]
        tabelas = {nome: t for s in secoes for nome, t in s.tabelas.items()}
        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=estatisticas["p_valor"],
            alfa=alfa,
            decisao=decidir(estatisticas["p_valor"], alfa),
            interpretacao=interpretacao,
            tabelas=tabelas,
            figuras=self._figuras_logit(dados, ajuste, prob, limiar, rotulos),
            avisos=[a for s in secoes for a in s.avisos],
            secoes=secoes,
            simulacao=SimuladorLogistico(
                dados, ajuste.params, cov, confianca, evento, nao_evento, limiar
            ),
        )

    @staticmethod
    def _ajustar(sm, y: np.ndarray, x: np.ndarray):
        """Ajuste por Newton-Raphson; avisos do statsmodels (separação, convergência) ficam
        silenciosos: a separação é detectada e avisada em português."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return sm.Logit(y, x).fit(disp=0, maxiter=100)
            except np.linalg.LinAlgError as erro:
                raise ErroExecucao(
                    "Não foi possível ajustar o modelo (matriz singular durante o ajuste). "
                    "Verifique se algum preditor separa perfeitamente as classes ou tem pouca "
                    "variação."
                ) from erro

    # ---------------- Seções ----------------
    def _secao_box_tidwell(
        self, sm, dados: DadosModelo, alfa: float, estatisticas: dict[str, float]
    ) -> Secao:
        """Grava z e p de cada preditor testado em `estatisticas` (bt_z_<nome>, bt_p_<nome>)."""
        numericas = [v for v in dados.variaveis if not v.categorica]
        if not numericas:
            return Secao(
                "Linearidade do logit",
                nivel=2,
                textos=["Sem preditores numéricos: não há linearidade do logit a verificar."],
            )
        linhas = []
        for variavel in numericas:
            x = dados.x[:, variavel.colunas[0]]
            if (x <= 0).any():
                linhas.append(
                    (variavel.nome, "—", "—", "Não aplicável: o preditor tem valores ≤ 0.")
                )
                continue
            extra = np.column_stack([dados.x, x * np.log(x)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    teste = sm.Logit(dados.y, extra).fit(disp=0, maxiter=100)
                    z, p = float(teste.tvalues[-1]), float(teste.pvalues[-1])
                except (np.linalg.LinAlgError, ValueError):
                    z, p = math.nan, math.nan
            if math.isnan(p):
                linhas.append((variavel.nome, "—", "—", "Não calculado (ajuste instável)."))
                continue
            estatisticas[f"bt_z_{variavel.nome}"] = z
            estatisticas[f"bt_p_{variavel.nome}"] = p
            texto = (
                "A relação com o logit não é linear: considere transformar o preditor "
                "(ex.: logaritmo) ou incluir um termo quadrático."
                if decidir(p, alfa) == REJEITA_H0
                else "Sem evidência de não linearidade no logit."
            )
            linhas.append((variavel.nome, formatar_numero(z, 3), formatar_p_valor(p), texto))
        tabela = pd.DataFrame(linhas, columns=COLUNAS_DIAGNOSTICO)
        tabela.attrs["dicas"] = {
            "z": "z de Wald do termo x·ln(x) acrescentado ao modelo (Box-Tidwell)."
        }
        return Secao(
            "Linearidade do logit",
            nivel=2,
            tabelas={"Linearidade do logit (Box-Tidwell)": tabela},
            notas=[
                "Box-Tidwell: cada preditor numérico positivo é testado acrescentando x·ln(x) ao "
                "modelo; p pequeno indica que o efeito no logit não é linear."
            ],
        )

    @staticmethod
    def _secao_hosmer_lemeshow(
        hl: dg.ResultadoHL | None, alfa: float, separacao: str | None = None
    ) -> Secao:
        if hl is None:
            return Secao(
                "Qualidade do ajuste",
                nivel=2,
                textos=[
                    "Hosmer-Lemeshow não calculado: as probabilidades previstas formam menos de "
                    "3 grupos distintos."
                ],
            )
        if separacao is not None:
            # Com separação as probabilidades previstas vão a 0 ou 1 e o teste "acerta" tudo:
            # um p alto aqui não significa bom ajuste.
            texto = (
                "Não interpretável: há separação entre as classes (veja 'Tamanho da amostra e "
                "separação'), e as probabilidades previstas ficam perto de 0 ou de 1."
            )
        elif decidir(hl.p_valor, alfa) == REJEITA_H0:
            texto = (
                "As probabilidades previstas se afastam das frequências observadas: o modelo se "
                "ajusta mal aos dados."
            )
        else:
            texto = (
                "Sem evidência de mau ajuste: as probabilidades previstas acompanham as "
                "frequências observadas."
            )
        tabela = pd.DataFrame(
            [
                (
                    "Hosmer-Lemeshow",
                    formatar_numero(hl.estatistica),
                    f"{hl.gl}",
                    formatar_p_valor(hl.p_valor),
                    texto,
                )
            ],
            columns=["Teste", "Estatística", "gl", "p-valor", "Interpretação"],
        )
        notas = [
            f"Hosmer-Lemeshow: casos agrupados pelos decis das probabilidades previstas "
            f"({hl.grupos} grupos)."
        ]
        return Secao(
            "Qualidade do ajuste", nivel=2, tabelas={"Qualidade do ajuste": tabela}, notas=notas
        )

    @staticmethod
    def _secao_amostra(
        e: dict[str, float], separacao: str | None, p: int, convergiu: bool = True
    ) -> Secao:
        avisos = []
        if not convergiu:
            avisos.append(
                "O ajuste por máxima verossimilhança não convergiu em 100 iterações: os "
                "coeficientes, os erros padrão e os p-valores podem não ser confiáveis."
            )
        if e["epv"] < EPV_MINIMO:
            avisos.append(
                f"Poucos eventos por variável (EPV = {formatar_numero(e['epv'], 1)} < "
                f"{EPV_MINIMO}): os coeficientes podem ficar instáveis e enviesados. Reduza o "
                "número de preditores ou reúna mais casos."
            )
        if separacao == "completa":
            avisos.append(
                "Separação completa: os preditores separam perfeitamente as duas classes. Os "
                "coeficientes tendem ao infinito e os erros padrão e p-valores não são confiáveis."
            )
        elif separacao == "quase":
            avisos.append(
                "Separação quase completa: os preditores separam as classes quase sem "
                "sobreposição. Alguns coeficientes e p-valores não são confiáveis."
            )
        textos = [
            f"Eventos por variável (EPV): {formatar_numero(e['epv'], 1)} "
            f"({int(min(e['eventos'], e['n'] - e['eventos']))} casos na classe menos frequente "
            f"para {p} coeficiente(s) de preditores)."
        ]
        if not avisos:
            textos.append("Tamanho de amostra adequado e sem separação entre as classes.")
        return Secao("Tamanho da amostra e separação", nivel=2, textos=textos, avisos=avisos)

    def _secao_modelo_logit(
        self, e: dict[str, float], alfa: float, nome_y: str, evento: str, dados: DadosModelo
    ) -> tuple[Secao, str]:
        interpretacao = interpretar(
            e["p_valor"],
            alfa,
            h0=(
                "todos os coeficientes dos preditores são zero; os preditores não ajudam a "
                f"prever '{nome_y}' = '{evento}'"
            ),
            h1="ao menos um coeficiente é diferente de zero",
            conclusao_rejeita=(
                f"Há evidência estatística de que os preditores ajudam a prever a chance de "
                f"'{nome_y}' = '{evento}' (R² de McFadden = {formatar_numero(e['mcfadden'], 4)})."
            ),
            conclusao_nao_rejeita=(
                f"Não há evidência suficiente de que os preditores ajudem a prever "
                f"'{nome_y}' = '{evento}'."
            ),
        )
        tabela = pd.DataFrame(
            [
                (
                    formatar_numero(e["lr"]),
                    f"{e['gl']:.0f}",
                    formatar_p_valor(e["p_valor"]),
                    formatar_numero(e["mcfadden"]),
                    formatar_numero(e["nagelkerke"]),
                    formatar_numero(e["aic"], 2),
                    formatar_numero(e["loglik"], 2),
                    f"{e['n']:.0f}",
                    f"{e['eventos']:.0f}",
                )
            ],
            columns=[
                "LR χ²",
                "gl",
                "p-valor",
                "R² McFadden",
                "R² Nagelkerke",
                "AIC",
                "Log-verossimilhança",
                "n",
                "Eventos",
            ],
        )
        tabela.attrs["dicas"] = {
            "LR χ²": "Razão de verossimilhança: 2·(LL do modelo − LL do modelo só com intercepto).",
            "R² McFadden": "1 − LL/LL₀ (LL₀ = log-verossimilhança do modelo nulo).",
            "R² Nagelkerke": "R² de Cox-Snell reescalado para ir de 0 a 1.",
            "AIC": "Critério de Akaike: −2·LL + 2·(número de coeficientes); menor é melhor.",
        }
        avisos = avisos_dos_dados(dados)
        secao = Secao(
            "Modelo",
            destaque=decidir(e["p_valor"], alfa).replace("H0", "H₀"),
            textos=[interpretacao],
            tabelas={"Modelo": tabela},
            avisos=avisos,
        )
        return secao, interpretacao

    @staticmethod
    def _secao_classificacao(e: dict[str, float], limiar: float, rotulos: dict) -> Secao:
        evento, outro = rotulos["evento"], rotulos["nao_evento"]
        vp, fn, fp, vn = (int(e[c]) for c in ("vp", "fn", "fp", "vn"))
        matriz = pd.DataFrame(
            [
                (f"'{evento}'", vp, fn, vp + fn),
                (f"'{outro}'", fp, vn, fp + vn),
                ("Total", vp + fp, fn + vn, vp + fn + fp + vn),
            ],
            columns=["Observado", f"Previsto '{evento}'", f"Previsto '{outro}'", "Total"],
        )
        matriz.attrs["destaques"] = [2]
        medidas = pd.DataFrame(
            [
                (
                    _percentual(e["acuracia"]),
                    _percentual(e["sensibilidade"]),
                    _percentual(e["especificidade"]),
                    formatar_numero(e["auc"]),
                )
            ],
            columns=["Acurácia", "Sensibilidade", "Especificidade", "AUC"],
        )
        medidas.attrs["dicas"] = {
            "Acurácia": "Proporção de casos classificados corretamente.",
            "Sensibilidade": f"Entre os casos '{evento}', proporção prevista como '{evento}'.",
            "Especificidade": f"Entre os casos '{outro}', proporção prevista como '{outro}'.",
            "AUC": "Área sob a curva ROC: chance de um caso do evento ter probabilidade prevista "
            "maior que um caso do outro valor (0,5 = acaso; 1 = perfeito). Não depende do limiar.",
        }
        rotulo_limiar = formatar_coeficiente(limiar).rstrip("0").rstrip(",")
        return Secao(
            "Classificação",
            textos=[f"Limiar de classificação: {rotulo_limiar} (evento quando P ≥ limiar)."],
            tabelas={"Matriz de confusão": matriz, "Medidas de classificação": medidas},
        )

    @staticmethod
    def _secao_coeficientes_logit(dados: DadosModelo, ajuste, confianca: float) -> Secao:
        ic = np.asarray(ajuste.conf_int(alpha=1 - confianca))
        rotulo = f"{confianca * 100:.0f}%"

        def numeros(i: int) -> tuple:
            return (
                formatar_coeficiente(ajuste.params[i]),
                formatar_coeficiente(ajuste.bse[i]),
                formatar_numero(ajuste.tvalues[i], 3),
                formatar_p_valor(ajuste.pvalues[i]),
                formatar_coeficiente(float(np.exp(ajuste.params[i]))),
                formatar_coeficiente(float(np.exp(ic[i, 0]))),
                formatar_coeficiente(float(np.exp(ic[i, 1]))),
            )

        with np.errstate(over="ignore"):  # separação: e^β → ∞ (exibido como +∞)
            linhas, destaques = linhas_coeficientes(dados, numeros, 7)
        tabela = pd.DataFrame(
            linhas,
            columns=[
                "Preditor",
                "Estimativa (log-odds)",
                "EP",
                "z",
                "p-valor",
                "Odds ratio",
                f"LI OR ({rotulo})",
                f"LS OR ({rotulo})",
            ],
        )
        tabela.attrs["destaques"] = destaques
        tabela.attrs["dicas"] = {
            "EP": "Erro padrão da estimativa.",
            "Odds ratio": "e^(estimativa): quanto a chance do evento é multiplicada quando o "
            "preditor aumenta 1 unidade (ou, na categórica, em relação à referência).",
            f"LI OR ({rotulo})": f"Limite inferior do IC de {rotulo} (Wald) da odds ratio.",
            f"LS OR ({rotulo})": f"Limite superior do IC de {rotulo} (Wald) da odds ratio.",
        }
        notas = ["Odds ratio > 1 aumenta a chance do evento; < 1 diminui."]
        if destaques:
            notas.append(
                "Categóricas: cada nível é comparado com o nível de referência (linha com —)."
            )
        return Secao("Coeficientes", tabelas={"Coeficientes": tabela}, notas=notas)

    # ---------------- Visualização ----------------
    @staticmethod
    def _figuras_logit(
        dados: DadosModelo, ajuste, prob: np.ndarray, limiar: float, rotulos: dict
    ) -> list[Figura | GrupoFiguras]:
        y = dados.y
        fpr, tpr = dg.curva_roc(y, prob)
        roc = dispersao(
            fpr,
            tpr,
            f"Curva ROC (AUC = {formatar_numero(dg.auc(y, prob), 3)})",
            "1 − especificidade",
            "Sensibilidade",
            [("Acaso", 0.0, 0.0, 1.0, 1.0, "tracejado")],
            conectar=True,
        )
        classes = boxplot(
            [
                (f"'{rotulos['evento']}'", prob[y == 1]),
                (f"'{rotulos['nao_evento']}'", prob[y == 0]),
            ],
            "Probabilidade prevista por classe observada",
            "Probabilidade prevista",
            referencias=[("Limiar", limiar, "tracejado")],
        )
        with np.errstate(all="ignore"):  # separação: log(0) nos casos com P = 0 ou 1
            residuos = np.nan_to_num(np.asarray(ajuste.resid_dev))

        grupo = residuos_por_eixo(
            dados,
            residuos,
            prob,
            "Resíduos de deviance",
            "Resíduo de deviance",
            "probabilidade prevista",
        )
        return [roc, classes, grupo]
