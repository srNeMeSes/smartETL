"""Testes para variáveis categóricas."""

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
from core.exatos import ic_odds_ratio_condicional, odds_ratio_condicional
from core.figuras import barras, barras_agrupadas
from core.interpretacao import decidir, formatar_numero, formatar_p_valor, interpretar
from core.tipos import niveis_coluna, rotulo_nivel
from core.validacao import erro_alfa, erro_opcao, erros_coluna

INDEPENDENCIA = "Independência"  # duas variáveis: tabela de contingência
ADERENCIA = "Aderência"  # uma variável: H₀ de proporções iguais
MODOS = (INDEPENDENCIA, ADERENCIA)
MINIMO_ESPERADO = 5


def _percentual(p: float) -> str:
    return f"{formatar_numero(p * 100, 1)}%"


def _aviso_esperadas(esperadas: np.ndarray, tabela_2x2: bool) -> list[str]:
    """Aviso de Cochran: esperadas abaixo de 5 tornam a aproximação qui-quadrado fraca."""
    baixas = int((esperadas < MINIMO_ESPERADO).sum())
    if not baixas:
        return []
    total = esperadas.size
    texto = (
        f"{baixas} de {total} célula(s) ({_percentual(baixas / total)}) têm frequência esperada "
        f"menor que {MINIMO_ESPERADO} (mínima = {formatar_numero(float(esperadas.min()), 2)}): a "
        "aproximação qui-quadrado pode não ser confiável."
    )
    if tabela_2x2:
        texto += " Para tabela 2×2, prefira o Teste exato de Fisher."
    else:
        texto += " Considere agrupar categorias com poucas observações."
    return [texto]


class QuiQuadrado(TesteBase):
    """Teste qui-quadrado de Pearson, em dois modos.

    - Independência: duas colunas categóricas; tabela de contingência montada com
      `pandas.crosstab` (níveis em ordem crescente; linhas com valor ausente em qualquer uma das
      colunas são descartadas). Wrapper de `scipy.stats.chi2_contingency`; a correção de
      continuidade de Yates é opcional (desligada por padrão) e, como no scipy, só se aplica a
      tabelas com 1 grau de liberdade (2×2). Tamanho de efeito: V de Cramér, calculado com a
      estatística sem correção.
    - Aderência (proporções iguais): uma coluna categórica; H₀: todas as categorias têm a mesma
      proporção (esperada n/k). Wrapper de `scipy.stats.chisquare`. Tamanho de efeito: w de Cohen.

    O p-valor é sempre a cauda superior da distribuição qui-quadrado. Sem card de comparação
    (não há hipóteses alternativas ≠/>/<). Avisos de Cochran quando há frequência esperada < 5.
    Decisão: rejeita H₀ quando p ≤ α.
    """

    id = "qui_quadrado"
    nome = "Qui-quadrado"
    grupo = "Categóricos"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec(
                "modo", "Tipo de teste", "opcao", padrao=INDEPENDENCIA, opcoes=list(MODOS)
            ),
            ParametroSpec("coluna1", "Variável 1", "coluna_categorica"),
            ParametroSpec("coluna2", "Variável 2", "coluna_categorica", obrigatorio=False),
            ParametroSpec("correcao", "Correção de Yates (tabela 2×2)", "booleano", padrao=False),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    # ---------------- Dados ----------------
    @staticmethod
    def contingencia(df: pd.DataFrame, coluna1: str, coluna2: str) -> tuple[pd.DataFrame, int]:
        """(tabela observada com níveis ordenados, linhas descartadas por valor ausente)."""
        validas = df[[coluna1, coluna2]].dropna()
        a = validas[coluna1].map(rotulo_nivel)
        b = validas[coluna2].map(rotulo_nivel)
        tabela = pd.crosstab(a, b).reindex(
            index=niveis_coluna(validas[coluna1]), columns=niveis_coluna(validas[coluna2])
        )
        tabela.index.name, tabela.columns.name = coluna1, coluna2
        return tabela, len(df) - len(validas)

    @staticmethod
    def frequencias(df: pd.DataFrame, coluna: str) -> pd.Series:
        rotulos = df[coluna].dropna().map(rotulo_nivel)
        return rotulos.value_counts().reindex(niveis_coluna(df[coluna]))

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        modo = params.get("modo", INDEPENDENCIA)
        erros = erro_opcao(modo, MODOS, "Escolha o tipo de teste.")
        c1, c2 = params.get("coluna1"), params.get("coluna2")
        erros += erros_coluna(df, c1, "a variável 1", numerica=False)
        if modo == INDEPENDENCIA:
            erros += erros_coluna(df, c2, "a variável 2", numerica=False)
            if not erros and c1 == c2:
                erros.append("As duas variáveis devem ser colunas diferentes.")
            if not erros:
                tabela, _ = self.contingencia(df, c1, c2)
                for coluna, k in ((c1, tabela.shape[0]), (c2, tabela.shape[1])):
                    if k < 2:
                        erros.append(
                            f"A coluna '{coluna}' precisa de ao menos 2 categorias com dados "
                            f"válidos (tem {k})."
                        )
        elif modo == ADERENCIA and not erros:
            k = len(self.frequencias(df, c1))
            if k < 2:
                erros.append(f"A coluna '{c1}' precisa de ao menos 2 categorias (tem {k}).")
        erros += erro_alfa(params.get("alfa", 0.05))
        return erros

    # ---------------- Execução ----------------
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)
        if params.get("modo", INDEPENDENCIA) == ADERENCIA:
            return self._aderencia(df, params)
        return self._independencia(df, params)

    def _independencia(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        c1, c2 = params["coluna1"], params["coluna2"]
        alfa = float(params.get("alfa", 0.05))
        yates = bool(params.get("correcao", False))
        tabela, descartadas = self.contingencia(df, c1, c2)
        observadas = tabela.to_numpy(dtype=float)
        n = int(observadas.sum())
        r, c = observadas.shape

        resultado = stats.chi2_contingency(observadas, correction=yates)
        sem_correcao = stats.chi2_contingency(observadas, correction=False)
        qui2, p_valor, gl = float(resultado.statistic), float(resultado.pvalue), int(resultado.dof)
        esperadas = resultado.expected_freq
        cramer_v = math.sqrt(float(sem_correcao.statistic) / (n * (min(r, c) - 1)))
        aplicou_yates = yates and gl == 1

        estatisticas = {
            "n": float(n),
            "n_descartadas": float(descartadas),
            "qui2": qui2,
            "qui2_sem_correcao": float(sem_correcao.statistic),
            "gl": float(gl),
            "p_valor": p_valor,
            "cramer_v": cramer_v,
            "celulas_esperado_baixo": float((esperadas < MINIMO_ESPERADO).sum()),
        }

        interpretacao = interpretar(
            p_valor,
            alfa,
            h0=f"'{c1}' e '{c2}' são independentes",
            h1=f"'{c1}' e '{c2}' estão associadas",
            conclusao_rejeita=f"Há evidência estatística de associação entre '{c1}' e '{c2}'.",
            conclusao_nao_rejeita=(
                f"Não há evidência suficiente de associação entre '{c1}' e '{c2}'."
            ),
        )

        avisos = []
        if descartadas:
            avisos.append(
                f"{descartadas} linha(s) com valor ausente em '{c1}' ou '{c2}' foram descartadas."
            )
        if yates and not aplicou_yates:
            avisos.append(
                "A correção de Yates só se aplica a tabelas 2×2 (1 grau de liberdade) e não foi "
                "usada."
            )
        avisos += _aviso_esperadas(esperadas, (r, c) == (2, 2))

        proporcoes_linha = observadas / observadas.sum(axis=1, keepdims=True)
        resumo = [
            ("Tipo de teste", INDEPENDENCIA),
            ("Observações (n)", f"{n}"),
            ("Dimensão da tabela", f"{r} × {c}"),
            ("Correção de Yates", "Sim" if aplicou_yates else "Não"),
            ("Estatística qui-quadrado (χ²)", formatar_numero(qui2)),
            ("Graus de liberdade", f"{gl}"),
            ("p-valor", formatar_p_valor(p_valor)),
            ("V de Cramér", formatar_numero(cramer_v)),
            (
                f"Células com esperado < {MINIMO_ESPERADO}",
                f"{int(estatisticas['celulas_esperado_baixo'])} de {esperadas.size}",
            ),
        ]
        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=interpretacao,
            tabelas={
                "Resumo": pd.DataFrame(resumo, columns=["Medida", "Valor"]),
                "Frequências observadas": self._tabela_com_totais(tabela),
                "Frequências esperadas": self._tabela_esperada(tabela, esperadas),
            },
            figuras=[
                barras_agrupadas(
                    [
                        (str(nivel), list(linha))
                        for nivel, linha in zip(tabela.index, proporcoes_linha, strict=True)
                    ],
                    [str(s) for s in tabela.columns],
                    f"'{c2}' por '{c1}' (% dentro de cada '{c1}')",
                    "Proporção",
                    maximo=1.0,
                    percentual=True,
                )
            ],
            avisos=avisos,
        )

    def _aderencia(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        c1 = params["coluna1"]
        alfa = float(params.get("alfa", 0.05))
        contagens = self.frequencias(df, c1)
        observadas = contagens.to_numpy(dtype=float)
        n, k = int(observadas.sum()), len(observadas)
        esperada = n / k

        resultado = stats.chisquare(observadas)
        qui2, p_valor, gl = float(resultado.statistic), float(resultado.pvalue), k - 1
        w = math.sqrt(qui2 / n)
        n_ausentes = int(df[c1].isna().sum())

        estatisticas = {
            "n": float(n),
            "n_ausentes": float(n_ausentes),
            "k": float(k),
            "qui2": qui2,
            "gl": float(gl),
            "p_valor": p_valor,
            "w_cohen": w,
            "esperada": esperada,
        }

        interpretacao = interpretar(
            p_valor,
            alfa,
            h0=f"as {k} categorias de '{c1}' têm a mesma proporção",
            h1="as proporções não são todas iguais",
            conclusao_rejeita=(
                f"Há evidência estatística de que as categorias de '{c1}' não ocorrem em "
                "proporções iguais."
            ),
            conclusao_nao_rejeita=(
                f"Não há evidência suficiente de que as categorias de '{c1}' ocorram em "
                "proporções diferentes."
            ),
        )

        avisos = []
        if n_ausentes:
            avisos.append(f"{n_ausentes} valor(es) ausente(s) em '{c1}' foram ignorados.")
        if params.get("coluna2"):
            avisos.append("A variável 2 não é usada no teste de aderência e foi ignorada.")
        if params.get("correcao"):
            avisos.append("A correção de Yates não se aplica ao teste de aderência.")
        avisos += _aviso_esperadas(np.full(k, esperada), tabela_2x2=False)

        frequencias = pd.DataFrame(
            {
                "Categoria": [str(c) for c in contagens.index],
                "Observada": [int(v) for v in observadas],
                "Esperada": [formatar_numero(esperada, 2)] * k,
                "Proporção observada": [_percentual(v / n) for v in observadas],
            }
        )
        resumo = [
            ("Tipo de teste", ADERENCIA),
            ("Observações (n)", f"{n}"),
            ("Categorias (k)", f"{k}"),
            ("Frequência esperada por categoria (n/k)", formatar_numero(esperada, 2)),
            ("Estatística qui-quadrado (χ²)", formatar_numero(qui2)),
            ("Graus de liberdade", f"{gl}"),
            ("p-valor", formatar_p_valor(p_valor)),
            ("w de Cohen", formatar_numero(w)),
        ]
        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=interpretacao,
            tabelas={
                "Resumo": pd.DataFrame(resumo, columns=["Medida", "Valor"]),
                "Frequências": frequencias,
            },
            figuras=[
                barras(
                    [(str(c), v / n) for c, v in zip(contagens.index, observadas, strict=True)],
                    f"Proporções observadas em '{c1}'",
                    "Proporção",
                    maximo=1.0,
                    referencias=[
                        (f"Proporção esperada (1/{k} = {_percentual(1 / k)})", 1 / k, "tracejado")
                    ],
                    percentual=True,
                )
            ],
            avisos=avisos,
        )

    @staticmethod
    def _tabela_com_totais(tabela: pd.DataFrame) -> pd.DataFrame:
        saida = tabela.copy()
        saida["Total"] = saida.sum(axis=1)
        saida.loc["Total"] = saida.sum(axis=0)
        saida = saida.astype(int).reset_index()
        saida.columns = [
            f"{tabela.index.name} \\ {tabela.columns.name}",
            *map(str, saida.columns[1:]),
        ]
        return saida

    @staticmethod
    def _tabela_esperada(tabela: pd.DataFrame, esperadas: np.ndarray) -> pd.DataFrame:
        saida = pd.DataFrame(
            [[formatar_numero(v, 2) for v in linha] for linha in esperadas],
            columns=[str(c) for c in tabela.columns],
        )
        saida.insert(
            0, f"{tabela.index.name} \\ {tabela.columns.name}", [str(i) for i in tabela.index]
        )
        return saida


# ---------------------------------------------------------------------------
# Teste exato de Fisher
# ---------------------------------------------------------------------------
ALTERNATIVAS_OR = {"OR ≠ 1": "two-sided", "OR > 1": "greater", "OR < 1": "less"}
ALTERNATIVA_PADRAO_OR = "OR ≠ 1"
_SIMBOLO_OR = {"two-sided": "≠", "greater": ">", "less": "<"}


class TesteFisher(TesteBase):
    """Teste exato de Fisher para uma tabela 2×2 (H₀: odds ratio = 1, independência).

    Entrada: duas colunas categóricas com exatamente 2 valores cada e o valor que é o "evento"
    em cada uma. A tabela é [[a, b], [c, d]] com linhas = (evento₁, outro₁) e colunas =
    (evento₂, outro₂); linhas com valor ausente em alguma das colunas são descartadas (aviso).
    p-valor exato: `scipy.stats.fisher_exact` (bilateral pelo método das probabilidades ≤ à
    da tabela observada, como no R). "OR > 1": o evento₁ aumenta a chance do evento₂.
    Odds ratio amostral = ad/bc; odds ratio condicional (EMV da hipergeométrica não central) com
    IC exato condicional (core/exatos.py: os valores de
    `scipy.stats.contingency.odds_ratio(kind="conditional")` e do `fisher.test` do R,
    vetorizados); o IC é unilateral quando H₁ é unilateral. Com célula zero,
    a odds ratio vai a 0 ou +∞ (com aviso). Sem card de comparação. Decisão: p ≤ α.
    """

    id = "fisher"
    nome = "Teste exato de Fisher"
    grupo = "Categóricos"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec("coluna1", "Variável 1 (linhas)", "coluna_binaria"),
            ParametroSpec("evento1", "Evento da variável 1", "nivel", depende_de="coluna1"),
            ParametroSpec("coluna2", "Variável 2 (colunas)", "coluna_binaria"),
            ParametroSpec("evento2", "Evento da variável 2", "nivel", depende_de="coluna2"),
            ParametroSpec(
                "alternativa",
                "Hipótese alternativa (H₁)",
                "opcao",
                padrao=ALTERNATIVA_PADRAO_OR,
                opcoes=list(ALTERNATIVAS_OR),
            ),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    @staticmethod
    def tabela_2x2(
        df: pd.DataFrame, coluna1: str, evento1: str, coluna2: str, evento2: str
    ) -> tuple[list[list[int]], tuple[str, str], tuple[str, str], int]:
        """([[a, b], [c, d]], (evento₁, outro₁), (evento₂, outro₂), linhas descartadas)."""
        validas = df[[coluna1, coluna2]].dropna()
        x = validas[coluna1].map(rotulo_nivel)
        y = validas[coluna2].map(rotulo_nivel)
        outro1 = next(n for n in niveis_coluna(validas[coluna1]) if n != evento1)
        outro2 = next(n for n in niveis_coluna(validas[coluna2]) if n != evento2)
        tabela = [
            [int(((x == linha) & (y == coluna)).sum()) for coluna in (evento2, outro2)]
            for linha in (evento1, outro1)
        ]
        return tabela, (evento1, outro1), (evento2, outro2), len(df) - len(validas)

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        erros: list[str] = []
        colunas = []
        for i in (1, 2):
            coluna, evento = params.get(f"coluna{i}"), params.get(f"evento{i}")
            erros_i = erros_coluna(df, coluna, f"a variável {i}", numerica=False)
            colunas.append(coluna)
            if erros_i:
                erros += erros_i
                continue
            niveis = niveis_coluna(df[coluna])
            if len(niveis) != 2:
                erros.append(
                    f"A coluna '{coluna}' deve ter exatamente 2 valores distintos "
                    f"(tem {len(niveis)})."
                )
            elif not evento:
                erros.append(f"Escolha o evento da variável {i}.")
            elif str(evento) not in niveis:
                erros.append(f"O valor '{evento}' não aparece na coluna '{coluna}'.")
        if not erros and colunas[0] == colunas[1]:
            erros.append("As duas variáveis devem ser colunas diferentes.")
        if not erros:
            validas = df[colunas].dropna()
            for coluna in colunas:
                k = validas[coluna].nunique()
                if k != 2:
                    erros.append(
                        f"Nas linhas completas, a coluna '{coluna}' deve ter os 2 valores "
                        f"(tem {k})."
                    )
        erros += erro_opcao(
            params.get("alternativa", ALTERNATIVA_PADRAO_OR),
            ALTERNATIVAS_OR,
            "Escolha uma hipótese alternativa válida.",
        )
        erros += erro_alfa(params.get("alfa", 0.05))
        return erros

    # ---------------- Execução ----------------
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)

        c1, c2 = params["coluna1"], params["coluna2"]
        e1, e2 = str(params["evento1"]), str(params["evento2"])
        alfa = float(params.get("alfa", 0.05))
        alternativa = ALTERNATIVAS_OR[params.get("alternativa", ALTERNATIVA_PADRAO_OR)]

        tabela, niveis1, niveis2, descartadas = self.tabela_2x2(df, c1, e1, c2, e2)
        (a, b), (c, d) = tabela
        n = a + b + c + d

        p_valor = float(stats.fisher_exact(tabela, alternative=alternativa).pvalue)
        or_cond = odds_ratio_condicional(a, b, c, d)
        ic_inferior, ic_superior = ic_odds_ratio_condicional(a, b, c, d, 1 - alfa, alternativa)
        celula_zero = 0 in (a, b, c, d)
        or_amostral = (a * d) / (b * c) if b * c else (math.inf if a * d else math.nan)

        estatisticas = {
            "n": float(n),
            "n_descartadas": float(descartadas),
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "d": float(d),
            "p_valor": p_valor,
            "odds_ratio_amostral": float(or_amostral),
            "odds_ratio_condicional": or_cond,
            "ic_inferior": float(ic_inferior),
            "ic_superior": float(ic_superior),
        }

        simbolo = _SIMBOLO_OR[alternativa]
        evento_1, evento_2 = f"'{c1}' = '{e1}'", f"'{c2}' = '{e2}'"
        if alternativa == "two-sided":
            rejeita = f"Há evidência estatística de associação entre {evento_1} e {evento_2}."
            nao = f"Não há evidência suficiente de associação entre {evento_1} e {evento_2}."
        else:
            efeito = "aumenta" if alternativa == "greater" else "diminui"
            rejeita = f"Há evidência estatística de que {evento_1} {efeito} a chance de {evento_2}."
            nao = f"Não há evidência suficiente de que {evento_1} {efeito} a chance de {evento_2}."
        interpretacao = interpretar(
            p_valor,
            alfa,
            h0="OR = 1, as variáveis são independentes",
            h1=f"OR {simbolo} 1",
            conclusao_rejeita=rejeita,
            conclusao_nao_rejeita=nao,
        )

        avisos = []
        if descartadas:
            avisos.append(
                f"{descartadas} linha(s) com valor ausente em '{c1}' ou '{c2}' foram descartadas."
            )
        if celula_zero:
            avisos.append(
                "A tabela tem uma célula com zero: a odds ratio vai a 0 ou +∞ e o IC fica aberto "
                "de um lado. O p-valor exato continua válido."
            )

        confianca = f"{(1 - alfa) * 100:.0f}%"
        tipo_ic = "" if alternativa == "two-sided" else " (unilateral)"
        resumo = [
            ("Observações (n)", f"{n}"),
            (f"Evento da variável 1 ('{c1}')", f"'{e1}'"),
            (f"Evento da variável 2 ('{c2}')", f"'{e2}'"),
            ("p-valor exato", formatar_p_valor(p_valor)),
            ("Odds ratio amostral (ad/bc)", formatar_numero(or_amostral)),
            ("Odds ratio condicional (EMV)", formatar_numero(or_cond)),
            (
                f"IC {confianca} exato para a odds ratio{tipo_ic}",
                f"[{formatar_numero(ic_inferior)}; {formatar_numero(ic_superior)}]",
            ),
        ]
        tabela_df = pd.DataFrame(tabela, index=list(niveis1), columns=list(niveis2))
        tabela_df.index.name, tabela_df.columns.name = c1, c2
        proporcoes = [[v / sum(linha) if sum(linha) else 0.0 for v in linha] for linha in tabela]

        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=interpretacao,
            tabelas={
                "Resumo": pd.DataFrame(resumo, columns=["Medida", "Valor"]),
                "Tabela 2×2": QuiQuadrado._tabela_com_totais(tabela_df),
            },
            figuras=[
                barras_agrupadas(
                    [(nivel, linha) for nivel, linha in zip(niveis1, proporcoes, strict=True)],
                    list(niveis2),
                    f"'{c2}' por '{c1}' (% dentro de cada '{c1}')",
                    "Proporção",
                    maximo=1.0,
                    percentual=True,
                )
            ],
            avisos=avisos,
        )


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------
ALTERNATIVAS_MCNEMAR = {"p₁ ≠ p₂": "two-sided", "p₁ > p₂": "greater", "p₁ < p₂": "less"}
ALTERNATIVA_PADRAO_MCNEMAR = "p₁ ≠ p₂"
LIMITE_EXATO = 25  # b + c abaixo disso: a decisão usa o teste exato
_TEXTO_MCNEMAR = {"two-sided": "diferente da", "greater": "maior que a", "less": "menor que a"}


def mcnemar_exato(b: int, c: int, alternativa: str) -> float:
    """Binomial nos pares discordantes: b ~ Bin(b + c, 1/2) sob H₀."""
    return float(stats.binomtest(b, b + c, 0.5, alternative=alternativa).pvalue)


def mcnemar_assintotico(b: int, c: int, alternativa: str) -> tuple[float, float]:
    """(estatística, p) com a correção de continuidade de Edwards.

    Bilateral: χ² = (|b − c| − 1)²/(b + c) com 1 gl. Unilateral: a raiz com sinal,
    z = (b − c − 1)/√(b + c) para "maior" e (b − c + 1)/√(b + c) para "menor".
    """
    m = b + c
    if alternativa == "two-sided":
        qui2 = max(abs(b - c) - 1, 0) ** 2 / m
        return qui2, float(stats.chi2.sf(qui2, 1))
    if alternativa == "greater":
        z = (b - c - 1) / math.sqrt(m)
        return z, float(stats.norm.sf(z))
    z = (b - c + 1) / math.sqrt(m)
    return z, float(stats.norm.cdf(z))


class TesteMcNemar(TesteBase):
    """Teste de McNemar para duas medidas binárias pareadas (H₀: p₁ = p₂).

    Entrada: duas colunas binárias na mesma linha (ex.: antes e depois) com os mesmos 2 valores
    e o "evento". Linhas com valor ausente em alguma das colunas são descartadas (aviso).
    Tabela de pares [[a, b], [c, d]]: linhas = medida 1 (evento, outro), colunas = medida 2
    (evento, outro); só os pares discordantes b e c entram no teste; p₁ − p₂ = (b − c)/n.
    O card mostra lado a lado o exato (binomial nos discordantes, `scipy.stats.binomtest`) e o
    assintótico com correção de continuidade de Edwards (qui-quadrado bilateral; raiz com sinal
    nas unilaterais). A decisão usa o exato quando b + c < 25 e o assintótico nos demais casos
    (regra de livro-texto; decisão do autor). Efeitos: odds ratio pareada b/c com IC exato
    (Clopper-Pearson sobre b/(b + c)) e diferença de proporções marginais com IC de Wald para
    dados pareados. ICs unilaterais quando H₁ é unilateral. Decisão: p ≤ α.
    """

    id = "mcnemar"
    nome = "McNemar"
    grupo = "Categóricos"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec("coluna1", "Medida 1 (ex.: antes)", "coluna_binaria"),
            ParametroSpec("coluna2", "Medida 2 (ex.: depois)", "coluna_binaria"),
            ParametroSpec("evento", "Evento", "nivel", depende_de="coluna1"),
            ParametroSpec(
                "alternativa",
                "Hipótese alternativa (H₁)",
                "opcao",
                padrao=ALTERNATIVA_PADRAO_MCNEMAR,
                opcoes=list(ALTERNATIVAS_MCNEMAR),
            ),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    def comparacao_inicial(self) -> ComparacaoPValores:
        return ComparacaoPValores(
            titulo_esquerda="Exato (binomial)",
            titulo_direita="Qui-quadrado",
            hipoteses=list(ALTERNATIVAS_MCNEMAR),
            linhas=[(None, None)] * len(ALTERNATIVAS_MCNEMAR),
        )

    @staticmethod
    def tabela_pares(
        df: pd.DataFrame, coluna1: str, coluna2: str, evento: str
    ) -> tuple[list[list[int]], str, int]:
        """([[a, b], [c, d]], valor "outro", linhas descartadas)."""
        validas = df[[coluna1, coluna2]].dropna()
        x = validas[coluna1].map(rotulo_nivel)
        y = validas[coluna2].map(rotulo_nivel)
        outro = next(n for n in niveis_coluna(df[coluna1]) if n != evento)
        tabela = [
            [int(((x == linha) & (y == coluna)).sum()) for coluna in (evento, outro)]
            for linha in (evento, outro)
        ]
        return tabela, outro, len(df) - len(validas)

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        c1, c2, evento = params.get("coluna1"), params.get("coluna2"), params.get("evento")
        erros = erros_coluna(df, c1, "a medida 1", numerica=False)
        erros += erros_coluna(df, c2, "a medida 2", numerica=False)
        if not erros and c1 == c2:
            erros.append("As duas medidas devem ser colunas diferentes.")
        if not erros:
            n1, n2 = niveis_coluna(df[c1]), niveis_coluna(df[c2])
            for coluna, niveis in ((c1, n1), (c2, n2)):
                if len(niveis) != 2:
                    erros.append(
                        f"A coluna '{coluna}' deve ter exatamente 2 valores distintos "
                        f"(tem {len(niveis)})."
                    )
            if not erros and set(n1) != set(n2):
                erros.append(
                    f"As duas medidas devem usar os mesmos 2 valores ('{c1}': {', '.join(n1)}; "
                    f"'{c2}': {', '.join(n2)})."
                )
            if not erros:
                if not evento:
                    erros.append("Escolha o evento.")
                elif str(evento) not in n1:
                    erros.append(f"O valor '{evento}' não aparece na coluna '{c1}'.")
        if not erros:
            tabela, _, _ = self.tabela_pares(df, c1, c2, str(evento))
            (_, b), (c, _) = tabela
            if sum(map(sum, tabela)) == 0:
                erros.append(f"Não há linhas com '{c1}' e '{c2}' preenchidas ao mesmo tempo.")
            elif b + c == 0:
                erros.append(
                    "Não há pares discordantes (b + c = 0): as duas medidas concordam em todas as "
                    "linhas e o McNemar não pode ser calculado."
                )
        erros += erro_opcao(
            params.get("alternativa", ALTERNATIVA_PADRAO_MCNEMAR),
            ALTERNATIVAS_MCNEMAR,
            "Escolha uma hipótese alternativa válida.",
        )
        erros += erro_alfa(params.get("alfa", 0.05))
        return erros

    # ---------------- Execução ----------------
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)

        c1, c2, evento = params["coluna1"], params["coluna2"], str(params["evento"])
        alfa = float(params.get("alfa", 0.05))
        alternativa = ALTERNATIVAS_MCNEMAR[params.get("alternativa", ALTERNATIVA_PADRAO_MCNEMAR)]

        tabela, outro, descartadas = self.tabela_pares(df, c1, c2, evento)
        (a, b), (c, d) = tabela
        n, discordantes = a + b + c + d, b + c

        p_exato = {alt: mcnemar_exato(b, c, alt) for alt in _SIMBOLO_OR}
        assintotico = {alt: mcnemar_assintotico(b, c, alt) for alt in _SIMBOLO_OR}
        usa_exato = discordantes < LIMITE_EXATO
        p_valor = p_exato[alternativa] if usa_exato else assintotico[alternativa][1]

        p1, p2 = (a + b) / n, (a + c) / n
        dif_ic = self._ic_diferenca(b, c, n, alfa, alternativa)
        or_pareada = b / c if c else math.inf
        or_ic = self._ic_odds_ratio(b, c, alfa, alternativa)

        estatisticas = {
            "n": float(n),
            "n_descartadas": float(descartadas),
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "d": float(d),
            "discordantes": float(discordantes),
            "qui2": assintotico["two-sided"][0],
            "p_exato": p_exato[alternativa],
            "p_assintotico": assintotico[alternativa][1],
            "usou_exato": float(usa_exato),
            "p_valor": p_valor,
            "p1": p1,
            "p2": p2,
            "diferenca": p1 - p2,
            "ic_inferior": dif_ic[0],
            "ic_superior": dif_ic[1],
            "odds_ratio_pareada": or_pareada,
            "or_ic_inferior": or_ic[0],
            "or_ic_superior": or_ic[1],
        }

        simbolo, texto = _SIMBOLO_OR[alternativa], _TEXTO_MCNEMAR[alternativa]
        de_quem = f"a proporção de '{evento}' em '{c1}'"
        em_c2 = f"proporção em '{c2}'"
        interpretacao = interpretar(
            p_valor,
            alfa,
            h0="p₁ = p₂",
            h1=f"p₁ {simbolo} p₂",
            conclusao_rejeita=f"Há evidência estatística de que {de_quem} é {texto} {em_c2}.",
            conclusao_nao_rejeita=(
                f"Não há evidência suficiente de que {de_quem} seja {texto} {em_c2}."
            ),
        )

        metodo = (
            f"Exato (b + c = {discordantes} < {LIMITE_EXATO})"
            if usa_exato
            else f"Qui-quadrado com Edwards (b + c = {discordantes} ≥ {LIMITE_EXATO})"
        )
        avisos = []
        if descartadas:
            avisos.append(
                f"{descartadas} linha(s) com valor ausente em '{c1}' ou '{c2}' foram descartadas."
            )
        if usa_exato:
            avisos.append(
                f"Poucos pares discordantes (b + c = {discordantes}): a decisão usa o teste exato; "
                "o valor do qui-quadrado no card é só comparativo."
            )
        if c == 0:
            avisos.append("Nenhum par do tipo c: a odds ratio pareada (b/c) vai a +∞.")

        confianca = f"{(1 - alfa) * 100:.0f}%"
        tipo_ic = "" if alternativa == "two-sided" else " (unilateral)"
        resumo = [
            ("Pares completos (n)", f"{n}"),
            ("Evento", f"'{evento}'"),
            (f"Pares discordantes b ('{evento}' → '{outro}')", f"{b}"),
            (f"Pares discordantes c ('{outro}' → '{evento}')", f"{c}"),
            ("Método da decisão", metodo),
            ("p-valor exato (binomial)", formatar_p_valor(p_exato[alternativa])),
            ("Qui-quadrado de McNemar (Edwards)", formatar_numero(assintotico["two-sided"][0])),
            ("p-valor assintótico", formatar_p_valor(assintotico[alternativa][1])),
            (
                f"Proporção de '{evento}' em '{c1}' (p̂₁)",
                f"{formatar_numero(p1)} ({_percentual(p1)})",
            ),
            (
                f"Proporção de '{evento}' em '{c2}' (p̂₂)",
                f"{formatar_numero(p2)} ({_percentual(p2)})",
            ),
            ("Diferença (p̂₁ − p̂₂)", formatar_numero(p1 - p2)),
            (
                f"IC {confianca} para p₁ − p₂{tipo_ic}",
                f"[{formatar_numero(dif_ic[0])}; {formatar_numero(dif_ic[1])}]",
            ),
            ("Odds ratio pareada (b/c)", formatar_numero(or_pareada)),
            (
                f"IC {confianca} exato para a odds ratio pareada{tipo_ic}",
                f"[{formatar_numero(or_ic[0])}; {formatar_numero(or_ic[1])}]",
            ),
        ]
        pares = pd.DataFrame(tabela, index=[evento, outro], columns=[evento, outro])
        pares.index.name, pares.columns.name = c1, c2

        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=p_valor,
            alfa=alfa,
            decisao=decidir(p_valor, alfa),
            interpretacao=interpretacao,
            tabelas={
                "Resumo": pd.DataFrame(resumo, columns=["Medida", "Valor"]),
                "Tabela de pares": QuiQuadrado._tabela_com_totais(pares),
            },
            figuras=[
                barras(
                    [(c1, p1), (c2, p2)],
                    f"Proporção de '{evento}' em '{c1}' e '{c2}'",
                    "Proporção",
                    maximo=1.0,
                    percentual=True,
                )
            ],
            avisos=avisos,
            comparacao=ComparacaoPValores(
                titulo_esquerda="Exato (binomial)",
                titulo_direita="Qui-quadrado",
                hipoteses=list(ALTERNATIVAS_MCNEMAR),
                linhas=[
                    (p_exato[alt], assintotico[alt][1]) for alt in ALTERNATIVAS_MCNEMAR.values()
                ],
            ),
        )

    @staticmethod
    def _ic_diferenca(b: int, c: int, n: int, alfa: float, alternativa: str) -> tuple[float, float]:
        """Wald para dados pareados: Var(p̂₁ − p̂₂) = [(b + c) − (b − c)²/n] / n²."""
        dif = (b - c) / n
        ep = math.sqrt(max((b + c) - (b - c) ** 2 / n, 0.0)) / n
        if alternativa == "two-sided":
            z = float(stats.norm.ppf(1 - alfa / 2))
            return dif - z * ep, dif + z * ep
        z = float(stats.norm.ppf(1 - alfa))
        return (dif - z * ep, 1.0) if alternativa == "greater" else (-1.0, dif + z * ep)

    @staticmethod
    def _ic_odds_ratio(b: int, c: int, alfa: float, alternativa: str) -> tuple[float, float]:
        """IC exato condicional para b/c: Clopper-Pearson para θ = b/(b + c), OR = θ/(1 − θ)."""
        cauda = alfa / 2 if alternativa == "two-sided" else alfa

        def razao(theta: float) -> float:
            return math.inf if theta >= 1 else theta / (1 - theta)

        baixo = float(stats.beta.ppf(cauda, b, c + 1)) if b > 0 else 0.0
        alto = float(stats.beta.ppf(1 - cauda, b + 1, c)) if c > 0 else 1.0
        if alternativa == "greater":
            return razao(baixo), math.inf
        if alternativa == "less":
            return 0.0, razao(alto)
        return razao(baixo), razao(alto)
