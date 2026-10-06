"""Testes para variáveis categóricas."""

import math

import numpy as np
import pandas as pd
from scipy import stats

from core.base import ErroValidacao, ParametroSpec, ResultadoTeste, TesteBase
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
