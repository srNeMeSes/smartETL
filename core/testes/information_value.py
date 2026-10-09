"""Information Value (IV) e Weight of Evidence (WoE) — grupo Relação.

Mede o poder de cada variável X para separar as duas classes de uma resposta binária y (ex.:
inadimplente Sim/Não). Não é um teste de hipótese: não há p-valor, α nem decisão.

Para cada categoria de X (Siddiqi, *Credit Risk Scorecards*, 2006):
    % eventos = eventos da categoria / total de eventos
    % não eventos = não eventos da categoria / total de não eventos
    WoE = ln(% não eventos / % eventos)        (> 0: a categoria tem menos eventos que a média)
    IV da categoria = (% não eventos − % eventos) × WoE  (sempre ≥ 0)
    IV da variável = soma dos IV das categorias
"""

import math

import numpy as np
import pandas as pd

from core.base import ErroValidacao, GrupoFiguras, ParametroSpec, ResultadoTeste, TesteBase
from core.figuras import barras
from core.interpretacao import formatar_numero
from core.tipos import e_numerica, niveis_coluna, ordenar_niveis, rotulo_nivel
from core.validacao import erro_opcao, erros_coluna

FAIXAS = ("5", "10", "20")
FAIXAS_PADRAO = "10"
AUSENTE = "(ausente)"
CORRECAO_ZERO = 0.5  # somado a eventos e não eventos de uma categoria com contagem zero
MAX_CATEGORIAS = 50  # acima disso o IV tende a ser inflado (aviso)
# Faixas de Siddiqi para o IV: [0; 0,02) [0,02; 0,1) [0,1; 0,3) [0,3; 0,5] (0,5; ∞)
FORCA_IV = ((0.02, "Sem poder preditivo"), (0.1, "Fraco"), (0.3, "Médio"), (0.5, "Forte"))
FORCA_SUSPEITA = "Muito forte (verificar vazamento)"
TABELA_GERAL = "Information Value por variável"
DICAS_CATEGORIA = {
    "% dos eventos": "Parte do total de eventos que cai nesta categoria.",
    "% dos não eventos": "Parte do total de não eventos que cai nesta categoria.",
    "WoE": "Weight of Evidence = ln(% dos não eventos / % dos eventos). Positivo: a categoria "
    "tem menos eventos que a média; negativo: mais.",
    "IV": "(% dos não eventos − % dos eventos) × WoE: quanto a categoria contribui para o IV.",
}


def classificar_iv(iv: float) -> str:
    """Poder preditivo pelas faixas de Siddiqi."""
    for limite, rotulo in FORCA_IV[:-1]:
        if iv < limite:
            return rotulo
    return FORCA_IV[-1][1] if iv <= FORCA_IV[-1][0] else FORCA_SUSPEITA  # 0,5 ainda é "Forte"


def _numero_faixa(valor: float, inteiros: bool) -> str:
    return formatar_numero(valor, 0 if inteiros else 2)


def categorizar(serie: pd.Series, faixas: int) -> tuple[pd.Series, list[str], bool]:
    """(rótulo da categoria de cada linha, categorias na ordem natural, se virou faixas).

    Numérica com mais valores distintos que `faixas`: faixas por quantis com quantidades
    parecidas de casos (`pandas.qcut`, faixas repetidas juntadas), cada uma rotulada pelo menor e
    pelo maior valor observado nela ("18 – 30"). Numérica com poucos valores e categóricas: cada
    valor é uma categoria. Ausentes formam a categoria "(ausente)", no fim.
    """
    validos = serie.dropna()
    rotulos = pd.Series(AUSENTE, index=serie.index, dtype=object)
    em_faixas = e_numerica(serie) and validos.nunique() > faixas
    if em_faixas:
        x = validos.astype(float)
        codigos = pd.qcut(x, q=faixas, labels=False, duplicates="drop")
        inteiros = bool((x % 1 == 0).all())
        ordem = []
        for codigo in sorted(codigos.unique()):
            dentro = x[codigos == codigo]
            baixo, alto = dentro.min(), dentro.max()
            texto = _numero_faixa(baixo, inteiros)
            if alto != baixo:
                texto += f" – {_numero_faixa(alto, inteiros)}"
            rotulos.loc[dentro.index] = texto
            ordem.append(texto)
    else:
        rotulos.loc[validos.index] = validos.map(rotulo_nivel)
        ordem = [rotulo_nivel(v) for v in ordenar_niveis(list(pd.unique(validos)))]
    if len(validos) < len(serie):
        ordem.append(AUSENTE)
    return rotulos, ordem, em_faixas


def tabela_woe(rotulos: pd.Series, y: np.ndarray, ordem: list[str]) -> pd.DataFrame:
    """WoE e IV de cada categoria (números, na ordem natural); `y` é 1 no evento, 0 no outro.

    Categoria sem eventos ou sem não eventos recebe +0,5 nas duas contagens (o WoE seria
    infinito); a coluna "corrigida" marca essas categorias.
    """
    tabela = (
        pd.DataFrame({"categoria": rotulos.to_numpy(), "y": y})
        .groupby("categoria", sort=False)["y"]
        .agg(n="size", eventos="sum")
        .reindex(ordem)
    )
    tabela["nao_eventos"] = tabela["n"] - tabela["eventos"]
    corrigida = (tabela["eventos"] == 0) | (tabela["nao_eventos"] == 0)
    eventos = tabela["eventos"] + CORRECAO_ZERO * corrigida
    nao_eventos = tabela["nao_eventos"] + CORRECAO_ZERO * corrigida
    tabela["pct_eventos"] = eventos / eventos.sum()
    tabela["pct_nao_eventos"] = nao_eventos / nao_eventos.sum()
    tabela["woe"] = np.log(tabela["pct_nao_eventos"] / tabela["pct_eventos"])
    tabela["iv"] = (tabela["pct_nao_eventos"] - tabela["pct_eventos"]) * tabela["woe"]
    tabela["corrigida"] = corrigida
    return tabela


def _percentual(p: float) -> str:
    return f"{formatar_numero(p * 100, 1)}%"


class TesteInformationValue(TesteBase):
    """Information Value (IV) de várias variáveis X para uma resposta binária y.

    Entrada: y com exatamente 2 valores + o "Evento" e as colunas X (numéricas e categóricas).
    Linhas com y ausente são descartadas (aviso); X ausente vira a categoria "(ausente)".
    Numéricas com mais valores distintos que o número de faixas viram faixas por quantis (5, 10
    ou 20; padrão 10, decisão do autor). Contagem zero: +0,5 nas duas contagens da categoria
    (decisão do autor). Resultado: tabela geral (Variável | IV | Poder preditivo, ordenada pelo
    maior IV) e uma tabela por variável com as categorias ordenadas pelo maior IV, com a linha
    Total. Força pelas faixas de Siddiqi. Sem p-valor e sem destaque de decisão (decisão do
    autor). Gráficos: IV por variável e, numa lista, o WoE de cada categoria.
    """

    id = "information_value"
    nome = "Information Value (IV)"
    grupo = "Relação"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec(
                "y",
                "Variável resposta (y)",
                "coluna_binaria",
                ajuda="Só colunas com exatamente 2 valores (ex.: sim/não).",
            ),
            ParametroSpec(
                "evento", "Evento (o valor que conta como sucesso)", "nivel", depende_de="y"
            ),
            ParametroSpec("preditores", "Variáveis X", "preditores", depende_de="y"),
            ParametroSpec(
                "faixas",
                "Faixas das variáveis quantitativas",
                "opcao",
                padrao=FAIXAS_PADRAO,
                opcoes=list(FAIXAS),
                ajuda=(
                    "Faixas por quantis, com quantidades parecidas de casos (ex.: Idade → 18 – 30)."
                ),
            ),
        ]

    # ---------------- Validação ----------------
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        nome_y, evento = params.get("y"), params.get("evento")
        preditores = list(params.get("preditores") or [])
        erros = erros_coluna(df, nome_y, "a variável resposta (y)", numerica=False)
        if not erros:
            niveis = niveis_coluna(df[nome_y])
            if len(niveis) != 2:
                erros.append(
                    f"A variável resposta '{nome_y}' deve ter exatamente 2 valores "
                    f"(tem {len(niveis)})."
                )
            elif not evento or str(evento) not in niveis:
                erros.append("Escolha qual valor da variável resposta é o evento.")
        if not preditores:
            erros.append("Selecione ao menos uma variável X.")
        for nome in preditores:
            if nome not in df.columns:
                erros.append(f"A coluna '{nome}' não existe no arquivo.")
        if nome_y in preditores:
            erros.append("A variável resposta não pode ser também uma variável X.")
        if len(set(preditores)) != len(preditores):
            erros.append("As variáveis X devem ser colunas diferentes.")
        erros += erro_opcao(
            params.get("faixas", FAIXAS_PADRAO), FAIXAS, "Escolha o número de faixas."
        )
        return erros

    # ---------------- Execução ----------------
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        erros = self.validar(df, params)
        if erros:
            raise ErroValidacao(erros)
        nome_y, evento = params["y"], str(params["evento"])
        preditores = list(params["preditores"])
        faixas = int(params.get("faixas", FAIXAS_PADRAO))

        linhas = df[df[nome_y].notna()]
        descartadas = len(df) - len(linhas)
        y = (linhas[nome_y].map(rotulo_nivel) == evento).to_numpy(dtype=float)
        eventos = int(y.sum())
        if eventos in (0, len(y)):
            raise ErroValidacao(
                f"Nas linhas com '{nome_y}' preenchida só há uma classe: é preciso haver casos "
                "das duas classes."
            )

        avisos = []
        if descartadas:
            avisos.append(f"{descartadas} linha(s) com '{nome_y}' ausente foram descartadas.")
        resultados = {}  # variável → (tabela numérica, faixas?)
        for nome in preditores:
            rotulos, ordem, em_faixas = categorizar(linhas[nome], faixas)
            tabela = tabela_woe(rotulos, y, ordem)
            resultados[nome] = (tabela, em_faixas)
            avisos += self._avisos_variavel(nome, tabela)

        ivs = {nome: float(t["iv"].sum()) for nome, (t, _) in resultados.items()}
        ordem_iv = sorted(ivs, key=lambda nome: ivs[nome], reverse=True)
        estatisticas: dict[str, float] = {
            "n": float(len(y)),
            "eventos": float(eventos),
            "n_descartadas": float(descartadas),
        }
        estatisticas |= {f"iv_{nome}": ivs[nome] for nome in ordem_iv}

        geral = pd.DataFrame(
            [
                (
                    nome,
                    formatar_numero(ivs[nome]),
                    classificar_iv(ivs[nome]),
                    len(resultados[nome][0]),
                )
                for nome in ordem_iv
            ],
            columns=["Variável", "IV", "Poder preditivo", "Categorias"],
        )
        geral.attrs["dicas"] = {
            "Poder preditivo": "Faixas de Siddiqi: < 0,02 sem poder; 0,02 a 0,1 fraco; 0,1 a 0,3 "
            "médio; 0,3 a 0,5 forte; acima de 0,5, suspeito (possível vazamento de informação)."
        }
        tabelas = {TABELA_GERAL: geral}
        for nome in ordem_iv:
            tabela, em_faixas = resultados[nome]
            tabelas[f"IV de '{nome}'" + (f" ({len(tabela)} faixas)" if em_faixas else "")] = (
                self._tabela_categorias(tabela)
            )

        return ResultadoTeste(
            teste_id=self.id,
            estatisticas=estatisticas,
            p_valor=None,
            alfa=math.nan,
            decisao="",
            interpretacao=self._interpretacao(ivs, ordem_iv, nome_y, evento, len(y), eventos),
            tabelas=tabelas,
            figuras=self._figuras(ivs, ordem_iv, resultados),
            avisos=avisos,
        )

    # ---------------- Auxiliares ----------------
    @staticmethod
    def _avisos_variavel(nome: str, tabela: pd.DataFrame) -> list[str]:
        avisos = []
        if len(tabela) == 1:
            avisos.append(f"A variável '{nome}' tem uma única categoria: IV = 0.")
        if len(tabela) > MAX_CATEGORIAS:
            avisos.append(
                f"A variável '{nome}' tem {len(tabela)} categorias: com tantas categorias o IV "
                "tende a ser inflado. Considere agrupá-las."
            )
        corrigidas = [str(c) for c in tabela.index[tabela["corrigida"]]]
        if corrigidas:
            lista = ", ".join(f"'{c}'" for c in corrigidas[:5])
            if len(corrigidas) > 5:
                lista += f" e mais {len(corrigidas) - 5}"
            avisos.append(
                f"Em '{nome}', a(s) categoria(s) {lista} não têm eventos ou não eventos: somado "
                f"{formatar_numero(CORRECAO_ZERO, 1)} às duas contagens para o WoE ser finito."
            )
        iv = float(tabela["iv"].sum())
        if iv > 0.5:
            avisos.append(
                f"O IV de '{nome}' ({formatar_numero(iv)}) passa de 0,5: poder suspeito. Verifique "
                "se a variável não carrega informação do próprio resultado (vazamento)."
            )
        return avisos

    @staticmethod
    def _tabela_categorias(tabela: pd.DataFrame) -> pd.DataFrame:
        """Categorias ordenadas pelo maior IV, com a linha Total em negrito."""
        ordenada = tabela.sort_values("iv", ascending=False, kind="stable")
        linhas = [
            (
                str(categoria),
                int(t.n),
                int(t.eventos),
                int(t.nao_eventos),
                _percentual(t.pct_eventos),
                _percentual(t.pct_nao_eventos),
                formatar_numero(t.woe),
                formatar_numero(t.iv),
            )
            for categoria, t in ordenada.iterrows()
        ]
        linhas.append(
            (
                "Total",
                int(tabela["n"].sum()),
                int(tabela["eventos"].sum()),
                int(tabela["nao_eventos"].sum()),
                "100,0%",
                "100,0%",
                "",
                formatar_numero(float(tabela["iv"].sum())),
            )
        )
        saida = pd.DataFrame(
            linhas,
            columns=[
                "Categoria",
                "n",
                "Eventos",
                "Não eventos",
                "% dos eventos",
                "% dos não eventos",
                "WoE",
                "IV",
            ],
        )
        saida.attrs["dicas"] = DICAS_CATEGORIA
        saida.attrs["destaques"] = [len(linhas) - 1]
        return saida

    @staticmethod
    def _interpretacao(
        ivs: dict[str, float], ordem: list[str], nome_y: str, evento: str, n: int, eventos: int
    ) -> str:
        melhor = ordem[0]
        texto = (
            f"Information Value de {len(ordem)} variável(is) para prever '{nome_y}' = '{evento}' "
            f"({n} casos, {eventos} eventos). A de maior poder é '{melhor}' "
            f"(IV = {formatar_numero(ivs[melhor])}: {classificar_iv(ivs[melhor]).lower()})."
        )
        uteis = [nome for nome in ordem if ivs[nome] >= FORCA_IV[0][0]]
        if not uteis:
            texto += " Nenhuma variável tem poder preditivo (IV < 0,02)."
        elif len(uteis) < len(ordem):
            texto += f" {len(ordem) - len(uteis)} variável(is) têm IV abaixo de 0,02 (sem poder)."
        return texto

    @staticmethod
    def _figuras(ivs: dict[str, float], ordem: list[str], resultados: dict) -> list:
        geral = barras(
            [(nome, ivs[nome]) for nome in ordem],
            "Information Value por variável",
            "IV",
            referencias=[("Médio (0,1)", 0.1, "tracejado"), ("Forte (0,3)", 0.3, "tracejado")],
        )
        woe = GrupoFiguras(
            "Variável",
            {
                nome: barras(
                    [(str(c), float(w)) for c, w in resultados[nome][0]["woe"].items()],
                    f"WoE por categoria de '{nome}'",
                    "WoE",
                )
                for nome in ordem
            },
            ordem[0],
        )
        return [geral, woe]
