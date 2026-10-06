"""Qui-quadrado (independência e aderência): checklist da seção 9 do CLAUDE.md.

Fontes dos valores de referência (independentes do wrapper, que usa scipy):
- χ² = Σ (O − E)²/E com E = (total da linha × total da coluna)/n, por numpy;
- correção de Yates: Σ (|O − E| − min(0,5; |O − E|))²/E;
- p-valores por fórmulas fechadas da cauda qui-quadrado: gl = 1 → erfc(√(χ²/2));
  gl = 2 → exp(−χ²/2); e statsmodels (`Table.test_nominal_association`) para tabelas maiores;
- V de Cramér = √(χ²/(n·(min(r, c) − 1))); w de Cohen = √(χ²/n).
"""

import math

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.contingency_tables import Table

from core.base import ErroValidacao
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.categoricos import ADERENCIA, INDEPENDENCIA, QuiQuadrado

REL = 1e-6


@pytest.fixture
def teste():
    return QuiQuadrado()


def _df_de_tabela(tabela, nomes_linhas, nomes_colunas, c1="turno", c2="nota"):
    linhas = []
    for nome_l, contagens in zip(nomes_linhas, tabela, strict=True):
        for nome_c, k in zip(nomes_colunas, contagens, strict=True):
            linhas += [(nome_l, nome_c)] * k
    return pd.DataFrame(linhas, columns=[c1, c2])


def _params(**extra):
    return {
        "modo": INDEPENDENCIA,
        "coluna1": "turno",
        "coluna2": "nota",
        "correcao": False,
        "alfa": 0.05,
    } | extra


def _qui2_manual(obs, yates=False):
    obs = np.asarray(obs, dtype=float)
    esperadas = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / obs.sum()
    diferenca = np.abs(obs - esperadas)
    if yates:
        diferenca = diferenca - np.minimum(0.5, diferenca)
    return float((diferenca**2 / esperadas).sum()), esperadas


TABELA_2X3 = [[20, 15, 5], [10, 15, 25]]  # gl = 2
TABELA_2X2 = [[18, 7], [9, 16]]  # gl = 1


# ---------------------------------------------------------------------------
# Independência: valores de referência
# ---------------------------------------------------------------------------


def test_independencia_2x3_contra_formula(teste):
    df = _df_de_tabela(TABELA_2X3, ["manhã", "noite"], ["A", "B", "C"])
    r = teste.executar(df, _params())
    qui2, _ = _qui2_manual(TABELA_2X3)
    e = r.estatisticas
    assert e["qui2"] == pytest.approx(qui2, rel=REL)
    assert e["gl"] == 2
    assert r.p_valor == pytest.approx(math.exp(-qui2 / 2), rel=REL)
    assert e["cramer_v"] == pytest.approx(math.sqrt(qui2 / (90 * 1)), rel=REL)
    assert e["n"] == 90


@pytest.mark.parametrize("yates", [False, True])
def test_independencia_2x2_com_e_sem_yates(teste, yates):
    df = _df_de_tabela(TABELA_2X2, ["manhã", "noite"], ["aprovado", "reprovado"])
    r = teste.executar(df, _params(correcao=yates))
    qui2, _ = _qui2_manual(TABELA_2X2, yates=yates)
    qui2_sem, _ = _qui2_manual(TABELA_2X2)
    assert r.estatisticas["qui2"] == pytest.approx(qui2, rel=REL)
    assert r.p_valor == pytest.approx(math.erfc(math.sqrt(qui2 / 2)), rel=REL)
    # V de Cramér sempre com a estatística sem correção.
    assert r.estatisticas["cramer_v"] == pytest.approx(math.sqrt(qui2_sem / 50), rel=REL)
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Correção de Yates"] == ("Sim" if yates else "Não")


def test_independencia_3x4_contra_statsmodels(teste):
    tabela = [[12, 8, 15, 5], [7, 14, 9, 10], [11, 6, 8, 15]]
    df = _df_de_tabela(tabela, ["r1", "r2", "r3"], ["a", "b", "c", "d"])
    r = teste.executar(df, _params())
    ref = Table(np.array(tabela)).test_nominal_association()
    assert r.estatisticas["qui2"] == pytest.approx(ref.statistic, rel=REL)
    assert r.estatisticas["gl"] == ref.df == 6
    assert r.p_valor == pytest.approx(ref.pvalue, rel=REL)
    qui2, _ = _qui2_manual(tabela)
    assert r.estatisticas["cramer_v"] == pytest.approx(math.sqrt(qui2 / (r.estatisticas["n"] * 2)))


def test_tabelas_observada_e_esperada(teste):
    df = _df_de_tabela(TABELA_2X3, ["noite", "manhã"][::-1], ["A", "B", "C"])
    r = teste.executar(df.sample(frac=1, random_state=1), _params())
    observadas = r.tabelas["Frequências observadas"]
    assert list(observadas.columns) == ["turno \\ nota", "A", "B", "C", "Total"]
    assert observadas.values.tolist() == [
        ["manhã", 20, 15, 5, 40],
        ["noite", 10, 15, 25, 50],
        ["Total", 30, 30, 30, 90],
    ]
    _, esperadas = _qui2_manual(TABELA_2X3)
    tabela_e = r.tabelas["Frequências esperadas"]
    assert tabela_e.iloc[0, 1:].tolist() == [f"{v:.2f}".replace(".", ",") for v in esperadas[0]]
    (figura,) = r.figuras
    assert figura.tipo == "barras_agrupadas"
    assert figura.dados["series"] == ["A", "B", "C"]
    assert [g["rotulo"] for g in figura.dados["grupos"]] == ["manhã", "noite"]
    assert figura.dados["grupos"][0]["valores"] == pytest.approx([0.5, 0.375, 0.125])


def test_independencia_decisao_e_interpretacao(teste):
    df = _df_de_tabela(TABELA_2X3, ["manhã", "noite"], ["A", "B", "C"])
    r = teste.executar(df, _params())
    assert r.decisao == REJEITA_H0
    assert "rejeita-se H₀ ('turno' e 'nota' são independentes)" in r.interpretacao
    assert "em favor de H₁ ('turno' e 'nota' estão associadas)" in r.interpretacao
    assert "Há evidência estatística de associação entre 'turno' e 'nota'." in r.interpretacao
    fraca = _df_de_tabela([[10, 11], [11, 10]], ["m", "n"], ["a", "b"])
    r = teste.executar(fraca, _params())
    assert r.decisao == NAO_REJEITA_H0
    assert "Não há evidência suficiente de associação" in r.interpretacao
    assert r.comparacao is None  # sem card de comparação


def test_aviso_esperadas_baixas_e_fisher(teste):
    df = _df_de_tabela([[4, 1], [2, 3]], ["x", "y"], ["p", "q"])
    r = teste.executar(df, _params())
    (aviso,) = r.avisos
    assert "4 de 4 célula(s) (100,0%) têm frequência esperada menor que 5" in aviso
    assert "prefira o Teste exato de Fisher" in aviso
    df = _df_de_tabela([[4, 1, 2], [2, 3, 6]], ["x", "y"], ["p", "q", "r"])
    assert "agrupar categorias" in teste.executar(df, _params()).avisos[0]


def test_yates_ignorada_fora_do_2x2(teste):
    df = _df_de_tabela(TABELA_2X3, ["manhã", "noite"], ["A", "B", "C"])
    r = teste.executar(df, _params(correcao=True))
    qui2, _ = _qui2_manual(TABELA_2X3)
    assert r.estatisticas["qui2"] == pytest.approx(qui2, rel=REL)
    assert any("só se aplica a tabelas 2×2" in a for a in r.avisos)


def test_linhas_incompletas_descartadas(teste):
    df = _df_de_tabela(TABELA_2X3, ["manhã", "noite"], ["A", "B", "C"])
    df = pd.concat([df, pd.DataFrame({"turno": [None, "manhã"], "nota": ["A", None]})])
    r = teste.executar(df, _params())
    assert r.estatisticas["n"] == 90
    assert "2 linha(s) com valor ausente em 'turno' ou 'nota' foram descartadas." in r.avisos


# ---------------------------------------------------------------------------
# Aderência
# ---------------------------------------------------------------------------


def _aderencia(contagens, nomes):
    return pd.DataFrame(
        {"cor": [n for n, k in zip(nomes, contagens, strict=True) for _ in range(k)]}
    )


@pytest.mark.parametrize(
    ("contagens", "nomes"), [([30, 18, 12], ["azul", "verde", "rosa"]), ([28, 12], ["sim", "não"])]
)
def test_aderencia_contra_formula(teste, contagens, nomes):
    r = teste.executar(
        _aderencia(contagens, nomes), _params(modo=ADERENCIA, coluna1="cor", coluna2=None)
    )
    n, k = sum(contagens), len(contagens)
    qui2 = sum((o - n / k) ** 2 / (n / k) for o in contagens)
    p = math.exp(-qui2 / 2) if k == 3 else math.erfc(math.sqrt(qui2 / 2))
    e = r.estatisticas
    assert e["qui2"] == pytest.approx(qui2, rel=REL)
    assert e["gl"] == k - 1
    assert r.p_valor == pytest.approx(p, rel=REL)
    assert e["w_cohen"] == pytest.approx(math.sqrt(qui2 / n), rel=REL)
    assert e["esperada"] == pytest.approx(n / k)


def test_aderencia_saida(teste):
    r = teste.executar(
        _aderencia([30, 18, 12], ["azul", "verde", "rosa"]),
        _params(modo=ADERENCIA, coluna1="cor", coluna2=None),
    )
    assert r.decisao == REJEITA_H0
    assert "rejeita-se H₀ (as 3 categorias de 'cor' têm a mesma proporção)" in r.interpretacao
    frequencias = r.tabelas["Frequências"]
    assert frequencias["Categoria"].tolist() == ["azul", "rosa", "verde"]
    assert frequencias["Observada"].tolist() == [30, 12, 18]
    assert frequencias["Esperada"].tolist() == ["20,00"] * 3
    (figura,) = r.figuras
    assert figura.tipo == "barras"
    assert figura.dados["referencias"][0]["valor"] == pytest.approx(1 / 3)
    assert r.comparacao is None


def test_aderencia_avisos(teste):
    df = _aderencia([3, 2, 1], ["a", "b", "c"])
    df.loc[len(df)] = [None]
    r = teste.executar(df, _params(modo=ADERENCIA, coluna1="cor", coluna2="cor", correcao=True))
    textos = " ".join(r.avisos)
    assert "1 valor(es) ausente(s) em 'cor' foram ignorados." in textos
    assert "variável 2 não é usada" in textos
    assert "Yates não se aplica" in textos
    assert "frequência esperada menor que 5" in textos


# ---------------------------------------------------------------------------
# Entradas inválidas
# ---------------------------------------------------------------------------


@pytest.fixture
def df():
    return _df_de_tabela(TABELA_2X3, ["manhã", "noite"], ["A", "B", "C"])


@pytest.mark.parametrize(
    ("params", "mensagem"),
    [
        ({"modo": "x"}, "Escolha o tipo de teste."),
        ({"coluna1": None}, "Selecione a variável 1."),
        ({"coluna2": None}, "Selecione a variável 2."),
        ({"coluna2": "nao_existe"}, "A coluna 'nao_existe' não existe no arquivo."),
        ({"coluna2": "turno"}, "As duas variáveis devem ser colunas diferentes."),
        ({"alfa": 0}, "O nível de significância (α) deve estar entre 0 e 1."),
    ],
)
def test_validacao(teste, df, params, mensagem):
    assert mensagem in teste.validar(df, _params(**params))
    with pytest.raises(ErroValidacao):
        teste.executar(df, _params(**params))


def test_validacao_categorias_insuficientes(teste):
    df = pd.DataFrame({"turno": ["m", "m", "m"], "nota": ["a", "b", "a"]})
    assert "A coluna 'turno' precisa de ao menos 2 categorias com dados válidos (tem 1)." in (
        teste.validar(df, _params())
    )
    erros = teste.validar(df, _params(modo=ADERENCIA, coluna2=None))
    assert "A coluna 'turno' precisa de ao menos 2 categorias (tem 1)." in erros


def test_aderencia_nao_exige_variavel_2(teste, df):
    assert teste.validar(df, _params(modo=ADERENCIA, coluna2=None)) == []


def test_formulario(teste):
    specs = teste.parametros()
    assert [(s.nome, s.tipo) for s in specs] == [
        ("modo", "opcao"),
        ("coluna1", "coluna_categorica"),
        ("coluna2", "coluna_categorica"),
        ("correcao", "booleano"),
        ("alfa", "alfa"),
    ]
    assert specs[0].opcoes == [INDEPENDENCIA, ADERENCIA] and specs[0].padrao == INDEPENDENCIA
    assert specs[2].obrigatorio is False
    assert specs[3].padrao is False
    assert teste.comparacao_inicial() is None
