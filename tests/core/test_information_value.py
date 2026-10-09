"""Information Value (IV) e WoE (core/testes/information_value.py).

Referências: o exemplo de 3 categorias é calculado à mão (fórmulas de Siddiqi, *Credit Risk
Scorecards*, 2006: WoE = ln(%não eventos / %eventos), IV = Σ (%não eventos − %eventos)·WoE);
as demais variáveis são conferidas contra uma implementação independente com
`pandas.crosstab`. As faixas de 1..100 em decis são exatas (10 valores por faixa).
"""

import math

import numpy as np
import pandas as pd
import pytest

from core.base import ErroValidacao, GrupoFiguras
from core.testes.information_value import (
    AUSENTE,
    FORCA_SUSPEITA,
    TABELA_GERAL,
    TesteInformationValue,
    categorizar,
    classificar_iv,
)

REL = 1e-12


def _base_manual() -> pd.DataFrame:
    """Categoria A: 10 eventos e 40 não eventos; B: 20 e 30; C: 30 e 10 (60 e 80 no total)."""
    linhas = []
    for categoria, eventos, nao in (("A", 10, 40), ("B", 20, 30), ("C", 30, 10)):
        linhas += [(categoria, "Sim")] * eventos + [(categoria, "Não")] * nao
    return pd.DataFrame(linhas, columns=["cat", "y"])


def _params(**extra):
    return {"y": "y", "evento": "Sim", "preditores": ["cat"], "faixas": "10"} | extra


def _iv_independente(x: pd.Series, y_evento: pd.Series) -> float:
    """IV sem correção, por crosstab (categorias já definidas em x)."""
    tabela = pd.crosstab(x, y_evento)
    pct_e = tabela[True] / tabela[True].sum()
    pct_ne = tabela[False] / tabela[False].sum()
    return float(((pct_ne - pct_e) * np.log(pct_ne / pct_e)).sum())


# ---------------------------------------------------------------------------
# Valores de referência
# ---------------------------------------------------------------------------


def test_exemplo_calculado_a_mao():
    r = TesteInformationValue().executar(_base_manual(), _params())
    esperado = {}
    for categoria, eventos, nao in (("A", 10, 40), ("B", 20, 30), ("C", 30, 10)):
        pe, pne = eventos / 60, nao / 80
        esperado[categoria] = (math.log(pne / pe), (pne - pe) * math.log(pne / pe))
    iv_total = sum(iv for _, iv in esperado.values())
    assert r.estatisticas["iv_cat"] == pytest.approx(iv_total, rel=REL)
    # À mão: A = (1/2 − 1/6)·ln 3 = 0,36620; B = (3/8 − 1/3)·ln(9/8) = 0,00491;
    # C = (1/8 − 1/2)·ln(1/4) = 0,51986.
    assert iv_total == pytest.approx(0.36620410 + 0.00490763 + 0.51986039, rel=1e-6)
    tabela = r.tabelas["IV de 'cat'"]
    # Ordenada pelo maior IV: C (0,5199), A (0,3662), B (0,0049); Total por último.
    assert tabela["Categoria"].tolist() == ["C", "A", "B", "Total"]
    for _, linha in tabela.iloc[:-1].iterrows():
        woe, iv = esperado[linha["Categoria"]]
        assert linha["WoE"] == f"{woe:.4f}".replace(".", ",")
        assert linha["IV"] == f"{iv:.4f}".replace(".", ",")
    total = tabela.iloc[-1]
    assert (total["n"], total["Eventos"], total["Não eventos"]) == (140, 60, 80)
    assert tabela.attrs["destaques"] == [3]
    assert tabela.loc[tabela["Categoria"] == "A", "% dos eventos"].item() == "16,7%"
    assert tabela.loc[tabela["Categoria"] == "A", "% dos não eventos"].item() == "50,0%"


def test_varias_variaveis_contra_crosstab_e_ordem_geral():
    rng = np.random.default_rng(7)
    n = 400
    forte = rng.choice(["a", "b", "c", "d"], n)
    risco = pd.Series(forte).map({"a": 0.1, "b": 0.3, "c": 0.5, "d": 0.8}).to_numpy()
    y = np.where(rng.uniform(size=n) < risco, "Sim", "Não")
    df = pd.DataFrame({"y": y, "forte": forte, "ruido": rng.choice(["u", "v"], n)})
    r = TesteInformationValue().executar(df, _params(preditores=["ruido", "forte"]))
    evento = pd.Series(y == "Sim")
    for nome in ("forte", "ruido"):
        assert r.estatisticas[f"iv_{nome}"] == pytest.approx(
            _iv_independente(df[nome], evento), rel=1e-10
        )
    geral = r.tabelas[TABELA_GERAL]
    assert next(iter(r.tabelas)) == TABELA_GERAL
    assert geral.columns.tolist() == ["Variável", "IV", "Poder preditivo", "Categorias"]
    assert geral["Variável"].tolist() == ["forte", "ruido"]  # maior IV primeiro
    assert list(r.tabelas)[1:] == ["IV de 'forte'", "IV de 'ruido'"]
    assert geral["Categorias"].tolist() == [4, 2]


# ---------------------------------------------------------------------------
# Faixas das quantitativas
# ---------------------------------------------------------------------------


def test_quantitativa_em_decis_com_rotulos_de_minimo_a_maximo():
    x = pd.Series(np.arange(1, 101))
    rotulos, ordem, em_faixas = categorizar(x, 10)
    assert em_faixas
    assert ordem == [f"{10 * i + 1} – {10 * i + 10}" for i in range(10)]
    assert rotulos.value_counts().tolist() == [10] * 10
    assert rotulos[0] == "1 – 10" and rotulos[99] == "91 – 100"


def test_quantitativa_decimal_e_cinco_faixas():
    x = pd.Series(np.linspace(0.5, 10.0, 20))
    _, ordem, _ = categorizar(x, 5)
    assert len(ordem) == 5 and ordem[0] == "0,50 – 2,00"


def test_quantitativa_com_poucos_valores_vira_categoria_por_valor():
    rotulos, ordem, em_faixas = categorizar(pd.Series([1, 2, 2, 3, 3, 3]), 10)
    assert not em_faixas and ordem == ["1", "2", "3"] and rotulos.tolist()[0] == "1"


def test_faixas_repetidas_sao_juntadas():
    x = pd.Series([0] * 80 + list(range(1, 21)))  # 80% iguais: decis repetidos
    _, ordem, em_faixas = categorizar(x, 10)
    assert em_faixas and len(ordem) < 10 and ordem[0] == "0"
    rotulos, _, _ = categorizar(x, 10)
    assert rotulos.value_counts()["0"] == 80


def test_execucao_com_faixas_no_titulo_da_tabela():
    rng = np.random.default_rng(3)
    idade = rng.integers(18, 70, 300)
    y = np.where(rng.uniform(size=300) < (idade - 18) / 60, "Sim", "Não")
    df = pd.DataFrame({"y": y, "idade": idade})
    r = TesteInformationValue().executar(df, _params(preditores=["idade"], faixas="5"))
    (nome_tabela,) = [n for n in r.tabelas if n != TABELA_GERAL]
    assert nome_tabela == "IV de 'idade' (5 faixas)"
    categorias = r.tabelas[nome_tabela]["Categoria"].tolist()[:-1]
    assert all(" – " in c for c in categorias)
    rotulos, _, _ = categorizar(df["idade"], 5)
    assert r.estatisticas["iv_idade"] == pytest.approx(
        _iv_independente(rotulos, pd.Series(y == "Sim")), rel=1e-10
    )


# ---------------------------------------------------------------------------
# Zeros, ausentes e avisos
# ---------------------------------------------------------------------------


def test_categoria_sem_eventos_recebe_meio_nas_contagens():
    linhas = [("A", "Não")] * 20 + [("B", "Sim")] * 10 + [("B", "Não")] * 10
    df = pd.DataFrame(linhas, columns=["cat", "y"])
    r = TesteInformationValue().executar(df, _params())
    # A: 0 eventos → 0,5 e 20,5; B: 10 e 10. Totais ajustados: 10,5 eventos e 30,5 não eventos.
    pe = {"A": 0.5 / 10.5, "B": 10 / 10.5}
    pne = {"A": 20.5 / 30.5, "B": 10 / 30.5}
    esperado = sum((pne[c] - pe[c]) * math.log(pne[c] / pe[c]) for c in "AB")
    assert r.estatisticas["iv_cat"] == pytest.approx(esperado, rel=REL)
    assert any("'A' não têm eventos ou não eventos: somado 0,5" in a for a in r.avisos)


def test_x_ausente_vira_categoria_e_y_ausente_e_descartado():
    df = _base_manual()
    df.loc[0:4, "cat"] = None  # 5 eventos de A sem categoria
    df.loc[140] = ["A", None]
    r = TesteInformationValue().executar(df, _params())
    tabela = r.tabelas["IV de 'cat'"]
    assert AUSENTE in tabela["Categoria"].tolist()
    assert tabela.loc[tabela["Categoria"] == AUSENTE, "n"].item() == 5
    assert "1 linha(s) com 'y' ausente foram descartadas." in r.avisos
    assert r.estatisticas["n"] == 140


def test_avisos_de_iv_suspeito_e_categoria_unica():
    df = pd.DataFrame({"y": ["Sim"] * 50 + ["Não"] * 50, "vaza": ["s"] * 50 + ["n"] * 50})
    df["const"] = "k"
    r = TesteInformationValue().executar(df, _params(preditores=["vaza", "const"]))
    assert any("O IV de 'vaza'" in a and "vazamento" in a for a in r.avisos)
    assert "A variável 'const' tem uma única categoria: IV = 0." in r.avisos
    assert r.estatisticas["iv_const"] == 0.0
    geral = r.tabelas[TABELA_GERAL]
    assert geral["Poder preditivo"].tolist() == [FORCA_SUSPEITA, "Sem poder preditivo"]


@pytest.mark.parametrize(
    ("iv", "forca"),
    [
        (0.0, "Sem poder preditivo"),
        (0.0199, "Sem poder preditivo"),
        (0.02, "Fraco"),
        (0.0999, "Fraco"),
        (0.1, "Médio"),
        (0.3, "Forte"),
        (0.5, "Forte"),
        (0.5001, FORCA_SUSPEITA),
    ],
)
def test_faixas_de_siddiqi(iv, forca):
    assert classificar_iv(iv) == forca


# ---------------------------------------------------------------------------
# Resultado sem decisão, interpretação e figuras
# ---------------------------------------------------------------------------


def test_sem_decisao_nem_p_valor():
    r = TesteInformationValue().executar(_base_manual(), _params())
    assert r.decisao == "" and r.p_valor is None and r.comparacao is None
    assert r.interpretacao.startswith(
        "Information Value de 1 variável(is) para prever 'y' = 'Sim' (140 casos, 60 eventos). "
        "A de maior poder é 'cat' (IV = 0,8910: muito forte (verificar vazamento))."
    )


def test_figuras_iv_e_woe():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "y": rng.choice(["Sim", "Não"], 200),
            "cat": rng.choice(["A", "B", "C"], 200),
            "num": rng.normal(size=200),
        }
    )
    r = TesteInformationValue().executar(df, _params(preditores=["num", "cat"], faixas="5"))
    geral, grupo = r.figuras
    ordem = r.tabelas[TABELA_GERAL]["Variável"].tolist()
    assert [c["rotulo"] for c in geral.dados["categorias"]] == ordem
    assert isinstance(grupo, GrupoFiguras) and grupo.rotulo == "Variável"
    assert list(grupo.opcoes) == ordem and grupo.padrao == ordem[0]
    woe_cat = grupo.opcoes["cat"].dados["categorias"]
    assert [c["rotulo"] for c in woe_cat] == ["A", "B", "C"]  # ordem natural das categorias


# ---------------------------------------------------------------------------
# Validação e formulário
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "trecho"),
    [
        ({"y": None}, "Selecione a variável resposta (y)."),
        ({"y": "nao_existe"}, "A coluna 'nao_existe' não existe no arquivo."),
        ({"y": "cat"}, "A variável resposta 'cat' deve ter exatamente 2 valores (tem 3)."),
        ({"evento": "talvez"}, "Escolha qual valor da variável resposta é o evento."),
        ({"preditores": []}, "Selecione ao menos uma variável X."),
        ({"preditores": ["cat", "y"]}, "A variável resposta não pode ser também uma variável X."),
        ({"preditores": ["cat", "cat"]}, "As variáveis X devem ser colunas diferentes."),
        ({"faixas": "7"}, "Escolha o número de faixas."),
    ],
)
def test_validacao(params, trecho):
    erros = TesteInformationValue().validar(_base_manual(), _params(**params))
    assert trecho in erros, erros
    with pytest.raises(ErroValidacao):
        TesteInformationValue().executar(_base_manual(), _params(**params))


def test_y_com_uma_so_classe_preenchida():
    df = _base_manual()
    df.loc[df["y"] == "Não", "y"] = None  # sobram só os "Sim"
    erros = TesteInformationValue().validar(df, _params())
    assert "A variável resposta 'y' deve ter exatamente 2 valores (tem 1)." in erros


def test_formulario():
    specs = TesteInformationValue().parametros()
    assert [(s.nome, s.tipo) for s in specs] == [
        ("y", "coluna_binaria"),
        ("evento", "nivel"),
        ("preditores", "preditores"),
        ("faixas", "opcao"),
    ]
    assert specs[1].depende_de == "y" and specs[2].depende_de == "y"
    assert specs[3].padrao == "10" and specs[3].opcoes == ["5", "10", "20"]
