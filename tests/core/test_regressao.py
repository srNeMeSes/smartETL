"""Regressão Linear (TesteRegressaoLinear): checklist da seção 9 e docs/regressao_linear.md.

Fonte dos valores de referência: tests/referencias_r/referencias.json (R 4.6.1 + lmtest 0.9.40,
car 3.1.5, nortest 1.0-4; ver tests/referencias_r/gerar_referencias.R) sobre datasets públicos
(mtcars, Prestige, longley, cars) e grande.csv. As funções de cálculo isoladas estão em
test_diagnosticos.py; aqui se testa o resultado completo do teste (seções, tabelas exibidas,
figuras, simulação) e as validações.
"""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import RAIZ

from core.base import ErroValidacao, GrupoFiguras
from core.diagnosticos import DW_SEM, VIF_MODERADA, VIF_MUITO_GRAVE
from core.interpretacao import REJEITA_H0
from core.testes.regressao import (
    INTERCEPTO,
    VALORES_AJUSTADOS,
    TesteRegressaoLinear,
    formatar_coeficiente,
)

PASTA = RAIZ / "tests" / "referencias_r"
R = json.loads((PASTA / "referencias.json").read_text(encoding="utf-8"))
REL = 1e-6
MODELOS = {
    "mtcars": ("mpg", ["wt", "hp", "cyl", "am"]),
    "prestige": ("prestige", ["education", "income", "type"]),
    "longley": (
        "Employed",
        ["GNP_deflator", "GNP", "Unemployed", "Armed_Forces", "Population", "Year"],
    ),
    "cars": ("dist", ["speed"]),
    "grande": ("y", ["x1", "x2", "g"]),
}
SECOES = [
    "Pressupostos",
    "Heterocedasticidade",
    "Autocorrelação",
    "Colinearidade",
    "Normalidade dos resíduos",
    "Modelo",
    "Coeficientes",
]


@pytest.fixture(scope="module")
def teste():
    return TesteRegressaoLinear()


def _df(nome):
    return pd.read_csv(PASTA / f"{nome}.csv", encoding="utf-8")


def _params(nome, **extra):
    y, preditores = MODELOS[nome]
    base = {
        "y": y,
        "preditores": preditores,
        "referencias": {},
        "ordenar_por": VALORES_AJUSTADOS,
        "confianca": "95%",
        "alfa": 0.05,
    }
    return base | extra


@pytest.fixture(scope="module", params=list(MODELOS))
def caso(request, teste):
    nome = request.param
    return nome, teste.executar(_df(nome), _params(nome)), R[nome]


def _secao(resultado, titulo):
    return next(s for s in resultado.secoes if s.titulo == titulo)


def _br(valor, casas):
    return f"{valor:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")


# ---------------------------------------------------------------------------
# Valores contra o R (resultado completo)
# ---------------------------------------------------------------------------


def test_estatisticas_contra_o_r(caso):
    _, r, ref = caso
    e = r.estatisticas
    pares = {
        "r2": ref["r2"],
        "r2_ajustado": ref["r2_ajustado"],
        "rmse": ref["rmse"],
        "f": ref["f"],
        "p_valor": ref["p_f"],
        "bp": ref["bp"]["estatistica"],
        "p_bp": ref["bp"]["p"],
        "gq": ref["gq_ajustados"]["estatistica"],
        "p_gq": ref["gq_ajustados"]["p"],
        "hmc": ref["hmc_ajustados"]["estatistica"],
        "dw": ref["dw"],
        "bg": ref["bg"]["estatistica"],
        "p_bg": ref["bg"]["p"],
        "normalidade": ref["normalidade"]["estatistica"],
        "p_normalidade": ref["normalidade"]["p"],
    }
    for chave, valor in pares.items():
        assert e[chave] == pytest.approx(valor, rel=REL), chave
    assert e["r"] == pytest.approx(ref["r2"] ** 0.5, rel=REL)
    assert (e["n"], e["gl1"], e["gl2"]) == (ref["n"], ref["gl1"], ref["gl2"])
    assert 0 <= e["p_hmc"] <= 1


def test_ordenacao_por_variavel(teste):
    r = teste.executar(_df("prestige"), _params("prestige", ordenar_por="income"))
    ref = R["prestige"]
    assert r.estatisticas["gq"] == pytest.approx(ref["gq_variavel"]["estatistica"], rel=REL)
    assert r.estatisticas["hmc"] == pytest.approx(ref["hmc_variavel"]["estatistica"], rel=REL)
    nota = _secao(r, "Heterocedasticidade").notas[0]
    assert "ordenados pelos 'income'" in nota


@pytest.mark.parametrize(("confianca", "sufixo"), [("95%", "95"), ("90%", "90")])
def test_tabela_de_coeficientes_e_ic(teste, confianca, sufixo):
    r = teste.executar(_df("mtcars"), _params("mtcars", confianca=confianca))
    tabela = r.tabelas["Coeficientes"]
    ref = R["mtcars"]
    assert tabela.columns.tolist() == [
        "Preditor",
        "Estimativa",
        "EP",
        f"LI ({confianca})",
        f"LS ({confianca})",
        "t",
        "p-valor",
    ]
    assert tabela["Preditor"].tolist() == [
        INTERCEPTO,
        "wt",
        "hp",
        "cyl (ref.: 8 cilindros)",
        "    4 cilindros",
        "    6 cilindros",
        "    8 cilindros",
        "am (ref.: Automático)",
        "    Manual",
        "    Automático",
    ]
    assert tabela.attrs["destaques"] == [3, 7]
    linhas = [0, 1, 2, 4, 5, 8]  # linhas com coeficiente, na ordem dos termos do R
    assert tabela["Estimativa"].iloc[linhas].tolist() == [
        formatar_coeficiente(v) for v in ref["estimativa"]
    ]
    assert tabela[f"LI ({confianca})"].iloc[linhas].tolist() == [
        formatar_coeficiente(v) for v in ref[f"ic{sufixo}_li"]
    ]
    assert tabela.iloc[6, 1:].tolist() == ["—"] * 6  # nível de referência
    assert tabela.iloc[3, 1:].tolist() == [""] * 6  # cabeçalho da categórica


def test_tabela_do_modelo_com_dica_do_rmse(caso):
    _, r, ref = caso
    tabela = r.tabelas["Modelo"]
    assert tabela.columns.tolist() == [
        "R",
        "R²",
        "R² ajustado",
        "RMSE",
        "F",
        "gl1",
        "gl2",
        "p-valor",
        "n",
    ]
    assert len(tabela) == 1
    assert tabela["R²"].iloc[0] == _br(ref["r2"], 4)
    assert tabela["n"].iloc[0] == str(ref["n"])
    assert "√(SQE / (n − p − 1))" in tabela.attrs["dicas"]["RMSE"]


# ---------------------------------------------------------------------------
# Pressupostos exibidos (só os valores do modelo, nunca as faixas)
# ---------------------------------------------------------------------------


def test_ordem_das_secoes(caso):
    _, r, _ = caso
    assert [s.titulo for s in r.secoes] == SECOES
    assert [s.nivel for s in r.secoes] == [1, 2, 2, 2, 2, 1, 1]
    assert _secao(r, "Modelo").destaque in ("Rejeita H₀", "Não rejeita H₀")


def test_durbin_watson_uma_linha_e_notas(caso):
    _, r, ref = caso
    tabela = r.tabelas["Durbin-Watson"]
    assert tabela.columns.tolist() == ["DW", "Interpretação"] and len(tabela) == 1
    assert tabela["DW"].iloc[0] == _br(ref["dw"], 2)
    autocorrelacao = _secao(r, "Autocorrelação")
    assert autocorrelacao.notas == [
        "Regra prática; os limites formais dependem de n e do número de preditores."
    ]
    assert autocorrelacao.avisos == [
        "Só é interpretável se as linhas tiverem ordem significativa (tempo, sequência de coleta)."
    ]


def test_vif_uma_linha_por_preditor(caso):
    nome, r, ref = caso
    if ref["vif"] == "nenhum":
        assert "Colinearidade (VIF)" not in r.tabelas
        assert _secao(r, "Colinearidade").textos == [
            "Com um único preditor não há colinearidade a avaliar."
        ]
        return
    tabela = r.tabelas["Colinearidade (VIF)"]
    assert tabela["Preditor"].tolist() == MODELOS[nome][1] == ref["vif"]["variaveis"]
    assert tabela["Interpretação"].iloc[0] in {
        "Sem colinearidade relevante",
        "Baixa/moderada",
        "Problemática",
        "Grave",
        "Muito grave",
    }


def test_vif_sem_categoricas_tem_colunas_da_especificacao(teste):
    r = teste.executar(_df("longley"), _params("longley"))
    tabela = r.tabelas["Colinearidade (VIF)"]
    assert tabela.columns.tolist() == ["Preditor", "VIF", "Tolerância", "Interpretação"]
    assert tabela["VIF"].tolist() == [_br(v, 2) for v in R["longley"]["vif"]["gvif"]]
    assert tabela["Tolerância"].iloc[0] == _br(1 / round(R["longley"]["vif"]["gvif"][0], 2), 2)
    assert set(tabela["Interpretação"]) <= {VIF_MUITO_GRAVE, "Grave", "Problemática", VIF_MODERADA}
    assert tabela["Interpretação"].iloc[1] == VIF_MUITO_GRAVE  # GNP: VIF ≈ 1788


def test_gvif_para_categorica_de_tres_niveis(teste):
    r = teste.executar(_df("mtcars"), _params("mtcars"))
    tabela = r.tabelas["Colinearidade (VIF)"]
    assert tabela.columns.tolist() == [
        "Preditor",
        "VIF / GVIF",
        "gl",
        "GVIF^(1/(2·gl))",
        "Tolerância",
        "Interpretação",
    ]
    cyl = tabela.iloc[2]
    ref = R["mtcars"]["vif"]
    assert cyl["gl"] == "2" and cyl["VIF / GVIF"] == _br(ref["gvif"][2], 2)
    assert cyl["GVIF^(1/(2·gl))"] == _br(ref["gvif_ajustado"][2], 2)
    # cyl: GVIF = 5,82, mas (GVIF^(1/4))² = 2,41 → "Baixa/moderada" (não "Problemática").
    assert cyl["Interpretação"] == VIF_MODERADA


def test_nenhuma_tabela_exibe_as_faixas(caso):
    _, r, _ = caso
    textos = " ".join(
        str(v) for tabela in r.tabelas.values() for v in tabela.astype(str).to_numpy().ravel()
    )
    for faixa in ("1 a < 2", "2 a < 5", "≥ 20", "0 a < 1,0", "1,5 a 2,5", "> 3,0 a 4"):
        assert faixa not in textos


def test_interpretacao_dos_pressupostos(teste):
    r = teste.executar(_df("cars"), _params("cars"))
    normal = r.tabelas["Normalidade dos resíduos"]
    assert normal["Teste"].iloc[0] == "Shapiro-Wilk"
    assert "se afastam da distribuição normal" in normal["Interpretação"].iloc[0]  # p = 0,022
    assert r.tabelas["Durbin-Watson"]["Interpretação"].iloc[0] == DW_SEM  # 1,68
    grande = teste.executar(_df("grande"), _params("grande"))
    assert grande.tabelas["Normalidade dos resíduos"]["Teste"].iloc[0] == (
        "Kolmogorov-Smirnov (Lilliefors)"
    )
    assert any("Com n grande" in nota for nota in _secao(grande, "Normalidade dos resíduos").notas)


def test_aviso_goldfeld_quandt_degenerado(teste):
    r = teste.executar(_df("mtcars"), _params("mtcars"))
    assert any("Goldfeld-Quandt: em uma das metades" in a for a in r.avisos)


# ---------------------------------------------------------------------------
# Decisão e casos-limite
# ---------------------------------------------------------------------------


def test_decisao_do_teste_f(caso):
    _, r, ref = caso
    assert r.decisao == REJEITA_H0  # todos os modelos de referência têm p(F) < 0,001
    assert r.p_valor == pytest.approx(ref["p_f"], rel=REL)
    assert "todos os coeficientes dos preditores são zero" in r.interpretacao
    assert _secao(r, "Modelo").textos == [r.interpretacao]


def test_troca_do_nivel_de_referencia(teste):
    padrao = teste.executar(_df("prestige"), _params("prestige"))
    trocado = teste.executar(_df("prestige"), _params("prestige", referencias={"type": "prof"}))
    assert trocado.estatisticas["r2"] == pytest.approx(padrao.estatisticas["r2"], rel=1e-12)
    preditores = trocado.tabelas["Coeficientes"]["Preditor"].tolist()
    assert "type (ref.: prof)" in preditores and "    bc" in preditores
    assert trocado.tabelas["Coeficientes"]["Estimativa"].iloc[4] == formatar_coeficiente(
        R["prestige_ref_prof"]["estimativa"][3]
    )


def test_linhas_removidas_e_nivel_raro(teste):
    r = teste.executar(_df("prestige"), _params("prestige"))
    assert "4 linha(s) com valor ausente em y ou em algum preditor foram removidas." in r.avisos
    df = _df("cars")
    df["tipo"] = ["raro"] * 3 + ["comum"] * 47
    r = teste.executar(df, _params("cars", preditores=["speed", "tipo"]))
    assert any("O nível 'raro' de 'tipo' tem só 3 observação(ões)" in a for a in r.avisos)


def test_apenas_categoricas(teste):
    r = teste.executar(_df("mtcars"), _params("mtcars", preditores=["cyl", "am"]))
    assert r.estatisticas["gl1"] == 3
    assert r.tabelas["Colinearidade (VIF)"]["Preditor"].tolist() == ["cyl", "am"]
    assert list(r.figuras[0].opcoes) == [VALORES_AJUSTADOS, "cyl", "am"]


@pytest.mark.parametrize(
    ("extra", "trecho"),
    [
        ({"y": None}, "Selecione a variável dependente (y)."),
        ({"y": "cyl"}, "A coluna 'cyl' não é numérica."),
        ({"preditores": []}, "Selecione ao menos um preditor."),
        ({"preditores": ["wt", "mpg"]}, "A variável dependente não pode ser também um preditor."),
        ({"preditores": ["wt", "nao_existe"]}, "A coluna 'nao_existe' não existe no arquivo."),
        ({"ordenar_por": "cyl"}, "A coluna 'cyl' não é numérica."),
        ({"confianca": "80%"}, "Escolha um nível de confiança válido."),
        ({"alfa": 2}, "O nível de significância (α) deve estar entre 0 e 1."),
    ],
)
def test_validacao_de_parametros(teste, extra, trecho):
    df = _df("mtcars")
    assert trecho in teste.validar(df, _params("mtcars", **extra))
    with pytest.raises(ErroValidacao):
        teste.executar(df, _params("mtcars", **extra))


def test_validacao_do_modelo(teste):
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=20), rng.normal(size=20)
    df = pd.DataFrame(
        {"y": rng.normal(size=20), "a": a, "b": b, "c": a + b, "k": 1.0, "g": ["x"] * 20}
    )
    base = {"y": "y", "alfa": 0.05, "confianca": "95%"}
    erros = teste.validar(df, base | {"preditores": ["a", "b", "c"]})
    assert erros == [
        "Colinearidade perfeita entre 'a', 'b', 'c': um deles é combinação exata dos outros. "
        "Remova um dos preditores."
    ]
    assert "constante nas linhas usadas" in teste.validar(df, base | {"preditores": ["a", "k"]})[0]
    assert "tem um único nível" in teste.validar(df, base | {"preditores": ["g"]})[0]
    pequeno = df.head(3)
    assert "é preciso n > p + 1" in teste.validar(pequeno, base | {"preditores": ["a", "b"]})[0]
    df["y"] = 5.0
    assert (
        "Todos os valores de 'y' são iguais" in teste.validar(df, base | {"preditores": ["a"]})[0]
    )
    df = _df("mtcars")
    df.loc[0, "hp"] = np.nan
    assert (
        "valores ausentes nas linhas do modelo"
        in teste.validar(df, _params("mtcars", preditores=["wt"], ordenar_por="hp"))[0]
    )


# ---------------------------------------------------------------------------
# Visualização e simulação
# ---------------------------------------------------------------------------


def test_figuras_de_residuos_e_qq(teste):
    r = teste.executar(_df("mtcars"), _params("mtcars"))
    grupo, qq = r.figuras
    assert isinstance(grupo, GrupoFiguras) and grupo.padrao == VALORES_AJUSTADOS
    assert [f.tipo for f in grupo.opcoes.values()] == [
        "dispersao",
        "dispersao",
        "dispersao",
        "boxplot",
        "boxplot",
    ]
    ajustados = grupo.opcoes[VALORES_AJUSTADOS].dados
    assert ajustados["linhas"][0]["y1"] == ajustados["linhas"][0]["y2"] == 0.0
    assert ajustados["n_total"] == 32
    caixas = grupo.opcoes["cyl"].dados
    assert [g["rotulo"] for g in caixas["grupos"]] == ["4 cilindros", "6 cilindros", "8 cilindros"]
    assert caixas["referencias"][0]["valor"] == 0.0
    assert qq.tipo == "dispersao" and qq.dados["rotulo_x"] == "Quantis teóricos (normal)"


@pytest.mark.parametrize("nome", ["mtcars", "cars", "prestige"])
def test_simulacao_contra_predict_do_r(teste, nome):
    novos = {
        "mtcars": {"wt": 3, "hp": 150, "cyl": "6 cilindros", "am": "Manual"},
        "cars": {"speed": 21},
        "prestige": {"education": 11, "income": 7000, "type": "wc"},
    }
    r = teste.executar(_df(nome), _params(nome))
    p = r.simulacao.prever(novos[nome])
    ref = R[nome]["previsao"]
    assert (p.valor, p.ic_inferior, p.ic_superior, p.ip_inferior, p.ip_superior) == pytest.approx(
        (ref["valor"], ref["ic_li"], ref["ic_ls"], ref["ip_li"], ref["ip_ls"]), rel=REL
    )
    total = p.figura.dados["inicio"]["valor"] + sum(c for _, c in p.contribuicoes)
    assert total == pytest.approx(p.valor, rel=1e-12)
    assert p.figura.dados["final"]["valor"] == pytest.approx(p.valor, rel=1e-12)


def test_simulacao_campos_equacao_e_extrapolacao(teste):
    df = _df("mtcars")
    sim = teste.executar(df, _params("mtcars")).simulacao
    wt, _, cyl, _ = sim.campos
    assert (wt.categorica, wt.inicial, wt.minimo, wt.maximo) == (
        False,
        pytest.approx(df["wt"].mean()),
        df["wt"].min(),
        df["wt"].max(),
    )
    assert cyl.niveis == ("4 cilindros", "6 cilindros", "8 cilindros")
    assert cyl.inicial == cyl.referencia == "8 cilindros"
    termos = sim.equacao()
    assert termos[0].texto == "mpg = 31,54" and termos[0].variavel is None
    assert [t.variavel for t in termos[1:]] == ["wt", "hp", "cyl", "cyl", "am"]
    assert termos[1].texto == " − 2,497·(wt)" and termos[3].texto == " + 2,164·(4 cilindros)"
    assert sim.prever(sim.valores_iniciais()).extrapolacoes == []
    fora = sim.prever(sim.valores_iniciais() | {"wt": 9})
    assert fora.extrapolacoes == [
        "'wt' = 9,000 está fora da faixa observada [1,513; 5,424]: a previsão é uma extrapolação."
    ]
    assert sim.confianca == 0.95


@pytest.mark.parametrize(
    ("valor", "texto"),
    [
        (3135.82, "3.135,82"),
        (35.41, "35,41"),
        (3.8783, "3,878"),
        (0.0010132, "0,001013"),
        (-2.4968, "-2,497"),
        (0.0, "0,00"),
    ],
)
def test_formatar_coeficiente(valor, texto):
    assert formatar_coeficiente(valor) == texto


def test_formulario(teste):
    specs = teste.parametros()
    assert [(s.nome, s.tipo) for s in specs] == [
        ("y", "coluna_numerica"),
        ("preditores", "preditores"),
        ("referencias", "niveis_referencia"),
        ("ordenar_por", "ordenacao"),
        ("confianca", "opcao"),
        ("alfa", "alfa"),
    ]
    assert "Só colunas numéricas" in specs[0].ajuda
    assert specs[2].depende_de == "preditores" and not specs[2].obrigatorio
    assert specs[3].padrao == VALORES_AJUSTADOS and specs[4].padrao == "95%"
