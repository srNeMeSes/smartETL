"""Regressão Logística: checklist da seção 9 e docs/regressao_logistica.md.

Fonte dos valores de referência: tests/referencias_r/referencias_logit.json, gerado por
tests/referencias_r/gerar_referencias_logit.R com R 4.6.1 (`glm(family = binomial)`,
`confint.default`, `predict(type = "link", se.fit = TRUE)`, `logLik`, `AIC`, `BIC`),
car 3.1.5 (`vif`), ResourceSelection 0.3.6 (`hoslem.test`) e pROC 1.19.1 (`auc`), sobre os
datasets públicos Mroz (carData) e birthwt (MASS). Os CSV da pasta são exatamente os dados usados.
Tolerância rel = 1e-6 (seção 9); o ajuste iterativo do statsmodels e o IRLS do R concordam a ~1e-8.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
from conftest import RAIZ

from core import diagnosticos as dg
from core.base import ErroValidacao, GrupoFiguras
from core.diagnosticos import VIF_SEM
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.regressao import VALORES_AJUSTADOS, formatar_coeficiente
from core.testes.regressao_logistica import TesteRegressaoLogistica, binarizar

PASTA = RAIZ / "tests" / "referencias_r"
R = json.loads((PASTA / "referencias_logit.json").read_text(encoding="utf-8"))
REL = 1e-6
MODELOS = {
    "mroz": ("lfp", "yes", ["k5", "k618", "age", "wc", "hc", "lwg", "inc"]),
    "birthwt": ("low", "1", ["age", "lwt", "race", "smoke", "ht", "ui"]),
}
SECOES = [
    "Pressupostos",
    "Colinearidade",
    "Linearidade do logit",
    "Qualidade do ajuste",
    "Tamanho da amostra e separação",
    "Modelo",
    "Classificação",
    "Coeficientes",
]


def _lista(valor):
    """O JSON do R grava vetores de um elemento como escalar."""
    return valor if isinstance(valor, list) else [valor]


@pytest.fixture(scope="module")
def teste():
    return TesteRegressaoLogistica()


def _df(nome):
    return pd.read_csv(PASTA / f"{nome}.csv", encoding="utf-8")


def _params(nome, **extra):
    y, evento, preditores = MODELOS[nome]
    base = {
        "y": y,
        "evento": evento,
        "preditores": preditores,
        "referencias": {},
        "limiar": 0.5,
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


# ---------------------------------------------------------------------------
# Valores contra o R
# ---------------------------------------------------------------------------


def test_coeficientes_e_ic_de_wald(caso):
    _, r, ref = caso
    sim = r.simulacao
    beta = sim.coeficientes
    ep = np.sqrt(np.diag(sim.covariancia))
    assert list(beta) == pytest.approx(ref["estimativa"], rel=REL)
    assert list(ep) == pytest.approx(ref["ep"], rel=REL)
    assert list(beta / ep) == pytest.approx(ref["z"], rel=REL)
    assert list(beta - 1.959963984540054 * ep) == pytest.approx(ref["ic95_li"], rel=REL)


def test_teste_global_pseudo_r2_e_criterios(caso):
    _, r, ref = caso
    e = r.estatisticas
    for chave, valor in {
        "loglik": ref["loglik"],
        "loglik_nulo": ref["loglik_nulo"],
        "lr": ref["lr"],
        "p_valor": ref["p_lr"],
        "mcfadden": ref["mcfadden"],
        "cox_snell": ref["cox_snell"],
        "nagelkerke": ref["nagelkerke"],
        "aic": ref["aic"],
        "bic": ref["bic"],
    }.items():
        assert e[chave] == pytest.approx(valor, rel=REL), chave
    assert (e["n"], e["eventos"], e["gl"]) == (ref["n"], ref["eventos"], ref["gl"])


def test_hosmer_lemeshow_e_auc(caso):
    _, r, ref = caso
    e = r.estatisticas
    assert e["hl"] == pytest.approx(ref["hl"]["estatistica"], rel=REL)
    assert e["gl_hl"] == ref["hl"]["gl"]
    assert e["p_hl"] == pytest.approx(ref["hl"]["p"], rel=REL)
    assert e["auc"] == pytest.approx(ref["auc"], rel=1e-12)


def test_matriz_de_confusao_no_limiar_padrao(caso):
    _, r, ref = caso
    e = r.estatisticas
    c = ref["confusao_05"]
    assert (e["vp"], e["fn"], e["fp"], e["vn"]) == (c["vp"], c["fn"], c["fp"], c["vn"])
    assert e["acuracia"] == pytest.approx((c["vp"] + c["vn"]) / ref["n"])
    assert e["sensibilidade"] == pytest.approx(c["vp"] / (c["vp"] + c["fn"]))
    assert e["especificidade"] == pytest.approx(c["vn"] / (c["vn"] + c["fp"]))


def test_gvif_como_car_vif(caso):
    _, r, ref = caso
    tabela = r.tabelas["Colinearidade (VIF)"]
    assert tabela["Preditor"].tolist() == ref["vif"]["variaveis"]
    coluna = "VIF / GVIF" if "VIF / GVIF" in tabela else "VIF"
    assert tabela[coluna].tolist() == [f"{v:.2f}".replace(".", ",") for v in ref["vif"]["gvif"]]
    assert set(tabela["Interpretação"]) == {VIF_SEM}


def test_box_tidwell_contra_glm_com_termo_xlnx(caso):
    nome, r, ref = caso
    variaveis = _lista(ref["box_tidwell"]["variaveis"])
    for var, z, p in zip(
        variaveis, _lista(ref["box_tidwell"]["z"]), _lista(ref["box_tidwell"]["p"]), strict=True
    ):
        assert r.estatisticas[f"bt_z_{var}"] == pytest.approx(z, rel=REL)
        assert r.estatisticas[f"bt_p_{var}"] == pytest.approx(p, rel=REL)
    tabela = r.tabelas["Linearidade do logit (Box-Tidwell)"]
    nao_aplicaveis = [
        linha.Preditor
        for linha in tabela.itertuples()
        if linha.Interpretação.startswith("Não aplicável")
    ]
    numericos = [v for v in MODELOS[nome][2] if v not in ("wc", "hc", "race", "smoke")]
    assert sorted(nao_aplicaveis + variaveis) == sorted(numericos)


@pytest.mark.parametrize("nome", list(MODELOS))
def test_simulacao_contra_predict_do_r(teste, nome):
    novos = {
        "mroz": {"k5": 1, "k618": 2, "age": 40, "wc": "yes", "hc": "no", "lwg": 1.2, "inc": 20},
        "birthwt": {"age": 25, "lwt": 120, "race": "negra", "smoke": "sim", "ht": 0, "ui": 1},
    }
    sim = teste.executar(_df(nome), _params(nome)).simulacao
    p = sim.prever(novos[nome])
    ref = R[nome]["previsao"]
    assert p.valor == pytest.approx(ref["prob"], rel=REL)
    assert (p.ic_inferior, p.ic_superior) == pytest.approx((ref["ic_li"], ref["ic_ls"]), rel=REL)
    assert math.isnan(p.ip_inferior) and math.isnan(p.ip_superior)
    logit = p.figura.dados["final"]["valor"]
    assert logit == pytest.approx(ref["logit"], rel=REL)
    assert p.classe == ("1" if nome == "birthwt" else "no")  # 0,774 ≥ 0,5; 0,480 < 0,5


def test_residuos_de_deviance_e_probabilidades(caso):
    _, r, ref = caso
    grupo = next(f for f in r.figuras if isinstance(f, GrupoFiguras))
    dados = grupo.opcoes[VALORES_AJUSTADOS].dados
    assert dados["x"][:5] == pytest.approx(ref["ajustados_5"], rel=REL)
    assert dados["y"][:5] == pytest.approx(ref["residuos_deviance_5"], rel=REL)


@pytest.mark.parametrize(("confianca", "sufixo"), [("95%", "95"), ("90%", "90")])
def test_tabela_de_coeficientes_com_odds_ratio(teste, confianca, sufixo):
    r = teste.executar(_df("birthwt"), _params("birthwt", confianca=confianca))
    tabela = r.tabelas["Coeficientes"]
    ref = R["birthwt"]
    assert tabela.columns.tolist() == [
        "Preditor",
        "Estimativa (log-odds)",
        "EP",
        "z",
        "p-valor",
        "Odds ratio",
        f"LI OR ({confianca})",
        f"LS OR ({confianca})",
    ]
    assert tabela["Preditor"].tolist() == [
        "(Intercepto)",
        "age",
        "lwt",
        "race (ref.: branca)",
        "    negra",
        "    outra",
        "    branca",
        "smoke (ref.: não)",
        "    sim",
        "    não",
        "ht",
        "ui",
    ]
    linhas = [0, 1, 2, 4, 5, 8, 10, 11]
    assert tabela["Odds ratio"].iloc[linhas].tolist() == [
        formatar_coeficiente(math.exp(b)) for b in ref["estimativa"]
    ]
    assert tabela[f"LI OR ({confianca})"].iloc[linhas].tolist() == [
        formatar_coeficiente(math.exp(v)) for v in ref[f"ic{sufixo}_li"]
    ]
    assert tabela.attrs["destaques"] == [3, 7]
    assert tabela.iloc[6, 1:].tolist() == ["—"] * 7


# ---------------------------------------------------------------------------
# Seções, decisão e opções
# ---------------------------------------------------------------------------


def test_ordem_das_secoes_e_decisao(caso):
    _, r, ref = caso
    assert [s.titulo for s in r.secoes] == SECOES
    assert r.decisao == REJEITA_H0 and _secao(r, "Modelo").destaque == "Rejeita H₀"
    assert r.p_valor == pytest.approx(ref["p_lr"], rel=REL)
    assert "os preditores não ajudam a prever" in r.interpretacao
    assert r.comparacao is None


def test_hosmer_lemeshow_interpretacao(teste):
    mroz = teste.executar(_df("mroz"), _params("mroz"))  # p = 0,002
    assert "o modelo se ajusta mal" in mroz.tabelas["Qualidade do ajuste"]["Interpretação"][0]
    bw = teste.executar(_df("birthwt"), _params("birthwt"))  # p = 0,207
    assert "Sem evidência de mau ajuste" in bw.tabelas["Qualidade do ajuste"]["Interpretação"][0]


def test_aviso_de_poucos_eventos_por_variavel(teste):
    r = teste.executar(_df("birthwt"), _params("birthwt"))  # 59 eventos / 7 = 8,4
    assert r.estatisticas["epv"] == pytest.approx(59 / 7)
    assert any("EPV = 8,4 < 10" in a for a in r.avisos)
    r = teste.executar(_df("mroz"), _params("mroz"))  # 325 / 7 = 46,4
    assert not any("EPV" in a for a in r.avisos)


@pytest.mark.parametrize("limiar", [0.3, 0.7])
def test_limiar_muda_a_classificacao_mas_nao_o_modelo(teste, limiar):
    padrao = teste.executar(_df("mroz"), _params("mroz"))
    r = teste.executar(_df("mroz"), _params("mroz", limiar=limiar))
    assert r.estatisticas["lr"] == padrao.estatisticas["lr"]
    assert r.estatisticas["auc"] == padrao.estatisticas["auc"]
    grupo = next(f for f in r.figuras if isinstance(f, GrupoFiguras))
    p = np.asarray(grupo.opcoes[VALORES_AJUSTADOS].dados["x"])
    y = binarizar(_df("mroz"), "lfp", "yes")["lfp"].to_numpy()
    assert r.estatisticas["vp"] == int(((p >= limiar) & (y == 1)).sum())
    assert r.estatisticas["fp"] == int(((p >= limiar) & (y == 0)).sum())
    assert (
        f"Limiar de classificação: {limiar}".replace(".", ",")
        in _secao(r, "Classificação").textos[0]
    )


def test_trocar_o_evento_inverte_os_coeficientes(teste):
    sim_yes = teste.executar(_df("mroz"), _params("mroz")).simulacao
    sim_no = teste.executar(_df("mroz"), _params("mroz", evento="no")).simulacao
    assert list(sim_no.coeficientes) == pytest.approx(list(-sim_yes.coeficientes), rel=1e-6)


def test_separacao_completa_e_quase_completa():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert dg.separacao(x, np.array([0, 0, 0, 1, 1, 1])) == "completa"
    assert dg.separacao(np.array([1.0, 2, 3, 3, 5, 6]), np.array([0, 0, 0, 1, 1, 1])) == "quase"
    assert dg.separacao(x, np.array([0, 1, 0, 1, 0, 1])) is None


def test_aviso_de_separacao_no_resultado(teste):
    rng = np.random.default_rng(5)
    x = np.round(rng.uniform(0, 10, 40), 2)
    df = pd.DataFrame({"y": np.where(x > 5, "sim", "não"), "x": x, "z": rng.normal(size=40)})
    r = teste.executar(df, {"y": "y", "evento": "sim", "preditores": ["x", "z"], "alfa": 0.05})
    assert any("Separação completa" in a for a in r.avisos)


@pytest.mark.parametrize(
    ("extra", "trecho"),
    [
        ({"y": None}, "Selecione a variável dependente (y)."),
        ({"y": "age"}, "A variável dependente 'age' deve ter exatamente 2 valores"),
        ({"evento": "talvez"}, "Escolha qual valor da variável dependente é o evento."),
        ({"preditores": []}, "Selecione ao menos um preditor."),
        ({"preditores": ["age", "lfp"]}, "A variável dependente não pode ser também um preditor."),
        ({"limiar": 1.0}, "O limiar de classificação deve ser um número entre 0 e 1."),
        ({"limiar": "abc"}, "O limiar de classificação deve ser um número entre 0 e 1."),
        ({"confianca": "80%"}, "Escolha um nível de confiança válido."),
        ({"alfa": 0}, "O nível de significância (α) deve estar entre 0 e 1."),
    ],
)
def test_validacao_de_parametros(teste, extra, trecho):
    df = _df("mroz")
    erros = teste.validar(df, _params("mroz", **extra))
    assert any(trecho in e for e in erros), erros
    with pytest.raises(ErroValidacao):
        teste.executar(df, _params("mroz", **extra))


def test_validacao_do_modelo(teste):
    df = _df("mroz")
    df["k5_dobro"] = 2 * df["k5"]
    erros = teste.validar(df, _params("mroz", preditores=["k5", "k5_dobro"]))
    assert any("Colinearidade perfeita entre 'k5', 'k5_dobro'" in e for e in erros)
    df = _df("mroz")
    df.loc[df["lfp"] == "no", "age"] = np.nan  # só sobram casos "yes"
    assert "só tem a classe 'yes'" in teste.validar(df, _params("mroz", preditores=["age"]))[0]


def test_nao_rejeita_com_preditor_de_ruido(teste):
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"y": rng.choice(["a", "b"], 200), "x": rng.normal(size=200)})
    r = teste.executar(df, {"y": "y", "evento": "b", "preditores": ["x"], "alfa": 0.05})
    assert r.decisao == NAO_REJEITA_H0 and _secao(r, "Modelo").destaque == "Não rejeita H₀"
    assert "Colinearidade (VIF)" not in r.tabelas  # um único preditor


# ---------------------------------------------------------------------------
# Visualização e simulação
# ---------------------------------------------------------------------------


def test_figuras(teste):
    r = teste.executar(_df("birthwt"), _params("birthwt", limiar=0.4))
    roc, classes, grupo = r.figuras
    assert roc.tipo == "dispersao" and roc.dados["conectar"] is True
    assert (roc.dados["x"][0], roc.dados["y"][0]) == (0.0, 0.0)
    assert (roc.dados["x"][-1], roc.dados["y"][-1]) == (1.0, 1.0)
    assert "AUC = 0,734" in roc.titulo
    assert [g["rotulo"] for g in classes.dados["grupos"]] == ["'1'", "'0'"]
    assert classes.dados["referencias"][0]["valor"] == 0.4
    assert list(grupo.opcoes) == [VALORES_AJUSTADOS, "age", "lwt", "race", "smoke", "ht", "ui"]
    assert grupo.opcoes["race"].tipo == "boxplot"


def test_textos_da_simulacao(teste):
    sim = teste.executar(_df("birthwt"), _params("birthwt")).simulacao
    termos = sim.equacao()
    assert termos[0].texto == "logit(P) = 0,4372"
    assert sim.titulo_resultado == "Probabilidade prevista de 'low' = '1'"
    assert sim.rodape_equacao == "P = P('low' = '1') = 1 / (1 + e^(−logit))"
    p = sim.prever({"age": 25, "lwt": 120, "race": "negra", "smoke": "sim", "ht": 0, "ui": 1})
    valor, linhas, notas = sim.textos_previsao(p)
    assert valor == "77,4%"
    assert linhas[0][0] == "IC 95% da probabilidade: [49,2%; 92,4%]"
    assert linhas[1][0] == "Classe prevista (limiar 0,5): '1'"
    assert notas[0].startswith("Sem intervalo de predição")


def test_formulario(teste):
    specs = teste.parametros()
    assert [(s.nome, s.tipo) for s in specs] == [
        ("y", "coluna_binaria"),
        ("evento", "nivel"),
        ("preditores", "preditores"),
        ("referencias", "niveis_referencia"),
        ("limiar", "numero"),
        ("confianca", "opcao"),
        ("alfa", "alfa"),
    ]
    assert specs[1].depende_de == "y" and specs[2].depende_de == "y"
    assert specs[4].padrao == 0.5
