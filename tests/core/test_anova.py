"""ANOVA de 1 fator (clássica e Welch, com Tukey HSD): checklist da seção 9 do CLAUDE.md.

Fontes dos valores de referência (independentes do wrapper, que usa scipy.stats.f_oneway):
- SQ entre/dentro, QM, F, η² e ω² pelas fórmulas de livro-texto, em Python puro;
- p pela cauda da F escrita como beta incompleta regularizada,
  P(F > f) = I_{d₂/(d₂ + d₁f)}(d₂/2, d₁/2) (`scipy.special.betainc`), e pela tabela ANOVA do
  statsmodels (`ols` + `anova_lm`);
- Welch: F* = [Σ wᵢ(x̄ᵢ − x̄_w)²/(k − 1)] / [1 + 2(k − 2)/(k² − 1)·Σ(1 − wᵢ/W)²/(nᵢ − 1)],
  wᵢ = nᵢ/sᵢ², gl₂ = (k² − 1)/(3·Σ(1 − wᵢ/W)²/(nᵢ − 1)), conferido com
  `statsmodels.stats.oneway.anova_oneway(use_var="unequal")`;
- Tukey-Kramer: q = |x̄ᵢ − x̄ⱼ|/√(QMD/2·(1/nᵢ + 1/nⱼ)) e a amplitude studentizada
  (`scipy.stats.studentized_range`); diferenças conferidas com `pairwise_tukeyhsd` do statsmodels;
- Levene (Brown-Forsythe): ANOVA à mão sobre |x − mediana do grupo|;
- IC da média de cada grupo: x̄ ± t·s/√n.
"""

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from conftest import BASES
from scipy import special
from scipy.stats import kruskal, studentized_range, t
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

from core.base import ErroValidacao
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.io import carregar_dados
from core.testes.anova import CLASSICA, TABELA_TUKEY, WELCH, TesteAnova1Fator

REL = 1e-9
# Grupos com tamanhos e variâncias diferentes (Welch ≠ clássica).
GRUPOS = {
    "A": [20.1, 22.4, 19.8, 21.7, 23.0, 20.9],
    "B": [24.3, 27.9, 22.1, 29.4, 25.6],
    "C": [21.2, 22.0, 21.5, 22.8, 21.9, 22.3, 21.4],
}


@pytest.fixture
def teste():
    return TesteAnova1Fator()


def _df(grupos):
    linhas = [(v, g) for g, valores in grupos.items() for v in valores]
    return pd.DataFrame(linhas[::-1], columns=["y", "g"])


def _params(**extra):
    return {"coluna": "y", "grupo": "g", "variante": CLASSICA, "tukey": False, "alfa": 0.05} | extra


def _media(x):
    return sum(x) / len(x)


def _var(x):
    m = _media(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)


def _cauda_f(f, d1, d2):
    return float(special.betainc(d2 / 2, d1 / 2, d2 / (d2 + d1 * f)))


def _anova_manual(grupos):
    todos = [v for vs in grupos.values() for v in vs]
    geral = _media(todos)
    sqe = sum(len(vs) * (_media(vs) - geral) ** 2 for vs in grupos.values())
    sqd = sum((v - _media(vs)) ** 2 for vs in grupos.values() for v in vs)
    k, n = len(grupos), len(todos)
    qmd = sqd / (n - k)
    f = (sqe / (k - 1)) / qmd
    return {
        "sqe": sqe,
        "sqd": sqd,
        "qmd": qmd,
        "f": f,
        "p": _cauda_f(f, k - 1, n - k),
        "eta2": sqe / (sqe + sqd),
        "omega2": (sqe - (k - 1) * qmd) / (sqe + sqd + qmd),
    }


def _welch_manual(grupos):
    vs = list(grupos.values())
    k = len(vs)
    w = [len(x) / _var(x) for x in vs]
    soma_w = sum(w)
    media_w = sum(wi * _media(x) for wi, x in zip(w, vs, strict=True)) / soma_w
    num = sum(wi * (_media(x) - media_w) ** 2 for wi, x in zip(w, vs, strict=True)) / (k - 1)
    lam = sum((1 - wi / soma_w) ** 2 / (len(x) - 1) for wi, x in zip(w, vs, strict=True))
    f = num / (1 + 2 * (k - 2) / (k**2 - 1) * lam)
    gl2 = (k**2 - 1) / (3 * lam)
    return f, gl2, _cauda_f(f, k - 1, gl2)


def _br(valor, casas=4):
    return f"{valor:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _p_br(p):
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Valores de referência
# ---------------------------------------------------------------------------


def test_classica_contra_formula(teste):
    r = teste.executar(_df(GRUPOS), _params())
    m = _anova_manual(GRUPOS)
    e = r.estatisticas
    assert (e["k"], e["n"], e["gl1"], e["gl2"]) == (3, 18, 2, 15)
    assert e["sq_entre"] == pytest.approx(m["sqe"], rel=REL)
    assert e["sq_dentro"] == pytest.approx(m["sqd"], rel=REL)
    assert e["qm_dentro"] == pytest.approx(m["qmd"], rel=REL)
    assert e["f"] == pytest.approx(m["f"], rel=REL)
    assert r.p_valor == pytest.approx(m["p"], rel=1e-7)
    assert e["eta2"] == pytest.approx(m["eta2"], rel=REL)
    assert e["omega2"] == pytest.approx(m["omega2"], rel=REL)


def test_classica_contra_statsmodels(teste):
    df = _df(GRUPOS)
    r = teste.executar(df, _params())
    tabela = anova_lm(smf.ols("y ~ C(g)", data=df).fit())
    assert r.estatisticas["sq_entre"] == pytest.approx(tabela.loc["C(g)", "sum_sq"], rel=1e-7)
    assert r.estatisticas["f"] == pytest.approx(tabela.loc["C(g)", "F"], rel=1e-7)
    assert r.p_valor == pytest.approx(tabela.loc["C(g)", "PR(>F)"], rel=1e-6)


def test_welch_contra_formula_e_statsmodels(teste):
    r = teste.executar(_df(GRUPOS), _params(variante=WELCH))
    f, gl2, p = _welch_manual(GRUPOS)
    e = r.estatisticas
    assert e["f"] == pytest.approx(f, rel=REL)
    assert e["gl1"] == 2 and e["gl2"] == pytest.approx(gl2, rel=REL)
    assert r.p_valor == pytest.approx(p, rel=1e-7)
    sm = anova_oneway([np.array(v) for v in GRUPOS.values()], use_var="unequal")
    assert e["f"] == pytest.approx(sm.statistic, rel=1e-7)
    assert r.p_valor == pytest.approx(sm.pvalue, rel=1e-6)
    # a decomposição clássica (tabela ANOVA e efeitos) continua disponível
    assert e["f_classico"] == pytest.approx(_anova_manual(GRUPOS)["f"], rel=REL)


def test_dois_grupos_equivale_ao_t(teste):
    # Com k = 2, F clássico = t² (pooled) e F de Welch = t² de Welch.
    grupos = {"x": GRUPOS["A"], "y": GRUPOS["B"]}
    a, b = (np.array(v) for v in grupos.values())
    sp2 = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    t_pooled = (a.mean() - b.mean()) / math.sqrt(sp2 * (1 / len(a) + 1 / len(b)))
    r = teste.executar(_df(grupos), _params())
    assert r.estatisticas["f"] == pytest.approx(t_pooled**2, rel=REL)
    t_welch = (a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    r = teste.executar(_df(grupos), _params(variante=WELCH))
    assert r.estatisticas["f"] == pytest.approx(t_welch**2, rel=REL)


@pytest.mark.parametrize("alfa", [0.01, 0.05, 0.10])
def test_tukey_kramer_contra_formula(teste, alfa):
    r = teste.executar(_df(GRUPOS), _params(tukey=True, alfa=alfa))
    tabela = r.tabelas[TABELA_TUKEY]
    assert tabela["Comparação"].tolist() == ["'A' × 'B'", "'A' × 'C'", "'B' × 'C'"]
    qmd, vs = _anova_manual(GRUPOS)["qmd"], list(GRUPOS.values())
    gl = sum(map(len, vs)) - len(vs)
    critico = studentized_range.ppf(1 - alfa, len(vs), gl)
    difs, ics, ps = [], [], []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        dif = _media(vs[i]) - _media(vs[j])
        ep = math.sqrt(qmd / 2 * (1 / len(vs[i]) + 1 / len(vs[j])))
        difs.append(_br(dif))
        ics.append(f"[{_br(dif - critico * ep)}; {_br(dif + critico * ep)}]")
        ps.append(float(studentized_range.sf(abs(dif) / ep, len(vs), gl)))
    confianca = f"{(1 - alfa) * 100:.0f}%"
    assert tabela["Diferença de médias"].tolist() == difs
    assert tabela[f"IC {confianca} (Tukey)"].tolist() == ics
    assert tabela["p ajustado (Tukey)"].tolist() == [_p_br(p) for p in ps]
    assert tabela["Diferem?"].tolist() == ["Sim" if p <= alfa else "Não" for p in ps]


def test_tukey_diferencas_iguais_ao_statsmodels(teste):
    df = _df(GRUPOS)
    r = teste.executar(df, _params(tukey=True))
    sm = pairwise_tukeyhsd(df["y"], df["g"])
    # statsmodels reporta grupo2 − grupo1; aqui é grupo1 − grupo2.
    assert r.tabelas[TABELA_TUKEY]["Diferença de médias"].tolist() == [
        _br(-d) for d in sm.meandiffs
    ]


def test_tabela_anova_e_grupos(teste):
    r = teste.executar(_df(GRUPOS), _params())
    m = _anova_manual(GRUPOS)
    anova = r.tabelas["Tabela ANOVA"]
    assert anova["Fonte"].tolist() == ["Entre grupos", "Dentro dos grupos", "Total"]
    assert anova["SQ"].tolist() == [_br(m["sqe"]), _br(m["sqd"]), _br(m["sqe"] + m["sqd"])]
    assert anova["gl"].tolist() == ["2", "15", "17"]
    assert anova["QM"].tolist()[:2] == [_br(m["sqe"] / 2), _br(m["qmd"])]
    assert anova["F"].tolist()[0] == _br(m["f"]) and anova["p"].tolist()[0] == _p_br(m["p"])
    grupos = r.tabelas["Grupos"]
    assert grupos["Grupo"].tolist() == ["A", "B", "C"] and grupos["n"].tolist() == [6, 5, 7]
    esperado = []
    for x in GRUPOS.values():
        margem = t.ppf(0.975, len(x) - 1) * math.sqrt(_var(x) / len(x))
        esperado.append(f"[{_br(_media(x) - margem)}; {_br(_media(x) + margem)}]")
    assert grupos["IC 95% da média"].tolist() == esperado
    assert grupos["Desvio padrão"].tolist() == [_br(math.sqrt(_var(x))) for x in GRUPOS.values()]


def test_levene_brown_forsythe_contra_formula(teste):
    r = teste.executar(_df(GRUPOS), _params())
    desvios = {g: [abs(v - float(np.median(vs))) for v in vs] for g, vs in GRUPOS.items()}
    assert r.estatisticas["p_levene"] == pytest.approx(_anova_manual(desvios)["p"], rel=1e-7)
    assert r.estatisticas["p_levene"] < 0.05
    assert any("Levene (centrado na mediana)" in a and "variante Welch" in a for a in r.avisos)


# ---------------------------------------------------------------------------
# Decisão, saída, card e avisos
# ---------------------------------------------------------------------------


def test_decisao_e_interpretacao(teste):
    r = teste.executar(_df(GRUPOS), _params())
    assert r.decisao == REJEITA_H0
    assert "rejeita-se H₀ (a média de 'y' é a mesma nos 3 grupos de 'g')" in r.interpretacao
    assert "em favor de H₁ (ao menos uma média difere das demais)" in r.interpretacao
    assert "a média de 'y' difere entre os grupos de 'g'." in r.interpretacao
    parecidos = {"A": [1.0, 2.0, 3.0, 4.0], "B": [1.5, 2.5, 3.5, 4.5]}
    r = teste.executar(_df(parecidos), _params())
    assert r.decisao == NAO_REJEITA_H0
    assert "Não há evidência suficiente" in r.interpretacao


def test_card_anova_x_kruskal(teste):
    inicial = teste.comparacao_inicial()
    assert (inicial.titulo_esquerda, inicial.titulo_direita) == ("ANOVA", "Kruskal-Wallis")
    assert inicial.hipoteses == ["algum μᵢ ≠ μⱼ"] and inicial.linhas == [(None, None)]
    for variante in (CLASSICA, WELCH):
        r = teste.executar(_df(GRUPOS), _params(variante=variante))
        p_kw = kruskal(*GRUPOS.values()).pvalue
        assert r.comparacao.linhas == [(r.p_valor, pytest.approx(p_kw, rel=REL))]


def test_resumo_figura_e_tukey_desligado(teste):
    r = teste.executar(_df(GRUPOS), _params(variante=WELCH))
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Variante"] == WELCH
    assert medidas["Graus de liberdade (numerador; denominador)"].startswith("2; ")
    for chave in ("F de Welch", "p-valor (Welch)", "η² (eta quadrado)", "ω² (ômega quadrado)"):
        assert chave in medidas
    assert TABELA_TUKEY not in r.tabelas
    (figura,) = r.figuras
    assert figura.tipo == "boxplot"
    assert [g["rotulo"] for g in figura.dados["grupos"]] == ["A", "B", "C"]


def test_tukey_nao_exibido_sem_rejeicao(teste):
    parecidos = {"A": [1.0, 2.0, 3.0, 4.0], "B": [1.5, 2.5, 3.5, 4.5]}
    r = teste.executar(_df(parecidos), _params(tukey=True))
    assert TABELA_TUKEY not in r.tabelas
    assert any("Tukey HSD) não exibidas" in a for a in r.avisos)


def test_avisos(teste):
    df = pd.concat([_df(GRUPOS), pd.DataFrame({"y": [np.nan, 1.0], "g": ["A", None]})])
    r = teste.executar(df, _params(variante=WELCH, tukey=True))
    assert r.estatisticas["n"] == 18
    assert "2 linha(s) com valor ou grupo ausente foram descartadas." in r.avisos
    assert any("a variante Welch é a adequada" in a for a in r.avisos)
    assert any("Tukey HSD assume variâncias iguais" in a for a in r.avisos)
    homogeneos = {"A": [1.0, 2.0, 3.0, 4.0], "B": [5.0, 6.0, 7.0, 8.0]}
    assert teste.executar(_df(homogeneos), _params()).avisos == []


def test_base_do_projeto(teste):
    df = carregar_dados(str(BASES / "fertilizantes_br.csv")).df
    params = _params(coluna="produtividade", grupo="fertilizante", tukey=True)
    r = teste.executar(df, params)
    grupos = {
        g: df.loc[df["fertilizante"] == g, "produtividade"].tolist()
        for g in sorted(df["fertilizante"].unique())
    }
    assert r.estatisticas["f"] == pytest.approx(_anova_manual(grupos)["f"], rel=REL)
    assert r.decisao == REJEITA_H0 and r.estatisticas["k"] == 4
    assert TABELA_TUKEY in r.tabelas


# ---------------------------------------------------------------------------
# Entradas inválidas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "mensagem"),
    [
        ({"coluna": None}, "Selecione a variável."),
        ({"grupo": None}, "Selecione o fator."),
        ({"coluna": "g"}, "A coluna 'g' não é numérica."),
        ({"coluna": "nao_existe"}, "A coluna 'nao_existe' não existe no arquivo."),
        ({"grupo": "y"}, "A variável e o fator devem ser colunas diferentes."),
        ({"variante": "Outra"}, "Escolha uma variante válida."),
        ({"alfa": 1.5}, "O nível de significância (α) deve estar entre 0 e 1."),
    ],
)
def test_validacao(teste, params, mensagem):
    df = _df(GRUPOS)
    assert mensagem in teste.validar(df, _params(**params))
    with pytest.raises(ErroValidacao):
        teste.executar(df, _params(**params))


def test_validacao_dados(teste):
    um = pd.DataFrame({"y": [1.0, 2.0], "g": ["a", "a"]})
    assert any("de 2 a 20 níveis" in e for e in teste.validar(um, _params()))
    pequeno = pd.DataFrame({"y": [1.0, 2.0, 3.0], "g": ["a", "a", "b"]})
    assert any(
        "ao menos 2 observações válidas; grupos com menos: 'b' (1)" in e
        for e in teste.validar(pequeno, _params())
    )
    constantes = pd.DataFrame({"y": [4.0, 4.0, 6.0, 6.0], "g": list("aabb")})
    assert any("dentro dos grupos é zero" in e for e in teste.validar(constantes, _params()))
    um_constante = pd.DataFrame({"y": [4.0, 4.0, 6.0, 7.0], "g": list("aabb")})
    assert teste.validar(um_constante, _params()) == []
    assert any(
        "Welch exige variância maior que zero" in e and "'a'" in e
        for e in teste.validar(um_constante, _params(variante=WELCH))
    )
    muitos = pd.DataFrame({"y": np.arange(42.0), "g": [f"g{i // 2}" for i in range(42)]})
    assert any("(tem 21)" in e for e in teste.validar(muitos, _params()))


def test_formulario(teste):
    specs = teste.parametros()
    assert [(s.nome, s.tipo, s.padrao) for s in specs] == [
        ("coluna", "coluna_numerica", None),
        ("grupo", "coluna_categorica", None),
        ("variante", "opcao", CLASSICA),
        ("tukey", "booleano", False),
        ("alfa", "alfa", 0.05),
    ]
    assert specs[2].opcoes == [CLASSICA, WELCH]
    assert teste.validar(_df(GRUPOS), _params()) == []
