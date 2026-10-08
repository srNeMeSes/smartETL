"""ANOVA de 2 fatores (Tipo II e III, com e sem interação): checklist da seção 9 do CLAUDE.md.

Fontes dos valores de referência (independentes do wrapper, que usa statsmodels ols + anova_lm):
- somas de quadrados por comparação de modelos com mínimos quadrados em numpy (`lstsq`) e
  codificação por soma escrita à mão:
  Tipo II: SQ(A) = RSS(1 + B) − RSS(1 + A + B), SQ(B) idem, SQ(A × B) = RSS(1 + A + B) − RSS(cheio);
  Tipo III: SQ(termo) = RSS(cheio sem o termo) − RSS(cheio);
- desenho balanceado: fórmulas de livro-texto SQ(A) = b·r·Σ(ȳᵢ.. − ȳ)², SQ(B) = a·r·Σ(ȳ.ⱼ. − ȳ)²,
  SQ(A × B) = r·Σ(ȳᵢⱼ. − ȳᵢ.. − ȳ.ⱼ. + ȳ)², SQ resíduo = Σ(y − ȳᵢⱼ.)²;
- p pela cauda da F como beta incompleta regularizada (`scipy.special.betainc`);
- Levene (Brown-Forsythe): ANOVA à mão sobre |y − mediana da combinação|.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
from conftest import BASES
from scipy import special

from core.base import ErroValidacao
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.io import carregar_dados
from core.testes.anova import TABELA_MEDIAS, TIPO_II, TIPO_III, TesteAnova2Fatores

REL = 1e-9
# Balanceado: 3 níveis de A × 2 de B, 3 repetições, com interação.
BALANCEADO = {
    ("a1", "b1"): [10.2, 11.5, 9.8],
    ("a1", "b2"): [12.1, 13.0, 12.6],
    ("a2", "b1"): [14.3, 15.1, 13.7],
    ("a2", "b2"): [20.2, 19.4, 21.0],
    ("a3", "b1"): [11.0, 10.4, 12.2],
    ("a3", "b2"): [11.8, 12.9, 11.1],
}
# Desbalanceado: tamanhos diferentes por combinação (Tipo II ≠ Tipo III).
DESBALANCEADO = {
    ("a1", "b1"): [10.2, 11.5, 9.8, 10.9],
    ("a1", "b2"): [12.1, 13.0],
    ("a2", "b1"): [14.3, 15.1, 13.7],
    ("a2", "b2"): [20.2, 19.4, 21.0, 18.8, 20.5],
    ("a3", "b1"): [11.0, 10.4],
    ("a3", "b2"): [11.8, 12.9, 11.1],
}


@pytest.fixture
def teste():
    return TesteAnova2Fatores()


def _df(celulas):
    linhas = [(v, a, b) for (a, b), valores in celulas.items() for v in valores]
    return pd.DataFrame(linhas[::-1], columns=["y", "A", "B"])


def _params(**extra):
    base = {
        "coluna": "y",
        "fator_a": "A",
        "fator_b": "B",
        "interacao": True,
        "tipo_sq": TIPO_II,
        "alfa": 0.05,
    }
    return base | extra


def _cauda_f(f, d1, d2):
    return float(special.betainc(d2 / 2, d1 / 2, d2 / (d2 + d1 * f)))


def _soma(codigos, k):
    """Colunas da codificação por soma: indicador(i) − indicador(último), i = 0..k − 2."""
    x = np.zeros((len(codigos), k - 1))
    for linha, c in enumerate(codigos):
        if c == k - 1:
            x[linha, :] = -1
        else:
            x[linha, c] = 1
    return x


def _rss(y, *blocos):
    x = np.column_stack([np.ones(len(y)), *blocos])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(((y - x @ coef) ** 2).sum()), x.shape[1]


def _manual(celulas, tipo, interacao=True):
    df = _df(celulas)
    niveis_a, niveis_b = sorted(df["A"].unique()), sorted(df["B"].unique())
    y = df["y"].to_numpy()
    xa = _soma([niveis_a.index(v) for v in df["A"]], len(niveis_a))
    xb = _soma([niveis_b.index(v) for v in df["B"]], len(niveis_b))
    xab = np.column_stack(
        [xa[:, i] * xb[:, j] for i in range(xa.shape[1]) for j in range(xb.shape[1])]
    )
    cheio = [xa, xb, xab] if interacao else [xa, xb]
    rss_cheio, p_cheio = _rss(y, *cheio)
    gl_res = len(y) - p_cheio
    if tipo == TIPO_II:
        aditivo, _ = _rss(y, xa, xb)
        sq = {"a": _rss(y, xb)[0] - aditivo, "b": _rss(y, xa)[0] - aditivo}
        if interacao:
            sq["ab"] = aditivo - rss_cheio
    else:
        sq = {}
        for chave, i in (("a", 0), ("b", 1), ("ab", 2))[: len(cheio)]:
            sq[chave] = _rss(y, *(c for k, c in enumerate(cheio) if k != i))[0] - rss_cheio
    gl = {"a": xa.shape[1], "b": xb.shape[1], "ab": xab.shape[1]}
    resultado = {"sq_residuo": rss_cheio, "gl_residuo": gl_res}
    for chave, valor in sq.items():
        f = (valor / gl[chave]) / (rss_cheio / gl_res)
        resultado |= {
            f"sq_{chave}": valor,
            f"gl_{chave}": gl[chave],
            f"f_{chave}": f,
            f"p_{chave}": _cauda_f(f, gl[chave], gl_res),
            f"eta2p_{chave}": valor / (valor + rss_cheio),
        }
    return resultado


def _br(valor, casas=4):
    return f"{valor:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _p_br(p):
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Valores de referência
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "celulas", [BALANCEADO, DESBALANCEADO], ids=["balanceado", "desbalanceado"]
)
@pytest.mark.parametrize("tipo", [TIPO_II, TIPO_III])
@pytest.mark.parametrize("interacao", [True, False])
def test_somas_de_quadrados_contra_lstsq(teste, celulas, tipo, interacao):
    r = teste.executar(_df(celulas), _params(tipo_sq=tipo, interacao=interacao))
    esperado = _manual(celulas, tipo, interacao)
    for chave, valor in esperado.items():
        rel = 1e-6 if chave.startswith("p_") else 1e-7
        assert r.estatisticas[chave] == pytest.approx(valor, rel=rel), chave
    assert ("f_ab" in r.estatisticas) is interacao


def test_balanceado_contra_formulas_de_livro(teste):
    r = teste.executar(_df(BALANCEADO), _params())
    df = _df(BALANCEADO)
    geral = df["y"].mean()
    m_a, m_b = df.groupby("A")["y"].mean(), df.groupby("B")["y"].mean()
    m_ab = df.groupby(["A", "B"])["y"].mean()
    r_rep, a, b = 3, 3, 2
    sq_a = b * r_rep * ((m_a - geral) ** 2).sum()
    sq_b = a * r_rep * ((m_b - geral) ** 2).sum()
    sq_ab = r_rep * sum(
        (m_ab[(i, j)] - m_a[i] - m_b[j] + geral) ** 2
        for i, j in itertools.product(m_a.index, m_b.index)
    )
    sq_res = sum(((v - np.mean(vs)) ** 2) for vs in BALANCEADO.values() for v in vs)
    e = r.estatisticas
    assert (e["sq_a"], e["sq_b"], e["sq_ab"], e["sq_residuo"]) == pytest.approx(
        (sq_a, sq_b, sq_ab, sq_res), rel=REL
    )
    assert (e["gl_a"], e["gl_b"], e["gl_ab"], e["gl_residuo"]) == (2, 1, 2, 12)
    assert e["r2"] == pytest.approx((sq_a + sq_b + sq_ab) / ((df["y"] - geral) ** 2).sum(), rel=REL)
    # balanceado: Tipo II e Tipo III coincidem
    r3 = teste.executar(_df(BALANCEADO), _params(tipo_sq=TIPO_III))
    for chave in ("sq_a", "sq_b", "sq_ab"):
        assert r3.estatisticas[chave] == pytest.approx(e[chave], rel=1e-7)


def test_desbalanceado_tipos_diferem_e_aviso(teste):
    r2 = teste.executar(_df(DESBALANCEADO), _params(tipo_sq=TIPO_II))
    r3 = teste.executar(_df(DESBALANCEADO), _params(tipo_sq=TIPO_III))
    assert r2.estatisticas["sq_a"] != pytest.approx(r3.estatisticas["sq_a"], rel=1e-3)
    assert r2.estatisticas["balanceado"] == 0.0
    assert any("Desenho desbalanceado" in a and "Tipo III" in a for a in r3.avisos)
    assert not any("desbalanceado" in a for a in teste.executar(_df(BALANCEADO), _params()).avisos)


def test_levene_entre_combinacoes(teste):
    r = teste.executar(_df(BALANCEADO), _params())
    desvios = [[abs(v - float(np.median(vs))) for v in vs] for vs in BALANCEADO.values()]
    todos = [v for vs in desvios for v in vs]
    geral = np.mean(todos)
    sqe = sum(len(vs) * (np.mean(vs) - geral) ** 2 for vs in desvios)
    sqd = sum((v - np.mean(vs)) ** 2 for vs in desvios for v in vs)
    k, n = len(desvios), len(todos)
    f = (sqe / (k - 1)) / (sqd / (n - k))
    assert r.estatisticas["p_levene"] == pytest.approx(_cauda_f(f, k - 1, n - k), rel=1e-7)


# ---------------------------------------------------------------------------
# Decisão, interpretação e saída
# ---------------------------------------------------------------------------


def test_decisao_pela_interacao(teste):
    r = teste.executar(_df(BALANCEADO), _params())
    assert r.p_valor == r.estatisticas["p_ab"] and r.decisao == REJEITA_H0
    assert r.interpretacao.startswith("Com α = 0,05: Interação 'A' × 'B': ")
    assert "rejeita-se H₀ (o efeito de 'A' sobre 'y' não depende de 'B')" in r.interpretacao
    assert "Efeito de 'A': " in r.interpretacao and "Efeito de 'B': " in r.interpretacao
    assert "Há evidência estatística de interação" in r.interpretacao
    assert "com cautela" in r.interpretacao


def test_decisao_sem_interacao(teste):
    r = teste.executar(_df(BALANCEADO), _params(interacao=False))
    e = r.estatisticas
    # Dois efeitos para uma decisão: Bonferroni (p × 2, limitado a 1).
    assert e["p_bonferroni_a"] == pytest.approx(min(1.0, 2 * e["p_a"]))
    assert e["p_bonferroni_b"] == pytest.approx(min(1.0, 2 * e["p_b"]))
    assert r.p_valor == min(e["p_bonferroni_a"], e["p_bonferroni_b"]) and r.decisao == REJEITA_H0
    assert "Interação" not in r.interpretacao
    assert "Efeito de 'A': p ajustado (Bonferroni) " in r.interpretacao
    anova = r.tabelas["Tabela ANOVA"]
    assert anova["p ajustado (Bonferroni)"].tolist() == [
        _p_br(e["p_bonferroni_a"]),
        _p_br(e["p_bonferroni_b"]),
        "",
    ]
    resumo = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert resumo["Correção (2 efeitos principais)"] == "Bonferroni: p ajustado = p × 2"
    assert "a média de 'y' difere entre os níveis de 'A' e de 'B'." in r.interpretacao
    rng = np.random.default_rng(3)
    ruido = {chave: list(rng.normal(10, 1, 4)) for chave in BALANCEADO}
    r = teste.executar(_df(ruido), _params(interacao=False))
    assert r.decisao == NAO_REJEITA_H0
    assert "Não há evidência suficiente de efeito de 'A' ou de 'B'" in r.interpretacao


def test_bonferroni_muda_a_decisao_entre_alfa_e_alfa_sobre_2(teste, monkeypatch):
    # p = 0,04 em A sem interação: sem correção rejeitaria; com Bonferroni (0,08) não rejeita.
    from core.testes import anova as modulo

    r = teste.executar(_df(BALANCEADO), _params(interacao=False))
    e = dict(r.estatisticas, p_a=0.04, p_b=0.5, p_bonferroni_a=0.08, p_bonferroni_b=1.0)
    texto = modulo.TesteAnova2Fatores._interpretacao(e, 0.05, "y", "A", "B", False)
    assert "Efeito de 'A': p ajustado (Bonferroni) = 0,080, não se rejeita H₀" in texto
    assert "Não há evidência suficiente de efeito" in texto
    assert modulo.bonferroni(0.6, 2) == 1.0 and modulo.bonferroni(0.02, 2) == 0.04
    r = teste.executar(_df(BALANCEADO), _params())  # com interação: p sem ajuste
    assert "p ajustado (Bonferroni)" not in r.tabelas["Tabela ANOVA"].columns


def test_tabelas_figura_e_sem_card(teste):
    r = teste.executar(_df(BALANCEADO), _params())
    m = _manual(BALANCEADO, TIPO_II)
    anova = r.tabelas["Tabela ANOVA"]
    assert anova["Fonte"].tolist() == ["'A'", "'B'", "'A' × 'B'", "Resíduo"]
    assert anova["SQ"].tolist() == [_br(m[f"sq_{c}"]) for c in ("a", "b", "ab", "residuo")]
    assert anova["gl"].tolist() == ["2", "1", "2", "12"]
    assert anova["p"].tolist()[:3] == [_p_br(m[f"p_{c}"]) for c in ("a", "b", "ab")]
    assert anova["η² parcial"].tolist()[:3] == [_br(m[f"eta2p_{c}"]) for c in ("a", "b", "ab")]
    medias = r.tabelas[TABELA_MEDIAS]
    assert medias.columns.tolist() == ["A (A)", "B (B)", "n", "Média", "Desvio padrão"]
    assert list(zip(medias["A (A)"], medias["B (B)"], strict=True)) == list(BALANCEADO)
    assert medias["Média"].tolist() == [_br(np.mean(v)) for v in BALANCEADO.values()]
    assert medias["Desvio padrão"].tolist() == [_br(np.std(v, ddof=1)) for v in BALANCEADO.values()]
    resumo = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert resumo["Desenho"] == "Balanceado" and resumo["Interação A × B"] == "Incluída"
    (figura,) = r.figuras
    assert figura.tipo == "barras_agrupadas" and figura.dados["series"] == ["b1", "b2"]
    assert [g["rotulo"] for g in figura.dados["grupos"]] == ["a1", "a2", "a3"]
    assert figura.dados["grupos"][1]["valores"] == pytest.approx(
        [np.mean(BALANCEADO[("a2", "b1")]), np.mean(BALANCEADO[("a2", "b2")])]
    )
    assert r.comparacao is None and teste.comparacao_inicial() is None


def test_sem_interacao_aceita_uma_observacao_por_combinacao(teste):
    uma = {chave: valores[:1] for chave, valores in BALANCEADO.items()}
    df = _df(uma)
    assert any("ao menos 2 observações" in e for e in teste.validar(df, _params()))
    r = teste.executar(df, _params(interacao=False))
    assert r.estatisticas["gl_residuo"] == 2
    medias = r.tabelas[TABELA_MEDIAS]
    assert set(medias["Desvio padrão"]) == {"—"}
    assert np.isnan(r.estatisticas["p_levene"])


def test_linhas_incompletas(teste):
    df = pd.concat(
        [_df(BALANCEADO), pd.DataFrame({"y": [np.nan, 1.0], "A": ["a1", None], "B": ["b1", "b1"]})]
    )
    r = teste.executar(df, _params())
    assert r.estatisticas["n"] == 18
    assert "2 linha(s) com valor ou fator ausente foram descartadas." in r.avisos


def test_base_do_projeto(teste):
    df = carregar_dados(str(BASES / "canteiros_br.csv")).df
    r = teste.executar(df, _params(coluna="producao", fator_a="irrigacao", fator_b="dose_adubo"))
    assert (r.estatisticas["niveis_a"], r.estatisticas["niveis_b"], r.estatisticas["n"]) == (
        3,
        2,
        30,
    )
    assert r.decisao == REJEITA_H0 and r.estatisticas["balanceado"] == 1.0


# ---------------------------------------------------------------------------
# Entradas inválidas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "mensagem"),
    [
        ({"coluna": None}, "Selecione a variável."),
        ({"fator_a": None}, "Selecione o fator A."),
        ({"fator_b": None}, "Selecione o fator B."),
        ({"coluna": "A"}, "A coluna 'A' não é numérica."),
        ({"fator_b": "nao_existe"}, "A coluna 'nao_existe' não existe no arquivo."),
        ({"fator_b": "A"}, "A variável e os dois fatores devem ser colunas diferentes."),
        ({"tipo_sq": "Tipo I"}, "Escolha o tipo de soma de quadrados."),
        ({"alfa": 0}, "O nível de significância (α) deve estar entre 0 e 1."),
    ],
)
def test_validacao(teste, params, mensagem):
    df = _df(BALANCEADO)
    assert mensagem in teste.validar(df, _params(**params))
    with pytest.raises(ErroValidacao):
        teste.executar(df, _params(**params))


def test_validacao_dados(teste):
    um_nivel = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "A": list("xxxx"), "B": list("ppqq")})
    assert any(
        "O fator 'A' deve ter de 2 a 20 níveis" in e for e in teste.validar(um_nivel, _params())
    )
    sem_celula = {k: v for k, v in BALANCEADO.items() if k != ("a3", "b2")}
    assert any(
        "vazias: 'a3' × 'b2' (0)" in e
        for e in teste.validar(_df(sem_celula), _params(interacao=False))
    )
    assert any("com menos: 'a3' × 'b2' (0)" in e for e in teste.validar(_df(sem_celula), _params()))
    minimo = pd.DataFrame({"y": [1.0, 2.0, 3.0], "A": ["x", "y", "x"], "B": ["p", "p", "q"]})
    assert any("vazias" in e for e in teste.validar(minimo, _params(interacao=False)))
    justo = pd.DataFrame({"y": [1.0, 2.0, 3.0, 5.0], "A": list("xyxy"), "B": list("ppqq")})
    assert teste.validar(justo, _params(interacao=False)) == []
    aditivo = pd.DataFrame({"y": [1.0, 3.0, 2.0, 4.0], "A": list("xyxy"), "B": list("ppqq")})
    assert any("variância residual" in e for e in teste.validar(aditivo, _params(interacao=False)))
    constantes = _df({chave: [5.0, 5.0] for chave in BALANCEADO})
    assert any("variância residual" in e for e in teste.validar(constantes, _params()))


def test_formulario(teste):
    specs = teste.parametros()
    assert [(s.nome, s.tipo, s.padrao) for s in specs] == [
        ("coluna", "coluna_numerica", None),
        ("fator_a", "coluna_categorica", None),
        ("fator_b", "coluna_categorica", None),
        ("interacao", "booleano", True),
        ("tipo_sq", "opcao", TIPO_II),
        ("alfa", "alfa", 0.05),
    ]
    assert specs[4].opcoes == [TIPO_II, TIPO_III]
    assert teste.validar(_df(BALANCEADO), _params()) == []
