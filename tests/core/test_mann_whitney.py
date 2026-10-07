"""Mann-Whitney U: checklist da seção 9 do CLAUDE.md.

Fontes dos valores de referência (independentes do wrapper, que usa scipy):
- p exato: enumeração das C(N, n₁) divisões dos postos (tests/referencias.py);
- p aproximado: z com correção de empates, σ² = n₁n₂/12·[(N + 1) − Σ(t³ − t)/(N(N − 1))], e
  correção de continuidade 0,5, com a normal de `statistics.NormalDist`;
- distribuição exata de U contra a enumeração; deslocamento de Hodges-Lehmann como mediana das
  diferenças com numpy e IC exato pelos quantis enumerados (método do `wilcox.test` do R);
- probabilidade de superioridade e r = z/√N pelas fórmulas.
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from referencias import distribuicao_u_enumerada, mann_whitney_exato
from scipy.stats import rankdata

from core.base import ErroValidacao
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.nao_parametricos import ALTERNATIVAS_MW, TesteMannWhitney, distribuicao_u

REL = 1e-9
A = [12.1, 14.3, 11.8, 15.2, 13.7]
B = [16.4, 15.9, 17.2, 14.8, 18.1, 16.7]


@pytest.fixture
def teste():
    return TesteMannWhitney()


def _df(x, y, g=("A", "B")):
    # O grupo "B" vem primeiro nos dados; o grupo 1 continua sendo "A" (ordem crescente).
    return pd.DataFrame({"y": list(y) + list(x), "g": [g[1]] * len(y) + [g[0]] * len(x)})


def _params(**extra):
    return {"coluna": "y", "grupo": "g", "alternativa": "G₁ ≠ G₂", "alfa": 0.05} | extra


def _rotulo(alternativa: str) -> str:
    return next(r for r, a in ALTERNATIVAS_MW.items() if a == alternativa)


def _aproximado_manual(x, y, alternativa):
    postos = rankdata(np.concatenate([x, y]))
    n1, n2 = len(x), len(y)
    total = n1 + n2
    u1 = postos[:n1].sum() - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    _, t = np.unique(postos, return_counts=True)
    sigma = math.sqrt(n1 * n2 / 12 * ((total + 1) - (t**3 - t).sum() / (total * (total - 1))))
    mu = n1 * n2 / 2
    sf = lambda u: 1 - NormalDist().cdf((u - mu - 0.5) / sigma)  # noqa: E731
    if alternativa == "greater":
        return sf(u1)
    if alternativa == "less":
        return sf(u2)
    return min(1.0, 2 * sf(max(u1, u2)))


# ---------------------------------------------------------------------------
# Valores de referência
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_p_exato_por_enumeracao(teste, alternativa):
    r = teste.executar(_df(A, B), _params(alternativa=_rotulo(alternativa)))
    bilateral, maior, menor = mann_whitney_exato(np.array(A), np.array(B))
    assert r.estatisticas["usou_exato"] == 1.0
    assert r.p_valor == pytest.approx(
        {"two-sided": bilateral, "greater": maior, "less": menor}[alternativa], rel=REL
    )


@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_p_aproximado_com_empates(teste, alternativa):
    x = [3, 5, 5, 7, 8, 9, 9, 10, 12, 4]
    y = [5, 6, 8, 9, 11, 12, 12, 13, 14, 15, 15]
    r = teste.executar(_df(x, y), _params(alternativa=_rotulo(alternativa)))
    assert r.estatisticas["usou_exato"] == 0.0
    assert r.p_valor == pytest.approx(
        _aproximado_manual(np.array(x, float), np.array(y, float), alternativa), rel=1e-7
    )


def test_amostras_grandes_sem_empates_usam_aproximacao(teste):
    rng = np.random.default_rng(4)
    x, y = rng.normal(10, 2, 20), rng.normal(11, 2, 25)
    r = teste.executar(_df(x, y), _params())
    assert r.estatisticas["usou_exato"] == 0.0
    assert r.p_valor == pytest.approx(_aproximado_manual(x, y, "two-sided"), rel=1e-7)


def test_u_somas_de_postos_e_efeitos(teste):
    r = teste.executar(_df(A, B), _params())
    e = r.estatisticas
    postos = rankdata(A + B)
    r1 = postos[:5].sum()
    assert (e["soma_postos1"], e["soma_postos2"]) == (r1, postos[5:].sum())
    assert e["u1"] == r1 - 15 and e["u1"] + e["u2"] == 30
    assert e["prob_superioridade"] == pytest.approx(e["u1"] / 30)
    sigma = math.sqrt(30 * 12 / 12)
    assert e["z"] == pytest.approx((e["u1"] - 15) / sigma, rel=REL)
    assert e["r"] == pytest.approx(e["z"] / math.sqrt(11), rel=REL)


@pytest.mark.parametrize(("n1", "n2"), [(1, 1), (2, 3), (4, 4), (5, 6)])
def test_distribuicao_u_contra_enumeracao(n1, n2):
    enumerada = distribuicao_u_enumerada(range(1, n1 + n2 + 1), n1)
    esperado = [enumerada.count(u) for u in range(n1 * n2 + 1)]
    assert distribuicao_u(n1, n2).tolist() == pytest.approx(esperado)


@pytest.mark.parametrize("alfa", [0.01, 0.05, 0.10])
@pytest.mark.parametrize("alternativa", ["two-sided", "greater", "less"])
def test_hodges_lehmann_e_ic_exato(teste, alfa, alternativa):
    r = teste.executar(_df(A, B), _params(alfa=alfa, alternativa=_rotulo(alternativa)))
    dif = sorted(a - b for a in A for b in B)
    assert r.estatisticas["deslocamento_hl"] == pytest.approx(float(np.median(dif)))
    us = sorted(distribuicao_u_enumerada(range(1, 12), 5))
    cauda = alfa / 2 if alternativa == "two-sided" else alfa
    qu = next(u for u in range(31) if sum(1 for v in us if v <= u) / len(us) >= cauda - 1e-12)
    qu = max(qu, 1)
    esperado = {
        "two-sided": (dif[qu - 1], dif[30 - qu]),
        "greater": (dif[qu - 1], math.inf),
        "less": (-math.inf, dif[30 - qu]),
    }[alternativa]
    assert (r.estatisticas["ic_inferior"], r.estatisticas["ic_superior"]) == pytest.approx(esperado)


# ---------------------------------------------------------------------------
# Decisão, interpretação e saída
# ---------------------------------------------------------------------------


def test_decisao_e_interpretacao(teste):
    r = teste.executar(_df(A, B), _params())
    assert r.decisao == REJEITA_H0  # p ≈ 0,0087
    assert "a distribuição de 'y' é a mesma em 'A' e 'B'" in r.interpretacao
    assert "Há evidência estatística de que 'y' difere entre 'A' e 'B'." in r.interpretacao
    r = teste.executar(_df(A, B), _params(alternativa="G₁ < G₂"))
    assert r.decisao == REJEITA_H0
    assert "'y' tende a ser maior no grupo 'B' do que no grupo 'A'" in r.interpretacao
    r = teste.executar(_df(A, B), _params(alternativa="G₁ > G₂"))
    assert r.decisao == NAO_REJEITA_H0
    assert (
        "Não há evidência suficiente de que 'y' tenda a ser maior no grupo 'A'" in r.interpretacao
    )


def test_resultado_completo(teste):
    r = teste.executar(_df(A, B), _params())
    medidas = dict(r.tabelas["Resumo"].itertuples(index=False, name=None))
    assert medidas["Grupo 1"] == "'A' (n = 5)" and medidas["Grupo 2"] == "'B' (n = 6)"
    assert (medidas["U₁"], medidas["U₂"]) == ("1,0", "29,0")
    assert medidas["Método do p-valor"] == "Exato"
    assert medidas["Deslocamento de Hodges-Lehmann (G₁ − G₂)"] == "-2,9500"
    assert "IC 95% para o deslocamento" in medidas
    (figura,) = r.figuras
    assert figura.tipo == "boxplot"
    assert [g["rotulo"] for g in figura.dados["grupos"]] == ["A", "B"]
    assert r.comparacao is None and teste.comparacao_inicial() is None  # sem card
    assert r.avisos == []


def test_linhas_incompletas_e_empates(teste):
    df = pd.concat(
        [_df([3, 5, 5, 7], [6, 8, 9, 9]), pd.DataFrame({"y": [np.nan, 4.0], "g": ["A", None]})]
    )
    r = teste.executar(df, _params())
    assert (r.estatisticas["n1"], r.estatisticas["n2"]) == (4, 4)
    assert "2 linha(s) com valor ou grupo ausente foram descartadas." in r.avisos
    assert any("Há empates entre os valores" in a for a in r.avisos)


# ---------------------------------------------------------------------------
# Entradas inválidas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "mensagem"),
    [
        ({"coluna": None}, "Selecione a variável."),
        ({"grupo": None}, "Selecione o grupo."),
        ({"coluna": "g"}, "A coluna 'g' não é numérica."),
        ({"grupo": "y"}, "A variável e o grupo devem ser colunas diferentes."),
        ({"alternativa": "μ ≠ μ₀"}, "Escolha uma hipótese alternativa válida."),
        ({"alfa": 0}, "O nível de significância (α) deve estar entre 0 e 1."),
    ],
)
def test_validacao(teste, params, mensagem):
    df = _df(A, B)
    assert mensagem in teste.validar(df, _params(**params))
    with pytest.raises(ErroValidacao):
        teste.executar(df, _params(**params))


def test_validacao_dados(teste):
    tres = pd.DataFrame({"y": [1, 2, 3, 4, 5, 6], "g": list("aabbcc")})
    assert any("exatamente 2 níveis" in e for e in teste.validar(tres, _params()))
    iguais = pd.DataFrame({"y": [5.0] * 4, "g": list("aabb")})
    assert any("Todos os valores de 'y' são iguais" in e for e in teste.validar(iguais, _params()))


def test_parametros_validos_e_formulario(teste):
    assert teste.validar(_df(A, B), _params()) == []
    assert [(s.nome, s.tipo) for s in teste.parametros()] == [
        ("coluna", "coluna_numerica"),
        ("grupo", "coluna_binaria"),
        ("alternativa", "opcao"),
        ("alfa", "alfa"),
    ]


def test_h1_textual_como_o_h0(teste):
    r = teste.executar(_df(A, B), _params())
    assert "em favor de H₁ (a distribuição de 'y' difere entre 'A' e 'B')" in r.interpretacao
    r = teste.executar(_df(A, B), _params(alternativa="G₁ < G₂"))
    assert "em favor de H₁ ('y' tende a ser maior em 'B' do que em 'A')" in r.interpretacao
    assert "G₁" not in r.interpretacao
