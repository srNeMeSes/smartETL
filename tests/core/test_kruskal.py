"""Kruskal-Wallis (com pós-teste de Dunn): checklist da seção 9 do CLAUDE.md.

Fontes dos valores de referência (independentes do wrapper, que usa scipy):
- H = [12/(N(N+1))·Σ Rᵢ²/nᵢ − 3(N+1)] / [1 − Σ(t³ − t)/(N³ − N)], com postos médios à mão;
- p pela cauda fechada da qui-quadrado: gl = 2 → exp(−H/2); gl = 1 → erfc(√(H/2));
- Dunn: z = (R̄ᵢ − R̄ⱼ)/√{[N(N+1)/12 − Σ(t³ − t)/(12(N − 1))]·(1/nᵢ + 1/nⱼ)} e a normal de
  `statistics.NormalDist`; ajuste de Holm conferido com `statsmodels.stats.multitest`;
- ε² = H/(N − 1).
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from core.base import ErroValidacao
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.testes.nao_parametricos import TesteKruskalWallis, ajuste_holm

REL = 1e-9
GRUPOS = {
    "A": [12.0, 15.0, 11.0, 14.0, 13.0],
    "B": [18.0, 20.0, 17.0, 19.0, 16.0],
    "C": [13.0, 14.0, 12.0, 15.0, 16.0],
}


@pytest.fixture
def teste():
    return TesteKruskalWallis()


def _df(grupos):
    linhas = [(v, g) for g, valores in grupos.items() for v in valores]
    return pd.DataFrame(linhas[::-1], columns=["y", "g"])


def _params(**extra):
    return {"coluna": "y", "grupo": "g", "dunn": False, "alfa": 0.05} | extra


def _postos_medios(valores):
    ordenados = sorted(valores)
    return [
        (ordenados.index(v) + 1 + len(ordenados) - ordenados[::-1].index(v)) / 2 for v in valores
    ]


def _manual(grupos):
    todos = [v for vs in grupos.values() for v in vs]
    postos = _postos_medios(todos)
    n = len(todos)
    somas, inicio = [], 0
    for vs in grupos.values():
        somas.append(sum(postos[inicio : inicio + len(vs)]))
        inicio += len(vs)
    h = (
        12
        / (n * (n + 1))
        * sum(r**2 / len(vs) for r, vs in zip(somas, grupos.values(), strict=True))
    )
    h -= 3 * (n + 1)
    empates = [todos.count(v) for v in set(todos)]
    correcao = 1 - sum(t**3 - t for t in empates) / (n**3 - n)
    medios = [r / len(vs) for r, vs in zip(somas, grupos.values(), strict=True)]
    return h / correcao, medios, sum(t**3 - t for t in empates), n


# ---------------------------------------------------------------------------
# Valores de referência
# ---------------------------------------------------------------------------


def test_h_e_p_contra_formula_tres_grupos(teste):
    r = teste.executar(_df(GRUPOS), _params())
    h, _, _, n = _manual(GRUPOS)
    assert r.estatisticas["h"] == pytest.approx(h, rel=REL)
    assert r.estatisticas["gl"] == 2
    assert r.p_valor == pytest.approx(math.exp(-h / 2), rel=REL)
    assert r.estatisticas["epsilon2"] == pytest.approx(h / (n - 1), rel=REL)


def test_dois_grupos_gl_1(teste):
    grupos = {"x": [3.1, 4.5, 2.2, 5.8, 3.9], "y": [6.1, 7.4, 5.2, 8.0, 6.6, 7.1]}
    r = teste.executar(_df(grupos), _params())
    h, _, _, _ = _manual(grupos)
    assert r.estatisticas["h"] == pytest.approx(h, rel=REL)
    assert r.p_valor == pytest.approx(math.erfc(math.sqrt(h / 2)), rel=REL)


def test_dunn_contra_formula_e_holm_do_statsmodels(teste):
    r = teste.executar(_df(GRUPOS), _params(dunn=True))
    _, medios, soma_empates, n = _manual(GRUPOS)
    tamanhos = [len(v) for v in GRUPOS.values()]
    base = n * (n + 1) / 12 - soma_empates / (12 * (n - 1))
    pares = [(0, 1), (0, 2), (1, 2)]
    zs = [
        (medios[i] - medios[j]) / math.sqrt(base * (1 / tamanhos[i] + 1 / tamanhos[j]))
        for i, j in pares
    ]
    ps = [2 * (1 - NormalDist().cdf(abs(z))) for z in zs]
    holm = multipletests(ps, method="holm")[1]
    tabela = r.tabelas["Comparações múltiplas (Dunn, Holm)"]
    assert tabela["Comparação"].tolist() == ["'A' × 'B'", "'A' × 'C'", "'B' × 'C'"]
    assert tabela["z"].tolist() == [f"{z:.3f}".replace(".", ",") for z in zs]
    assert ajuste_holm(ps) == pytest.approx(list(holm), rel=REL)
    assert tabela["Diferem?"].tolist() == ["Sim" if p <= 0.05 else "Não" for p in holm]


def test_holm_casos_simples():
    assert ajuste_holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert ajuste_holm([0.5]) == [0.5]
    assert ajuste_holm([0.9, 0.8]) == [1.0, 1.0]


# ---------------------------------------------------------------------------
# Decisão, saída e pós-teste
# ---------------------------------------------------------------------------


def test_decisao_e_interpretacao(teste):
    r = teste.executar(_df(GRUPOS), _params())
    assert r.decisao == REJEITA_H0
    assert "rejeita-se H₀ (a distribuição de 'y' é a mesma nos 3 grupos de 'g')" in r.interpretacao
    assert "em favor de H₁ (ao menos um grupo difere dos demais)" in r.interpretacao
    assert "Há evidência estatística de que 'y' difere entre os grupos de 'g'." in r.interpretacao
    parecidos = {"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [1.5, 2.5, 3.5, 4.5, 5.5]}
    r = teste.executar(_df(parecidos), _params())
    assert r.decisao == NAO_REJEITA_H0
    assert "Não há evidência suficiente" in r.interpretacao


def test_tabela_de_grupos_e_figura(teste):
    r = teste.executar(_df(GRUPOS), _params())
    _, medios, _, _ = _manual(GRUPOS)
    grupos = r.tabelas["Grupos"]
    assert grupos["Grupo"].tolist() == ["A", "B", "C"]
    assert grupos["n"].tolist() == [5, 5, 5]
    assert grupos["Mediana"].tolist() == ["13,0000", "18,0000", "14,0000"]
    assert grupos["Posto médio"].tolist() == [f"{m:.2f}".replace(".", ",") for m in medios]
    assert "Comparações múltiplas (Dunn, Holm)" not in r.tabelas  # desligado por padrão
    (figura,) = r.figuras
    assert figura.tipo == "boxplot" and [g["rotulo"] for g in figura.dados["grupos"]] == [
        "A",
        "B",
        "C",
    ]
    assert r.comparacao is None and teste.comparacao_inicial() is None


def test_dunn_nao_exibido_sem_rejeicao(teste):
    parecidos = {"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [1.5, 2.5, 3.5, 4.5, 5.5]}
    r = teste.executar(_df(parecidos), _params(dunn=True))
    assert "Comparações múltiplas (Dunn, Holm)" not in r.tabelas
    assert any("Dunn) não exibidas" in a for a in r.avisos)


def test_avisos(teste):
    grupos = {"A": [1.0, 2.0, 2.0], "B": [5.0, 6.0, 7.0, 8.0, 9.0], "C": [3.0, 4.0, 4.0, 5.0, 6.0]}
    df = pd.concat([_df(grupos), pd.DataFrame({"y": [np.nan, 1.0], "g": ["A", None]})])
    r = teste.executar(df, _params())
    textos = " ".join(r.avisos)
    assert "2 linha(s) com valor ou grupo ausente foram descartadas." in textos
    assert "Grupos com menos de 5 observações ('A' (n = 3))" in textos
    assert "H foi corrigido para empates" in textos


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
    iguais = pd.DataFrame({"y": [4.0] * 4, "g": list("aabb")})
    assert any("Todos os valores de 'y' são iguais" in e for e in teste.validar(iguais, _params()))
    muitos = pd.DataFrame({"y": np.arange(42.0), "g": [f"g{i // 2}" for i in range(42)]})
    assert any("(tem 21)" in e for e in teste.validar(muitos, _params()))


def test_formulario(teste):
    specs = teste.parametros()
    assert [(s.nome, s.tipo, s.padrao) for s in specs] == [
        ("coluna", "coluna_numerica", None),
        ("grupo", "coluna_categorica", None),
        ("dunn", "booleano", False),
        ("alfa", "alfa", 0.05),
    ]
    assert teste.validar(_df(GRUPOS), _params()) == []
