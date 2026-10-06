"""Registro dos testes: 21 ids, rótulos e ordem oficiais (CLAUDE.md §7)."""

import pytest

from core import registry
from core.base import TesteBase

LISTA_OFICIAL = [
    # Médias
    ("teste_t_1am", "Teste t (uma amostra)"),
    ("teste_t_2am", "Teste t (duas amostras)"),
    ("teste_t_pareado", "Teste t (pareado)"),
    # Proporções
    ("teste_z_1prop", "Teste Z (uma proporção)"),
    ("teste_z_2prop", "Teste Z (duas proporções)"),
    # Categóricos
    ("qui_quadrado", "Qui-quadrado"),
    ("fisher", "Teste exato de Fisher"),
    ("mcnemar", "McNemar"),
    # Não paramétricos
    ("teste_sinal", "Teste do sinal"),
    ("wilcoxon", "Wilcoxon"),
    ("mann_whitney", "Mann-Whitney U"),
    ("kruskal_wallis", "Kruskal-Wallis"),
    ("friedman", "Friedman"),
    # ANOVA
    ("anova_1fator", "ANOVA (1 fator)"),
    ("anova_2fator", "ANOVA (2 fatores)"),
    # Regressão / diagnóstico
    ("regres_linear", "Regressão Linear"),
    ("regres_logit", "Regressão Logística"),
    ("durbin_watson", "Durbin-Watson"),
    ("breusch_pagan", "Breusch-Pagan"),
    ("white", "White"),
    ("vif", "VIF"),
]

GRUPOS = [
    "Médias",
    "Proporções",
    "Categóricos",
    "Não paramétricos",
    "ANOVA",
    "Regressão",
    "Diagnóstico",
]


def test_21_testes_registrados():
    assert len(registry.listar()) == 21


def test_sem_ids_nem_rotulos_duplicados():
    ids = [t.id for t in registry.listar()]
    nomes = [t.nome for t in registry.listar()]
    assert len(set(ids)) == len(ids)
    assert len(set(nomes)) == len(nomes)


def test_ids_rotulos_e_ordem_iguais_a_lista_oficial():
    assert [(t.id, t.nome) for t in registry.listar()] == LISTA_OFICIAL
    assert registry.testes_hipotese == LISTA_OFICIAL


def test_grupos_na_ordem_do_catalogo():
    grupos = registry.por_grupo()
    assert list(grupos) == GRUPOS
    assert [t.id for g in grupos.values() for t in g] == [tid for tid, _ in LISTA_OFICIAL]


def test_obter_e_primeiro():
    assert registry.obter("wilcoxon").nome == "Wilcoxon"
    assert registry.primeiro().id == "teste_t_1am"


def test_obter_id_desconhecido():
    with pytest.raises(ValueError, match="não está cadastrado"):
        registry.obter("nao_existe")


def test_classes_registradas_sao_consistentes():
    for info in registry.listar():
        if info.classe is None:
            assert info.criar() is None
            assert not info.disponivel
            continue
        teste = info.criar()
        assert isinstance(teste, TesteBase)
        assert (teste.id, teste.nome, teste.grupo) == (info.id, info.nome, info.grupo)


def test_testes_implementados():
    # Fase 3: atualizar a cada teste implementado (ordem da seção 7).
    assert [t.id for t in registry.listar() if t.disponivel] == [
        "teste_t_1am",
        "teste_t_2am",
        "teste_t_pareado",
        "teste_z_1prop",
        "teste_z_2prop",
        "qui_quadrado",
        "fisher",
        "mcnemar",
    ]
