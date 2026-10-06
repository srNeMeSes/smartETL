"""A lista de testes da sidebar deve ser idêntica à lista oficial do CLAUDE.md (§7)."""

from utils import testes_hipotese

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


def test_tem_21_testes():
    assert len(testes_hipotese) == 21


def test_itens_sao_pares_de_texto():
    for item in testes_hipotese:
        assert isinstance(item, tuple) and len(item) == 2
        assert all(isinstance(campo, str) and campo for campo in item)


def test_sem_ids_duplicados():
    ids = [tid for tid, _ in testes_hipotese]
    assert len(set(ids)) == len(ids)


def test_sem_rotulos_duplicados():
    rotulos = [rotulo for _, rotulo in testes_hipotese]
    assert len(set(rotulos)) == len(rotulos)


def test_ids_rotulos_e_ordem_iguais_a_lista_oficial():
    assert list(testes_hipotese) == LISTA_OFICIAL
