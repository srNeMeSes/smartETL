"""Catálogo dos testes: fonte única de ids, rótulos, grupos e classes."""

from dataclasses import dataclass

from core.base import TesteBase
from core.testes.anova import TesteAnova1Fator, TesteAnova2Fatores
from core.testes.categoricos import QuiQuadrado, TesteFisher, TesteMcNemar
from core.testes.medias import TesteT1Amostra, TesteT2Amostras, TesteTPareado
from core.testes.nao_parametricos import (
    TesteFriedman,
    TesteKruskalWallis,
    TesteMannWhitney,
    TesteSinal,
    TesteWilcoxon,
)
from core.testes.proporcoes import TesteZ1Prop, TesteZ2Prop
from core.testes.regressao import TesteRegressaoLinear


@dataclass(frozen=True)
class TesteInfo:
    id: str
    nome: str
    grupo: str
    classe: type[TesteBase] | None = None  # None = ainda não implementado

    @property
    def disponivel(self) -> bool:
        return self.classe is not None

    def criar(self) -> TesteBase | None:
        return self.classe() if self.classe is not None else None


_CATALOGO: tuple[TesteInfo, ...] = (
    TesteInfo("teste_t_1am", "Teste t (uma amostra)", "Médias", TesteT1Amostra),
    TesteInfo("teste_t_2am", "Teste t (duas amostras)", "Médias", TesteT2Amostras),
    TesteInfo("teste_t_pareado", "Teste t (pareado)", "Médias", TesteTPareado),
    TesteInfo("teste_z_1prop", "Teste Z (uma proporção)", "Proporções", TesteZ1Prop),
    TesteInfo("teste_z_2prop", "Teste Z (duas proporções)", "Proporções", TesteZ2Prop),
    TesteInfo("qui_quadrado", "Qui-quadrado", "Categóricos", QuiQuadrado),
    TesteInfo("fisher", "Teste exato de Fisher", "Categóricos", TesteFisher),
    TesteInfo("mcnemar", "McNemar", "Categóricos", TesteMcNemar),
    TesteInfo("teste_sinal", "Teste do sinal", "Não paramétricos", TesteSinal),
    TesteInfo("wilcoxon", "Wilcoxon", "Não paramétricos", TesteWilcoxon),
    TesteInfo("mann_whitney", "Mann-Whitney U", "Não paramétricos", TesteMannWhitney),
    TesteInfo("kruskal_wallis", "Kruskal-Wallis", "Não paramétricos", TesteKruskalWallis),
    TesteInfo("friedman", "Friedman", "Não paramétricos", TesteFriedman),
    TesteInfo("anova_1fator", "ANOVA (1 fator)", "ANOVA", TesteAnova1Fator),
    TesteInfo("anova_2fator", "ANOVA (2 fatores)", "ANOVA", TesteAnova2Fatores),
    TesteInfo("regres_linear", "Regressão Linear", "Regressão", TesteRegressaoLinear),
    TesteInfo("regres_logit", "Regressão Logística", "Regressão"),
)

_POR_ID = {info.id: info for info in _CATALOGO}

testes_hipotese: list[tuple[str, str]] = [(info.id, info.nome) for info in _CATALOGO]


def listar() -> list[TesteInfo]:
    """Todos os testes, na ordem oficial."""
    return list(_CATALOGO)


def por_grupo() -> dict[str, list[TesteInfo]]:
    """Testes agrupados, preservando a ordem dos grupos e dos testes."""
    grupos: dict[str, list[TesteInfo]] = {}
    for info in _CATALOGO:
        grupos.setdefault(info.grupo, []).append(info)
    return grupos


def obter(teste_id: str) -> TesteInfo:
    try:
        return _POR_ID[teste_id]
    except KeyError:
        raise ValueError(f"O teste '{teste_id}' não está cadastrado.") from None


def primeiro() -> TesteInfo:
    return _CATALOGO[0]
