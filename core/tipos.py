"""Detecção do tipo de cada coluna (numérica, categórica, binária)."""

from dataclasses import dataclass

import pandas as pd
from pandas.api import types as ptypes

# Colunas numéricas discretas (só inteiros) com até este número de níveis também são categóricas.
MAX_NIVEIS_CATEGORICA = 10


@dataclass(frozen=True)
class PerfilColuna:
    """Resumo de uma coluna; pode ter mais de um papel (0/1 é numérica e binária)."""

    nome: str
    numerica: bool
    categorica: bool
    binaria: bool
    n_validos: int
    n_nulos: int
    n_distintos: int


def _numerica(serie: pd.Series) -> bool:
    return ptypes.is_numeric_dtype(serie) and not ptypes.is_bool_dtype(serie)


def _discreta(validos: pd.Series) -> bool:
    """Booleana, inteira, ou float cujos valores são todos inteiros (ex.: inteiros com NaN)."""
    if ptypes.is_bool_dtype(validos) or ptypes.is_integer_dtype(validos):
        return True
    if ptypes.is_float_dtype(validos):
        return bool((validos % 1 == 0).all())
    return False


def perfilar_coluna(nome: str, serie: pd.Series) -> PerfilColuna:
    """Regras documentadas:

    - numérica: dtype numérico (exceto booleano) com ao menos um valor;
    - binária: exatamente 2 valores distintos (ignorando NaN);
    - categórica: não numérica, ou numérica discreta com até `MAX_NIVEIS_CATEGORICA` níveis;
    - coluna sem nenhum valor válido não tem papel algum.
    """
    validos = serie.dropna()
    n_validos = int(validos.size)
    n_distintos = int(validos.nunique())
    if n_validos == 0:
        return PerfilColuna(nome, False, False, False, 0, int(serie.size), 0)
    numerica = _numerica(serie)
    categorica = not numerica or (_discreta(validos) and n_distintos <= MAX_NIVEIS_CATEGORICA)
    return PerfilColuna(
        nome=nome,
        numerica=numerica,
        categorica=categorica,
        binaria=n_distintos == 2,
        n_validos=n_validos,
        n_nulos=int(serie.size - n_validos),
        n_distintos=n_distintos,
    )


def detectar_tipos(df: pd.DataFrame) -> dict[str, PerfilColuna]:
    """Perfil de todas as colunas, na ordem do DataFrame."""
    return {str(nome): perfilar_coluna(str(nome), df[nome]) for nome in df.columns}
