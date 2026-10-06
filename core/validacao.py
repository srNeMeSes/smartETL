"""Regras de validação reutilizáveis (sem Flet)."""

import math

import pandas as pd
from pandas.api import types as ptypes

# Colunas inteiras/booleanas com até este número de valores distintos também são categóricas.
MAX_NIVEIS_CATEGORICA = 10


def converter_numero(texto: str | float | int | None) -> float:
    """Converte texto digitado em float; aceita vírgula decimal ("1,5" ou "1.234,5")."""
    if isinstance(texto, (int, float)) and not isinstance(texto, bool):
        valor = float(texto)
    else:
        bruto = (texto or "").strip().replace(" ", "")
        if not bruto:
            raise ValueError("valor vazio")
        if "," in bruto:
            bruto = bruto.replace(".", "").replace(",", ".")
        valor = float(bruto)
    if not math.isfinite(valor):
        raise ValueError("valor não finito")
    return valor


def _numerica(serie: pd.Series) -> bool:
    return ptypes.is_numeric_dtype(serie) and not ptypes.is_bool_dtype(serie)


def colunas_por_tipo(df: pd.DataFrame, tipo: str) -> list[str]:
    """Nomes das colunas compatíveis com o tipo de um `ParametroSpec`.

    Regras provisórias (a detecção de tipos completa vem na Fase 2):
    - numérica / multi_coluna: dtype numérico (exceto booleano);
    - binária: exatamente 2 valores distintos (ignorando NaN);
    - categórica: não numérica, ou inteira/booleana com até `MAX_NIVEIS_CATEGORICA` níveis.
    """
    colunas: list[str] = []
    for nome in df.columns:
        serie = df[nome]
        if tipo in ("coluna_numerica", "multi_coluna"):
            compativel = _numerica(serie)
        elif tipo == "coluna_binaria":
            compativel = serie.nunique(dropna=True) == 2
        elif tipo == "coluna_categorica":
            discreta = ptypes.is_integer_dtype(serie) or ptypes.is_bool_dtype(serie)
            compativel = not _numerica(serie) or (
                discreta and serie.nunique(dropna=True) <= MAX_NIVEIS_CATEGORICA
            )
        else:
            raise ValueError(f"Tipo de coluna desconhecido: {tipo}")
        if compativel:
            colunas.append(str(nome))
    return colunas
