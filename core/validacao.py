"""Regras de validação reutilizáveis (sem Flet)."""

import math

import pandas as pd

from core.tipos import PerfilColuna, detectar_tipos


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


def colunas_por_tipo(
    df: pd.DataFrame, tipo: str, perfis: dict[str, PerfilColuna] | None = None
) -> list[str]:
    """Nomes das colunas compatíveis com o tipo de um `ParametroSpec` (regras em core/tipos.py).

    `perfis` evita recalcular a detecção quando ela já foi feita na leitura do arquivo.
    """
    perfis = perfis if perfis is not None else detectar_tipos(df)
    if tipo in ("coluna_numerica", "multi_coluna"):
        papel = "numerica"
    elif tipo == "coluna_binaria":
        papel = "binaria"
    elif tipo == "coluna_categorica":
        papel = "categorica"
    else:
        raise ValueError(f"Tipo de coluna desconhecido: {tipo}")
    return [nome for nome, perfil in perfis.items() if getattr(perfil, papel)]
