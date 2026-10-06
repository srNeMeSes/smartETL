"""Regras de validação reutilizáveis (sem Flet)."""

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from core.tipos import PerfilColuna, detectar_tipos, e_numerica


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


def erros_coluna(
    df: pd.DataFrame, coluna: object, artigo_rotulo: str = "a variável", numerica: bool = True
) -> list[str]:
    """Coluna selecionada, existente e (se `numerica`) numérica e sem infinitos."""
    if not coluna:
        return [f"Selecione {artigo_rotulo}."]
    if coluna not in df.columns:
        return [f"A coluna '{coluna}' não existe no arquivo."]
    if numerica:
        if not e_numerica(df[coluna]):
            return [f"A coluna '{coluna}' não é numérica."]
        if np.isinf(df[coluna].dropna().astype(float)).any():
            return [f"A coluna '{coluna}' contém valores infinitos."]
    return []


def erro_opcao(valor: object, opcoes: Iterable[str], mensagem: str) -> list[str]:
    return [] if valor in set(opcoes) else [mensagem]


def erro_alfa(alfa: object) -> list[str]:
    try:
        if 0 < float(alfa) < 1:
            return []
    except (TypeError, ValueError):
        pass
    return ["O nível de significância (α) deve estar entre 0 e 1."]


def erro_numero(valor: object, mensagem: str) -> list[str]:
    try:
        converter_numero(valor)  # type: ignore[arg-type]
        return []
    except (TypeError, ValueError):
        return [mensagem]
