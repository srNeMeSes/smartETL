"""Regras de validação reutilizáveis (sem Flet)."""

import math
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from core.tipos import PerfilColuna, detectar_tipos, e_numerica

# "1.000", "2.500", "12.345.678": no padrão brasileiro, milhar; no americano, decimal.
_PONTO_AMBIGUO = re.compile(r"[+-]?[1-9]\d{0,2}(\.\d{3})+")


class NumeroAmbiguo(ValueError):
    """Texto como "1.000", que pode ser mil (milhar) ou um (decimal); a mensagem é exibida."""

    def __init__(self, texto: str):
        inteiro = texto.replace(".", "")
        decimal = texto.replace(".", ",")
        super().__init__(
            f"O valor '{texto}' é ambíguo (milhar ou decimal): escreva {inteiro} para o número "
            f"inteiro ou {decimal} para decimal."
        )


def converter_numero(texto: str | float | int | None) -> float:
    """Converte texto digitado em float; aceita vírgula decimal ("1,5" ou "1.234,5").

    Sem vírgula, o ponto é decimal ("1.5", "0.05"), exceto quando o texto só pode ser lido
    como milhar ou como decimal — "1.000", "2.500" —: aí lança `NumeroAmbiguo` (decisão do
    autor: nunca calcular com um valor que o usuário pode não ter querido dizer).
    """
    if isinstance(texto, (int, float)) and not isinstance(texto, bool):
        valor = float(texto)
    else:
        bruto = (texto or "").strip().replace(" ", "")
        if not bruto:
            raise ValueError("valor vazio")
        if _PONTO_AMBIGUO.fullmatch(bruto):
            raise NumeroAmbiguo(bruto)
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
    if tipo == "preditores":  # numéricas e categóricas (identificadores de texto ficam de fora)
        return [nome for nome, perfil in perfis.items() if perfil.numerica or perfil.categorica]
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
    except NumeroAmbiguo as erro:
        return [str(erro)]
    except (TypeError, ValueError):
        return [mensagem]
