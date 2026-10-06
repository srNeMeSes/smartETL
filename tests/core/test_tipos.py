"""Detecção de tipos de coluna (core/tipos.py)."""

import numpy as np
import pandas as pd
import pytest

from core.tipos import MAX_NIVEIS_CATEGORICA, detectar_tipos, perfilar_coluna


def _papeis(serie: pd.Series) -> tuple[bool, bool, bool]:
    p = perfilar_coluna("c", serie)
    return p.numerica, p.categorica, p.binaria


@pytest.mark.parametrize(
    ("valores", "esperado"),
    [
        ([1.5, 2.5, 3.5, 4.5], (True, False, False)),  # contínua
        ([1, 2, 3, 4], (True, True, False)),  # inteira com poucos níveis
        ([0, 1, 1, 0], (True, True, True)),  # 0/1
        ([1.0, 2.0, np.nan, 3.0], (True, True, False)),  # inteiros com NaN viram float
        (["a", "b", "a", "c"], (False, True, False)),
        (["sim", "não", "sim", None], (False, True, True)),
        ([True, False, True], (False, True, True)),
        ([1.5, 2.5], (True, False, True)),  # 2 valores contínuos: numérica e binária
    ],
)
def test_papeis(valores, esperado):
    assert _papeis(pd.Series(valores)) == esperado


def test_inteira_com_muitos_niveis_nao_e_categorica():
    serie = pd.Series(range(MAX_NIVEIS_CATEGORICA + 1))
    assert _papeis(serie) == (True, False, False)
    assert _papeis(pd.Series(range(MAX_NIVEIS_CATEGORICA))) == (True, True, False)


def test_coluna_vazia_sem_papel():
    perfil = perfilar_coluna("v", pd.Series([np.nan, np.nan]))
    assert (perfil.numerica, perfil.categorica, perfil.binaria) == (False, False, False)
    assert (perfil.n_validos, perfil.n_nulos, perfil.n_distintos) == (0, 2, 0)


def test_contagens():
    perfil = perfilar_coluna("x", pd.Series([1, 1, 2, None]))
    assert (perfil.n_validos, perfil.n_nulos, perfil.n_distintos) == (3, 1, 2)


def test_detectar_tipos_preserva_ordem(df_exemplo):
    perfis = detectar_tipos(df_exemplo)
    assert list(perfis) == list(df_exemplo.columns)
    assert [n for n, p in perfis.items() if p.numerica] == ["id", "qtd", "valor", "total"]
    assert perfis["users"].categorica and not perfis["users"].numerica
