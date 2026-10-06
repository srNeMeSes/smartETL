"""Fixtures compartilhadas pelos testes de caracterização da Fase 0."""

import sys
from collections.abc import Iterator
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

# Os módulos atuais ficam soltos na raiz do projeto (sem pacote).
# Na Fase 1 isto será substituído por `pythonpath` no pyproject.toml.
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


@pytest.fixture
def df_exemplo() -> pd.DataFrame:
    """Dataset pequeno e determinístico, igual ao do print de referência."""
    qtd = [21.0, 8.0, 4.0, 6.0, 4.0, 2.0]
    valor = [462.0, 273.0, 1701.0, 638.0, 1547.0, 1734.0]
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "users": [f"user{i}" for i in range(1, 7)],
            "qtd": qtd,
            "valor": valor,
            "total": [q * v for q, v in zip(qtd, valor, strict=True)],
        }
    )


@pytest.fixture
def csv_valido(tmp_path: Path, df_exemplo: pd.DataFrame) -> Path:
    caminho = tmp_path / "dados.csv"
    df_exemplo.to_csv(caminho, index=False)
    return caminho


@pytest.fixture
def xlsx_valido(tmp_path: Path, df_exemplo: pd.DataFrame) -> Path:
    caminho = tmp_path / "dados.xlsx"
    df_exemplo.to_excel(caminho, index=False)
    return caminho


@pytest.fixture
def csv_vazio(tmp_path: Path) -> Path:
    caminho = tmp_path / "vazio.csv"
    caminho.write_bytes(b"")
    return caminho


@pytest.fixture
def xlsx_vazio(tmp_path: Path) -> Path:
    caminho = tmp_path / "vazio.xlsx"
    openpyxl.Workbook().save(caminho)
    return caminho


def iterar_controles(controle) -> Iterator:
    """Percorre a árvore de controles Flet (via `controls`, `content` e `tabs`)."""
    yield controle
    for atributo in ("content", "controls", "tabs"):
        filho = getattr(controle, atributo, None)
        if filho is None or isinstance(filho, str):
            continue
        filhos = filho if isinstance(filho, list) else [filho]
        for item in filhos:
            if hasattr(item, "_c"):  # só controles Flet
                yield from iterar_controles(item)


def textos(controle) -> list[str]:
    """Todos os valores de `ft.Text` dentro de um controle."""
    import flet as ft

    return [c.value for c in iterar_controles(controle) if isinstance(c, ft.Text)]
