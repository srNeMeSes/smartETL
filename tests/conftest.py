"""Fixtures compartilhadas (datasets pequenos e determinísticos)."""

from pathlib import Path

import openpyxl
import pandas as pd
import pytest

# Pasta das bases de teste manual do projeto (também usadas em testes de integração).
RAIZ = Path(__file__).resolve().parent.parent
BASES = RAIZ / "bases"


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


@pytest.fixture
def teste_indisponivel(monkeypatch):
    """Torna a Regressão Logística "ainda não disponível" durante o teste (todos os testes do
    catálogo já estão implementados; o estado indisponível continua precisando de cobertura)."""
    from core import registry

    info = registry.obter("regres_logit")
    monkeypatch.setitem(
        registry._POR_ID, info.id, registry.TesteInfo(info.id, info.nome, info.grupo)
    )
    return info.id
