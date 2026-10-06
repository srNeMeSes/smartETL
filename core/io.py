"""Leitura dos arquivos de dados (CSV/XLSX). A leitura robusta de CSV vem na Fase 2."""

from pathlib import Path

import pandas as pd

EXTENSOES_SUPORTADAS = (".xlsx", ".csv")


class FormatoNaoSuportado(ValueError):
    """Extensão de arquivo fora de EXTENSOES_SUPORTADAS."""


def importar_dados(path: str | Path, extensao: str | None = None) -> pd.DataFrame:
    """Lê um arquivo .xlsx ou .csv; a extensão é deduzida do caminho se não for informada."""
    extensao = (extensao if extensao is not None else Path(path).suffix).lower()
    if extensao == ".xlsx":
        return pd.read_excel(path)
    if extensao == ".csv":
        return pd.read_csv(path)
    raise FormatoNaoSuportado(
        f"Formato de arquivo não suportado: {extensao}. Utilize apenas .xlsx ou .csv."
    )
