"""Caracterização de `load_table.importar_dados` (comportamento atual)."""

import pandas as pd
import pytest

from load_table import importar_dados


def test_csv_valido(csv_valido, df_exemplo):
    df = importar_dados(csv_valido, ".csv")
    pd.testing.assert_frame_equal(df, df_exemplo)


def test_xlsx_valido(xlsx_valido, df_exemplo):
    df = importar_dados(xlsx_valido, ".xlsx")
    pd.testing.assert_frame_equal(df, df_exemplo, check_dtype=False)
    # Caracterização: no XLSX, floats com valor inteiro (21.0) voltam como int64.
    assert df["qtd"].dtype == "int64"


def test_extensao_em_maiusculas_e_aceita(csv_valido, df_exemplo):
    df = importar_dados(csv_valido, ".CSV")
    assert df.shape == df_exemplo.shape


@pytest.mark.parametrize("extensao", [".txt", ".xls", "csv", ""])
def test_extensao_invalida(csv_valido, extensao):
    # A extensão precisa vir com o ponto ("csv" sozinho também é recusado).
    with pytest.raises(ValueError, match="não suportado"):
        importar_dados(csv_valido, extensao)


def test_csv_vazio_lanca_erro_do_pandas(csv_vazio):
    # Hoje o erro não é tratado: chega ao callback como EmptyDataError.
    with pytest.raises(pd.errors.EmptyDataError):
        importar_dados(csv_vazio, ".csv")


def test_xlsx_vazio_retorna_dataframe_vazio(xlsx_vazio):
    df = importar_dados(xlsx_vazio, ".xlsx")
    assert df.empty
    assert df.shape == (0, 0)


@pytest.mark.xfail(
    strict=True,
    reason="Problema 9: read_csv sem sep/decimal/encoding; CSV brasileiro (;, vírgula, latin-1) "
    "não é lido. Corrigir na Fase 2 (core/io.py).",
)
def test_csv_brasileiro(tmp_path):
    caminho = tmp_path / "br.csv"
    caminho.write_bytes("nome;valor\nJoão;1,5\nAna;2,0\n".encode("latin-1"))
    df = importar_dados(caminho, ".csv")
    assert list(df.columns) == ["nome", "valor"]
    assert df["valor"].tolist() == [1.5, 2.0]
