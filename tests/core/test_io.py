"""Leitura de CSV/XLSX em core/io.py (Fase 2)."""

import math

import openpyxl
import pandas as pd
import pytest

from core.io import (
    DadosCarregados,
    ErroLeitura,
    FormatoNaoSuportado,
    carregar_dados,
    detectar_decimal,
    detectar_encoding,
    detectar_separador,
    importar_dados,
)


@pytest.fixture
def escrever(tmp_path):
    """Grava `conteudo` (texto ou bytes) em um arquivo temporário e devolve o caminho."""

    def _escrever(nome: str, conteudo: str | bytes, encoding: str = "utf-8"):
        caminho = tmp_path / nome
        dados = conteudo if isinstance(conteudo, bytes) else conteudo.encode(encoding)
        caminho.write_bytes(dados)
        return caminho

    return _escrever


# ---------------------------------------------------------------------------
# Arquivos válidos
# ---------------------------------------------------------------------------


def test_csv_padrao(csv_valido, df_exemplo):
    dados = carregar_dados(csv_valido)
    assert isinstance(dados, DadosCarregados)
    pd.testing.assert_frame_equal(dados.df, df_exemplo)
    assert (dados.encoding, dados.separador, dados.decimal) == ("utf-8", ",", ".")
    assert dados.nome_arquivo == "dados.csv"
    assert dados.avisos == []


def test_xlsx_valido(xlsx_valido, df_exemplo):
    dados = carregar_dados(xlsx_valido)
    pd.testing.assert_frame_equal(dados.df, df_exemplo, check_dtype=False)
    assert dados.encoding is None and dados.separador is None


def test_csv_brasileiro_latin1(escrever):
    # Antes da Fase 2 era xfail (problema 9).
    caminho = escrever("br.csv", "nome;valor\nJoão;1,5\nAna;2,0\n", encoding="latin-1")
    dados = carregar_dados(caminho)
    assert list(dados.df.columns) == ["nome", "valor"]
    assert dados.df["valor"].tolist() == [1.5, 2.0]
    assert dados.df["nome"].tolist() == ["João", "Ana"]
    assert (dados.encoding, dados.separador, dados.decimal) == ("cp1252", ";", ",")


def test_csv_brasileiro_com_milhar(escrever):
    caminho = escrever("br.csv", "item;preco;qtd\na;1.234,50;3\nb;12,00;1.000\nc;0,5;2\n")
    df = importar_dados(caminho)
    assert df["preco"].tolist() == [1234.5, 12.0, 0.5]
    assert df["qtd"].tolist() == [3, 1000, 2]


def test_ponto_e_virgula_com_ponto_decimal(escrever):
    caminho = escrever("x.csv", "a;b\n1.5;2\n3.25;4\n")
    dados = carregar_dados(caminho)
    assert dados.decimal == "."
    assert dados.df["a"].tolist() == [1.5, 3.25]


@pytest.mark.parametrize(
    ("encoding", "esperado"),
    [("utf-8-sig", "utf-8-sig"), ("utf-16", "utf-16"), ("utf-8", "utf-8"), ("cp1252", "cp1252")],
)
def test_encodings(escrever, encoding, esperado):
    caminho = escrever("e.csv", "cidade,população\nSão Paulo,12\nBrasília,3\n", encoding=encoding)
    dados = carregar_dados(caminho)
    assert dados.encoding == esperado
    assert list(dados.df.columns) == ["cidade", "população"]
    assert dados.df["cidade"].tolist() == ["São Paulo", "Brasília"]


def test_latin1_fallback():
    # 0x81 não existe em cp1252: só latin-1 decodifica.
    assert detectar_encoding(b"a;b\n\x81;1\n") == "latin-1"


@pytest.mark.parametrize(("sep", "nome"), [("\t", "tabulação"), ("|", "barra")])
def test_outros_separadores(escrever, sep, nome):
    caminho = escrever("s.csv", f"a{sep}b\n1{sep}2\n3{sep}4\n")
    dados = carregar_dados(caminho)
    assert dados.separador == sep
    assert dados.df.shape == (2, 2)


def test_campos_com_aspas_e_separador_dentro(escrever):
    caminho = escrever("q.csv", 'nome;obs\n"Silva; J.";"a, b"\nAna;c\n')
    df = importar_dados(caminho)
    assert df["nome"].tolist() == ["Silva; J.", "Ana"]
    assert df.shape == (2, 2)


def test_uma_coluna_com_virgula_decimal(escrever):
    caminho = escrever("u.csv", "valor\n1,5\n2,5\n3,0\n")
    dados = carregar_dados(caminho)
    assert list(dados.df.columns) == ["valor"]
    assert dados.separador is None
    assert dados.df["valor"].tolist() == [1.5, 2.5, 3.0]


def test_cabecalho_com_espacos_e_separador_sobrando(escrever):
    caminho = escrever("t.csv", " a ; b ;\n1;2;\n3;4;\n")
    df = importar_dados(caminho)
    assert list(df.columns) == ["a", "b"]


def test_nan_em_coluna_numerica(escrever):
    caminho = escrever("n.csv", "x;y\n1,5;a\n;b\n2,5;\n")
    df = importar_dados(caminho)
    assert df["x"].dtype == "float64"
    assert math.isnan(df["x"][1])
    assert df["y"].isna().sum() == 1


def test_coluna_mista_gera_aviso(escrever):
    caminho = escrever("m.csv", "id;valor\n1;10\n2;20\n3;erro\n4;30\n")
    dados = carregar_dados(caminho)
    assert not pd.api.types.is_numeric_dtype(dados.df["valor"])
    assert len(dados.avisos) == 1
    assert "mistura números e texto" in dados.avisos[0]
    assert "'erro'" in dados.avisos[0]


def test_texto_comum_nao_gera_aviso(escrever):
    caminho = escrever("t.csv", "nome,grupo\nAna,A\nBia,B\n")
    assert carregar_dados(caminho).avisos == []


def test_perfis_calculados(csv_valido):
    perfis = carregar_dados(csv_valido).perfis
    assert list(perfis) == ["id", "users", "qtd", "valor", "total"]
    assert perfis["qtd"].numerica and not perfis["users"].numerica


# ---------------------------------------------------------------------------
# Detecção (unidades)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("a;b\n1;2\n", ";"),
        ("a,b\n1.5,2\n", ","),
        ("a;b\n1,5;2,5\n", ";"),
        ("nome,valor\n", ","),
        ("valor\n1,5\n2,5\n", None),
        ("x\n1\n2\n", None),
    ],
)
def test_detectar_separador(texto, esperado):
    assert detectar_separador(texto) == esperado


@pytest.mark.parametrize(
    ("texto", "sep", "esperado"),
    [
        ("a;b\n1,5;2\n", ";", (",", None)),
        ("a;b\n1.234,5;2\n", ";", (",", ".")),
        ("a;b\n1,5;2.5\n", ";", (".", None)),
        ("a;b\n1.5;2\n", ";", (".", None)),
        ("a;b\n1.234;2\n", ";", (".", None)),  # ambíguo: tratado como ponto decimal
        ("a,b\n1,5\n", ",", (".", None)),
        ("v\n1,5\n", None, (",", None)),
    ],
)
def test_detectar_decimal(texto, sep, esperado):
    assert detectar_decimal(texto, sep) == esperado


# ---------------------------------------------------------------------------
# Erros (mensagens amigáveis, nunca exceção do pandas)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extensao", [".txt", ".xls", "csv", ""])
def test_extensao_invalida(csv_valido, extensao):
    with pytest.raises(FormatoNaoSuportado, match="não suportado") as erro:
        carregar_dados(csv_valido, extensao)
    assert isinstance(erro.value, ErroLeitura)


def test_extensao_em_maiusculas(csv_valido, df_exemplo):
    assert importar_dados(csv_valido, ".CSV").shape == df_exemplo.shape


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(ErroLeitura, match=r"'nada\.csv' não foi encontrado"):
        carregar_dados(tmp_path / "nada.csv")


@pytest.mark.parametrize("conteudo", [b"", b"   \n\n  "])
def test_csv_vazio(escrever, conteudo):
    with pytest.raises(ErroLeitura, match="está vazio"):
        carregar_dados(escrever("v.csv", conteudo))


def test_csv_so_cabecalho(escrever):
    with pytest.raises(ErroLeitura, match=r"não contém dados \(só o cabeçalho\)"):
        carregar_dados(escrever("h.csv", "a;b\n"))


def test_csv_binario(escrever):
    with pytest.raises(ErroLeitura, match="não parece ser um CSV"):
        carregar_dados(escrever("img.csv", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"))


def test_csv_linha_com_colunas_a_mais(escrever):
    with pytest.raises(ErroLeitura, match=r"mais colunas que o cabeçalho \(linha 4\)"):
        carregar_dados(escrever("r.csv", "a;b\n1;2\n3;4\n5;6;7\n"))


def test_xlsx_vazio(xlsx_vazio):
    with pytest.raises(ErroLeitura, match="não contém dados"):
        carregar_dados(xlsx_vazio)


def test_xlsx_corrompido(escrever):
    with pytest.raises(ErroLeitura, match="não é uma planilha Excel"):
        carregar_dados(escrever("q.xlsx", b"isto nao e um xlsx"))


def test_xlsx_varias_planilhas(tmp_path):
    caminho = tmp_path / "abas.xlsx"
    livro = openpyxl.Workbook()
    livro.active.title = "Dados"
    livro.active.append(["x", "y"])
    livro.active.append([1, None])
    livro.active.append([2, 3.5])
    livro.create_sheet("Outra").append(["z"])
    livro.save(caminho)
    dados = carregar_dados(caminho)
    assert list(dados.df.columns) == ["x", "y"]
    assert dados.df["y"].isna().sum() == 1
    assert dados.avisos == ["O arquivo tem 2 planilhas; foi lida apenas a primeira ('Dados')."]
