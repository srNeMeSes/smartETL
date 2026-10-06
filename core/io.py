"""Leitura robusta de CSV e XLSX, com detecção de encoding, separador e decimal.

Regras de detecção do CSV (documentadas e testadas em tests/core/test_io.py):

- Encoding: BOM UTF-8/UTF-16 se houver; senão UTF-8 estrito; senão cp1252; senão latin-1.
- Separador: entre `;`, `,`, tab e `|`, o que dá o mesmo número de campos (≥ 2) no cabeçalho
  e na maior fração das primeiras linhas. Nenhum candidato → arquivo de uma coluna.
- Decimal: com separador `,` é sempre `.`. Senão é `,` quando há mais números como "1,5" ou
  "1.234,5" do que números inequívocos como "1.5" (um "1.234" isolado é ambíguo e conta como
  ponto decimal). Milhar `.` só é usado com decimal `,` e quando não há nenhum "1.5" na amostra.
- Primeira linha é o cabeçalho; espaços nos nomes das colunas são removidos; colunas sem nome
  e totalmente vazias (separador sobrando no fim da linha) são descartadas.
"""

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from core.tipos import PerfilColuna, detectar_tipos

EXTENSOES_SUPORTADAS = (".xlsx", ".csv")
SEPARADORES = (";", ",", "\t", "|")
NOMES_SEPARADORES = {";": "ponto e vírgula", ",": "vírgula", "\t": "tabulação", "|": "barra"}
LINHAS_AMOSTRA = 50
# Separador para arquivos de uma coluna só (caractere de controle que não aparece em texto).
_SEM_SEPARADOR = "\x1f"

_NUMERO_VIRGULA = re.compile(r"^[+-]?(\d+|\d{1,3}(\.\d{3})+),\d+$")
_NUMERO_PONTO = re.compile(r"^[+-]?\d+\.\d+$")
_MILHAR_PONTO = re.compile(r"^[+-]?\d{1,3}(\.\d{3})+(,\d+)?$")
_LINHA_ERRO = re.compile(r"line (\d+)")


class ErroLeitura(Exception):
    """Arquivo que não pôde ser lido; a mensagem é exibida ao usuário."""


class FormatoNaoSuportado(ErroLeitura, ValueError):
    """Extensão de arquivo fora de EXTENSOES_SUPORTADAS."""


@dataclass
class DadosCarregados:
    df: pd.DataFrame
    caminho: str = ""
    perfis: dict[str, PerfilColuna] = field(default_factory=dict)
    encoding: str | None = None  # só CSV
    separador: str | None = None  # só CSV; None = arquivo de uma coluna
    decimal: str | None = None  # só CSV
    avisos: list[str] = field(default_factory=list)

    @property
    def nome_arquivo(self) -> str:
        return Path(self.caminho).name


# ---------------------------------------------------------------------------
# Entrada pública
# ---------------------------------------------------------------------------
def carregar_dados(caminho: str | Path, extensao: str | None = None) -> DadosCarregados:
    """Lê um .csv ou .xlsx; a extensão é deduzida do caminho se não for informada.

    Lança `ErroLeitura` (ou `FormatoNaoSuportado`) com mensagem pronta para o usuário.
    """
    caminho = Path(caminho)
    nome = caminho.name
    extensao = (extensao if extensao is not None else caminho.suffix).lower()
    if extensao not in EXTENSOES_SUPORTADAS:
        raise FormatoNaoSuportado(
            f"Formato de arquivo não suportado: {extensao or '(sem extensão)'}. "
            "Utilize apenas .xlsx ou .csv."
        )
    try:
        conteudo = caminho.read_bytes()
    except FileNotFoundError:
        raise ErroLeitura(f"O arquivo '{nome}' não foi encontrado.") from None
    except OSError:
        raise ErroLeitura(
            f"Não foi possível abrir o arquivo '{nome}'. Verifique se ele não está aberto "
            "em outro programa."
        ) from None
    if not conteudo.strip():
        raise ErroLeitura(f"O arquivo '{nome}' está vazio.")

    dados = _ler_csv(conteudo, nome) if extensao == ".csv" else _ler_xlsx(conteudo, nome)
    dados.caminho = str(caminho)
    _finalizar(dados, nome)
    return dados


def importar_dados(caminho: str | Path, extensao: str | None = None) -> pd.DataFrame:
    """Atalho que devolve só o DataFrame."""
    return carregar_dados(caminho, extensao).df


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def detectar_encoding(conteudo: bytes) -> str:
    if conteudo.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if conteudo.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    for encoding in ("utf-8", "cp1252"):
        try:
            conteudo.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"  # decodifica qualquer sequência de bytes


def _linhas_amostra(texto: str) -> list[str]:
    return [linha for linha in texto.splitlines() if linha.strip()][:LINHAS_AMOSTRA]


def detectar_separador(texto: str) -> str | None:
    """Separador mais consistente na amostra, ou None se o arquivo tiver uma coluna só."""
    linhas = _linhas_amostra(texto)
    melhor: tuple[float, int, int] | None = None
    escolhido = None
    for prioridade, sep in enumerate(SEPARADORES):
        contagens = [len(campos) for campos in csv.reader(linhas, delimiter=sep)]
        if not contagens:
            continue
        moda, frequencia = Counter(contagens).most_common(1)[0]
        if moda < 2 or contagens[0] != moda:
            continue
        nota = (frequencia / len(contagens), moda, -prioridade)
        if melhor is None or nota > melhor:
            melhor, escolhido = nota, sep
    return escolhido


def detectar_decimal(texto: str, separador: str | None) -> tuple[str, str | None]:
    """(decimal, milhar) a partir das células da amostra (sem o cabeçalho)."""
    if separador == ",":
        return ".", None
    linhas = _linhas_amostra(texto)[1:]
    if separador:
        celulas = [c.strip() for campos in csv.reader(linhas, delimiter=separador) for c in campos]
    else:
        celulas = [linha.strip() for linha in linhas]
    virgula = sum(bool(_NUMERO_VIRGULA.match(c)) for c in celulas)
    milhar = sum(bool(_MILHAR_PONTO.match(c)) for c in celulas)
    ponto = sum(bool(_NUMERO_PONTO.match(c)) and not _MILHAR_PONTO.match(c) for c in celulas)
    if virgula > ponto:
        return ",", ("." if milhar and not ponto else None)
    return ".", None


def _ler_csv(conteudo: bytes, nome: str) -> DadosCarregados:
    encoding = detectar_encoding(conteudo)
    texto = conteudo.decode(encoding)
    if "\x00" in texto:
        raise ErroLeitura(f"O arquivo '{nome}' não parece ser um CSV (conteúdo binário).")
    separador = detectar_separador(texto)
    decimal, milhar = detectar_decimal(texto, separador)
    try:
        df = pd.read_csv(
            io.StringIO(texto),
            sep=separador or _SEM_SEPARADOR,
            decimal=decimal,
            thousands=milhar,
            skipinitialspace=True,
        )
    except pd.errors.EmptyDataError:
        raise ErroLeitura(f"O arquivo '{nome}' está vazio.") from None
    except pd.errors.ParserError as erro:
        linha = _LINHA_ERRO.search(str(erro))
        onde = f" (linha {linha.group(1)})" if linha else ""
        raise ErroLeitura(
            f"Não foi possível interpretar o arquivo '{nome}': há linhas com mais colunas "
            f"que o cabeçalho{onde}."
        ) from None
    return DadosCarregados(
        df,
        encoding=encoding,
        separador=separador,
        decimal=decimal,
        avisos=_avisos_colunas_mistas(df, decimal, milhar),
    )


def _avisos_colunas_mistas(df: pd.DataFrame, decimal: str, milhar: str | None) -> list[str]:
    """Colunas de texto em que a maioria dos valores é número: provável erro de digitação."""
    avisos = []
    for nome in df.columns:
        serie = df[nome]
        if pd.api.types.is_numeric_dtype(serie) or pd.api.types.is_bool_dtype(serie):
            continue
        validos = serie.dropna().astype(str).str.strip()
        if validos.empty:
            continue
        normalizados = validos
        if milhar:
            normalizados = normalizados.str.replace(milhar, "", regex=False)
        if decimal != ".":
            normalizados = normalizados.str.replace(decimal, ".", regex=False)
        numeros = pd.to_numeric(normalizados, errors="coerce").notna()
        nao_numericos = int((~numeros).sum())
        if numeros.mean() >= 0.5 and nao_numericos:
            exemplo = validos[~numeros].iloc[0]
            avisos.append(
                f"A coluna '{nome}' mistura números e texto ({nao_numericos} valor(es) não "
                f"numérico(s), ex.: '{exemplo}'); foi tratada como texto."
            )
    return avisos


# ---------------------------------------------------------------------------
# XLSX
# ---------------------------------------------------------------------------
def _ler_xlsx(conteudo: bytes, nome: str) -> DadosCarregados:
    try:
        with pd.ExcelFile(io.BytesIO(conteudo), engine="openpyxl") as planilha:
            abas = planilha.sheet_names
            df = planilha.parse(abas[0]) if abas else pd.DataFrame()
    except Exception:
        raise ErroLeitura(
            f"Não foi possível ler o arquivo '{nome}': não é uma planilha Excel (.xlsx) válida."
        ) from None
    avisos = []
    if len(abas) > 1:
        avisos.append(
            f"O arquivo tem {len(abas)} planilhas; foi lida apenas a primeira ('{abas[0]}')."
        )
    return DadosCarregados(df, avisos=avisos)


# ---------------------------------------------------------------------------
# Comum
# ---------------------------------------------------------------------------
def _finalizar(dados: DadosCarregados, nome: str) -> None:
    df = dados.df
    df.columns = [str(c).strip() for c in df.columns]
    sobrando = [c for c in df.columns if c.startswith("Unnamed:") and df[c].isna().all()]
    if sobrando:
        df = df.drop(columns=sobrando)
    if len(df.columns) == 0:
        raise ErroLeitura(f"O arquivo '{nome}' não contém dados.")
    if len(df) == 0:
        raise ErroLeitura(f"O arquivo '{nome}' não contém dados (só o cabeçalho).")
    dados.df = df
    dados.perfis = detectar_tipos(df)
