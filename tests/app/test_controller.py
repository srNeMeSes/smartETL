"""Fluxo do controller com uma visão falsa (sem Flet)."""

import asyncio
from typing import ClassVar

import pandas as pd
import pytest

from app.controller import Controller
from app.state import AppState
from core import registry
from core.base import (
    ErroExecucao,
    ErroValidacao,
    ParametroSpec,
    ResultadoTeste,
    TesteBase,
)


class VisaoFalsa:
    """Registra as chamadas do controller."""

    def __init__(self, caminho=None, params=None):
        self.caminho = caminho
        self.params = params if params is not None else {}
        self.chamadas: list[tuple] = []
        self.notificacoes: list[tuple[str, bool]] = []

    async def escolher_arquivo(self):
        self.chamadas.append(("escolher_arquivo",))
        if isinstance(self.caminho, Exception):
            raise self.caminho
        return self.caminho

    def exibir_dados(self, df):
        self.chamadas.append(("dados", df.shape))

    def exibir_sem_arquivo(self):
        self.chamadas.append(("sem_arquivo",))

    def exibir_indisponivel(self, info):
        self.chamadas.append(("indisponivel", info.id))

    def exibir_formulario(self, teste, dados):
        self.chamadas.append(("formulario", teste.id, tuple(dados.df.columns)))
        self.perfis = dados.perfis

    def coletar_parametros(self):
        if isinstance(self.params, Exception):
            raise self.params
        return self.params

    def exibir_processando(self):
        self.chamadas.append(("processando",))

    def exibir_sem_resultado(self):
        self.chamadas.append(("sem_resultado",))

    def exibir_resultado(self, resultado):
        self.chamadas.append(("resultado", resultado.teste_id))

    def notificar(self, mensagem, erro=False):
        self.notificacoes.append((mensagem, erro))

    def ultima(self):
        return self.chamadas[-1]


@pytest.fixture
def visao():
    return VisaoFalsa()


@pytest.fixture
def controller(visao):
    return Controller(AppState(), visao)


def test_estado_inicial_sem_arquivo(controller, visao):
    controller.iniciar()
    assert controller.estado.teste_id == "teste_t_1am"
    assert visao.chamadas == [("sem_arquivo",)]


def test_carregar_arquivo_atualiza_tabela_e_painel(controller, visao, csv_valido, df_exemplo):
    controller.iniciar()
    assert controller.carregar_arquivo(str(csv_valido))
    assert controller.estado.df.shape == df_exemplo.shape
    assert controller.estado.nome_arquivo == "dados.csv"
    assert visao.chamadas[-2:] == [
        ("dados", df_exemplo.shape),
        ("formulario", "teste_t_1am", tuple(df_exemplo.columns)),
    ]
    assert visao.notificacoes[-1] == ("dados.csv: 6 linhas e 5 colunas carregadas.", False)


def test_arquivo_carregado_depois_da_selecao_atualiza_painel(controller, visao, csv_valido):
    controller.iniciar()
    controller.selecionar_teste("wilcoxon")
    assert visao.ultima() == ("sem_arquivo",)
    controller.carregar_arquivo(str(csv_valido))
    assert visao.ultima() == ("indisponivel", "wilcoxon")
    controller.selecionar_teste("teste_t_1am")
    assert visao.ultima()[:2] == ("formulario", "teste_t_1am")


def test_selecionar_id_desconhecido(controller):
    with pytest.raises(ValueError):
        controller.selecionar_teste("nao_existe")


@pytest.mark.parametrize(
    ("fixture", "trecho"),
    [("csv_vazio", "está vazio"), ("xlsx_vazio", "não contém dados")],
)
def test_arquivo_invalido_vira_mensagem(controller, visao, request, fixture, trecho):
    caminho = request.getfixturevalue(fixture)
    assert not controller.carregar_arquivo(str(caminho))
    mensagem, erro = visao.notificacoes[-1]
    assert erro and trecho in mensagem
    assert controller.estado.df is None


def test_arquivo_corrompido_vira_mensagem(controller, visao, tmp_path):
    arquivo = tmp_path / "quebrado.xlsx"
    arquivo.write_bytes(b"isto nao e um xlsx")
    assert not controller.carregar_arquivo(str(arquivo))
    mensagem, erro = visao.notificacoes[-1]
    assert erro and "Não foi possível ler o arquivo 'quebrado.xlsx'" in mensagem


def test_extensao_invalida_vira_mensagem(controller, visao, tmp_path):
    arquivo = tmp_path / "dados.txt"
    arquivo.write_text("x")
    assert not controller.carregar_arquivo(str(arquivo))
    assert "não suportado" in visao.notificacoes[-1][0]


def test_abrir_arquivo(csv_valido):
    visao = VisaoFalsa(caminho=str(csv_valido))
    controller = Controller(AppState(), visao)
    asyncio.run(controller.abrir_arquivo())
    assert controller.estado.df is not None


@pytest.mark.parametrize("caminho", [None, RuntimeError("falhou")])
def test_abrir_arquivo_cancelado_ou_com_erro(caminho):
    visao = VisaoFalsa(caminho=caminho)
    controller = Controller(AppState(), visao)
    asyncio.run(controller.abrir_arquivo())
    assert controller.estado.df is None
    if caminho is not None:
        assert visao.notificacoes[-1][1] is True


def test_csv_brasileiro_de_ponta_a_ponta(controller, visao, tmp_path):
    arquivo = tmp_path / "vendas.csv"
    arquivo.write_bytes("região;receita;qtd\nSão Paulo;1.234,50;3\nRio;980,00;2\n".encode("cp1252"))
    assert controller.carregar_arquivo(str(arquivo))
    df = controller.estado.df
    assert list(df.columns) == ["região", "receita", "qtd"]
    assert df["receita"].tolist() == [1234.5, 980.0]
    assert visao.perfis is controller.estado.perfis  # perfis calculados uma vez, na leitura
    assert visao.perfis["receita"].numerica
    assert visao.notificacoes[-1] == ("vendas.csv: 2 linhas e 3 colunas carregadas.", False)


def test_avisos_da_leitura_aparecem_na_notificacao(controller, visao, tmp_path):
    arquivo = tmp_path / "misto.csv"
    arquivo.write_text("id;valor\n1;10\n2;20\n3;x\n", encoding="utf-8")
    assert controller.carregar_arquivo(str(arquivo))
    mensagem, erro = visao.notificacoes[-1]
    assert not erro
    linhas = mensagem.split("\n")
    assert linhas[0] == "misto.csv: 3 linhas e 2 colunas carregadas."
    assert linhas[1].startswith("Atenção: A coluna 'valor' mistura números e texto")


def test_arquivo_novo_substitui_o_anterior(controller, csv_valido, tmp_path):
    controller.carregar_arquivo(str(csv_valido))
    outro = tmp_path / "outro.csv"
    outro.write_text("a;b\n1;2\n", encoding="utf-8")
    controller.carregar_arquivo(str(outro))
    assert controller.estado.nome_arquivo == "outro.csv"
    assert controller.estado.df.shape == (1, 2)


def test_falha_na_leitura_mantem_arquivo_anterior(controller, csv_valido, csv_vazio):
    controller.carregar_arquivo(str(csv_valido))
    assert not controller.carregar_arquivo(str(csv_vazio))
    assert controller.estado.nome_arquivo == "dados.csv"


# ---------------- Executar ----------------


def test_executar_sem_arquivo(controller, visao):
    assert controller.executar() is None
    assert visao.notificacoes == [("Carregue um arquivo para começar.", True)]


def test_executar_teste_indisponivel(controller, visao, csv_valido):
    controller.carregar_arquivo(str(csv_valido))
    controller.selecionar_teste("mcnemar")
    assert controller.executar() is None
    assert visao.notificacoes[-1] == ("O McNemar ainda não está disponível nesta versão.", False)


def test_executar_t_1am_ainda_nao_implementado(controller, visao, csv_valido):
    controller.carregar_arquivo(str(csv_valido))
    assert controller.executar() is None
    mensagem, erro = visao.notificacoes[-1]
    assert erro and "ainda não foi implementado" in mensagem
    assert ("processando",) not in visao.chamadas


def test_executar_parametros_invalidos(csv_valido):
    visao = VisaoFalsa(params=ErroValidacao(["Selecione a variável.", "Informe um número."]))
    controller = Controller(AppState(), visao)
    controller.carregar_arquivo(str(csv_valido))
    assert controller.executar() is None
    assert visao.notificacoes[-1] == ("Selecione a variável.\nInforme um número.", True)


# Teste falso para exercitar o fluxo completo carregar → selecionar → executar.
class _TesteFalso(TesteBase):
    id = "falso"
    nome = "Teste falso"
    grupo = "Médias"
    erros: ClassVar[list[str]] = []
    falha: Exception | None = None

    def parametros(self):
        return [ParametroSpec("coluna", "Variável", "coluna_numerica")]

    def validar(self, df, params):
        return self.erros

    def executar(self, df, params):
        if self.falha is not None:
            raise self.falha
        media = float(df[params["coluna"]].mean())
        return ResultadoTeste("falso", {"media": media}, 0.5, 0.05, "Não rejeita H0", "ok")


def _controller_falso(visao, monkeypatch, **atributos):
    for nome, valor in atributos.items():
        monkeypatch.setattr(_TesteFalso, nome, valor)
    info = registry.TesteInfo("falso", "Teste falso", "Médias", _TesteFalso)
    estado = AppState(teste_id="falso")
    return Controller(estado, visao, obter_teste=lambda _id: info)


def test_fluxo_completo_com_resultado(csv_valido, monkeypatch):
    visao = VisaoFalsa(params={"coluna": "qtd"})
    controller = _controller_falso(visao, monkeypatch)
    controller.carregar_arquivo(str(csv_valido))
    resultado = controller.executar()
    assert resultado.estatisticas["media"] == pytest.approx(pd.Series([21, 8, 4, 6, 4, 2]).mean())
    assert visao.chamadas[-2:] == [("processando",), ("resultado", "falso")]
    assert controller.estado.ultimo_resultado is resultado
    assert controller.estado.params == {"coluna": "qtd"}


def test_validar_com_erros_nao_executa(csv_valido, monkeypatch):
    visao = VisaoFalsa(params={"coluna": "qtd"})
    controller = _controller_falso(visao, monkeypatch, erros=["n mínimo é 2."])
    controller.carregar_arquivo(str(csv_valido))
    assert controller.executar() is None
    assert visao.notificacoes[-1] == ("n mínimo é 2.", True)
    assert ("processando",) not in visao.chamadas


@pytest.mark.parametrize(
    ("falha", "trecho"),
    [(ErroExecucao("Matriz singular."), "Matriz singular."), (ZeroDivisionError(), "inesperado")],
)
def test_falha_na_execucao_limpa_processando(csv_valido, monkeypatch, falha, trecho):
    visao = VisaoFalsa(params={"coluna": "qtd"})
    controller = _controller_falso(visao, monkeypatch, falha=falha)
    controller.carregar_arquivo(str(csv_valido))
    assert controller.executar() is None
    assert visao.chamadas[-2:] == [("processando",), ("sem_resultado",)]
    mensagem, erro = visao.notificacoes[-1]
    assert erro and trecho in mensagem
    assert controller.estado.ultimo_resultado is None
