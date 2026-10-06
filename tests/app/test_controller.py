"""Fluxo do controller com uma visão falsa (sem Flet)."""

import asyncio
from typing import ClassVar

import pandas as pd
import pytest
from conftest import BASES

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
    controller.selecionar_teste("durbin_watson")
    assert visao.ultima() == ("sem_arquivo",)
    controller.carregar_arquivo(str(csv_valido))
    assert visao.ultima() == ("indisponivel", "durbin_watson")
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
    controller.selecionar_teste("durbin_watson")
    assert controller.executar() is None
    assert visao.notificacoes[-1] == (
        "O Durbin-Watson ainda não está disponível nesta versão.",
        False,
    )


def test_integracao_t_1am_carregar_selecionar_executar(controller, visao, csv_valido):
    # Checklist §9, item 7: dataset de exemplo → selecionar teste → executar → ResultadoTeste.
    controller.carregar_arquivo(str(csv_valido))
    controller.selecionar_teste("teste_t_1am")
    visao.params = {"coluna": "qtd", "mu0": 5.0, "alternativa": "μ > μ₀", "alfa": 0.05}
    resultado = controller.executar()
    assert isinstance(resultado, ResultadoTeste)
    assert resultado.teste_id == "teste_t_1am"
    assert resultado.estatisticas["n"] == 6
    assert resultado.comparacao is not None and len(resultado.comparacao.linhas) == 3
    assert visao.chamadas[-2:] == [("processando",), ("resultado", "teste_t_1am")]
    assert controller.estado.ultimo_resultado is resultado


def test_integracao_t_1am_validacao_do_teste(controller, visao, csv_valido):
    controller.carregar_arquivo(str(csv_valido))
    visao.params = {"coluna": "users", "mu0": 5.0, "alternativa": "μ ≠ μ₀", "alfa": 0.05}
    assert controller.executar() is None
    assert visao.notificacoes[-1] == ("A coluna 'users' não é numérica.", True)
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


def test_execucao_em_andamento_ignora_novo_clique(controller, visao, csv_valido):
    controller.carregar_arquivo(str(csv_valido))
    visao.params = {"coluna": "qtd", "mu0": 5.0, "alternativa": "μ ≠ μ₀", "alfa": 0.05}
    with controller._executando:  # simula uma execução ainda rodando em outra thread
        assert controller.executar() is None
    assert ("processando",) not in visao.chamadas
    assert controller.executar() is not None  # liberado, executa normalmente


def test_integracao_t_2am_carregar_selecionar_executar(controller, visao, tmp_path):
    # Checklist §9, item 7, para o teste t de duas amostras.
    arquivo = tmp_path / "turmas.csv"
    arquivo.write_text("nota;turma\n7,5;A\n8,0;A\n6,5;A\n9,0;B\n8,5;B\n9,5;B\n", encoding="utf-8")
    controller.carregar_arquivo(str(arquivo))
    controller.selecionar_teste("teste_t_2am")
    assert visao.ultima()[:2] == ("formulario", "teste_t_2am")
    visao.params = {
        "coluna": "nota",
        "grupo": "turma",
        "variancias": "Diferentes (Welch)",
        "alternativa": "μ₁ < μ₂",
        "alfa": 0.05,
    }
    resultado = controller.executar()
    assert isinstance(resultado, ResultadoTeste) and resultado.teste_id == "teste_t_2am"
    assert (resultado.estatisticas["n1"], resultado.estatisticas["n2"]) == (3, 3)
    assert resultado.comparacao.titulo_direita == "Mann-Whitney"
    assert visao.chamadas[-1] == ("resultado", "teste_t_2am")


def test_integracao_t_pareado_carregar_selecionar_executar(controller, visao, tmp_path):
    # Checklist §9, item 7, para o teste t pareado (linha incompleta descartada com aviso).
    arquivo = tmp_path / "pressao.csv"
    arquivo.write_text(
        "paciente;antes;depois\n1;140;132\n2;152;141\n3;138;139\n4;160;149\n5;145;\n6;150;143\n",
        encoding="utf-8",
    )
    controller.carregar_arquivo(str(arquivo))
    controller.selecionar_teste("teste_t_pareado")
    assert visao.ultima()[:2] == ("formulario", "teste_t_pareado")
    visao.params = {
        "coluna1": "antes",
        "coluna2": "depois",
        "alternativa": "μ₁ > μ₂",
        "alfa": 0.05,
    }
    resultado = controller.executar()
    assert isinstance(resultado, ResultadoTeste) and resultado.teste_id == "teste_t_pareado"
    assert resultado.estatisticas["n"] == 5
    assert any("1 linha(s) com valor ausente" in a for a in resultado.avisos)
    assert visao.chamadas[-1] == ("resultado", "teste_t_pareado")


def test_integracao_z_1prop_carregar_selecionar_executar(controller, visao, tmp_path):
    # Checklist §9, item 7, para o teste Z de uma proporção.
    arquivo = tmp_path / "votos.csv"
    linhas = ["voto"] + ["Sim"] * 30 + ["Não"] * 20
    arquivo.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    controller.carregar_arquivo(str(arquivo))
    controller.selecionar_teste("teste_z_1prop")
    assert visao.ultima()[:2] == ("formulario", "teste_z_1prop")
    visao.params = {
        "coluna": "voto",
        "sucesso": "Sim",
        "p0": 0.5,
        "alternativa": "p > p₀",
        "alfa": 0.05,
    }
    resultado = controller.executar()
    assert isinstance(resultado, ResultadoTeste) and resultado.teste_id == "teste_z_1prop"
    assert resultado.estatisticas["p_hat"] == pytest.approx(0.6)
    assert resultado.comparacao.titulo_direita == "Binomial exato"
    assert visao.chamadas[-1] == ("resultado", "teste_z_1prop")


def test_integracao_z_2prop_carregar_selecionar_executar(controller, visao, tmp_path):
    # Checklist §9, item 7, para o teste Z de duas proporções.
    arquivo = tmp_path / "lojas.csv"
    linhas = ["comprou;loja"] + ["Sim;A"] * 30 + ["Não;A"] * 20 + ["Sim;B"] * 18 + ["Não;B"] * 27
    arquivo.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    controller.carregar_arquivo(str(arquivo))
    controller.selecionar_teste("teste_z_2prop")
    assert visao.ultima()[:2] == ("formulario", "teste_z_2prop")
    visao.params = {
        "coluna": "comprou",
        "sucesso": "Sim",
        "grupo": "loja",
        "alternativa": "p₁ ≠ p₂",
        "alfa": 0.05,
    }
    resultado = controller.executar()
    assert isinstance(resultado, ResultadoTeste) and resultado.teste_id == "teste_z_2prop"
    assert (resultado.estatisticas["p1"], resultado.estatisticas["p2"]) == pytest.approx((0.6, 0.4))
    assert resultado.estatisticas["odds_ratio"] == pytest.approx(2.25)
    assert resultado.comparacao.titulo_direita == "Fisher exato"
    assert visao.chamadas[-1] == ("resultado", "teste_z_2prop")


def test_integracao_qui_quadrado_carregar_selecionar_executar(controller, visao, tmp_path):
    # Checklist §9, item 7, para o qui-quadrado (independência e aderência).
    arquivo = tmp_path / "turmas.csv"
    linhas = ["turno;conceito"] + ["Manhã;A"] * 20 + ["Manhã;B"] * 15 + ["Manhã;C"] * 5
    linhas += ["Noite;A"] * 10 + ["Noite;B"] * 15 + ["Noite;C"] * 25
    arquivo.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    controller.carregar_arquivo(str(arquivo))
    controller.selecionar_teste("qui_quadrado")
    assert visao.ultima()[:2] == ("formulario", "qui_quadrado")
    visao.params = {
        "modo": "Independência",
        "coluna1": "turno",
        "coluna2": "conceito",
        "correcao": False,
        "alfa": 0.05,
    }
    resultado = controller.executar()
    assert isinstance(resultado, ResultadoTeste) and resultado.teste_id == "qui_quadrado"
    assert resultado.estatisticas["gl"] == 2 and resultado.comparacao is None
    visao.params = visao.params | {
        "modo": "Aderência",
        "coluna1": "conceito",
        "coluna2": None,
    }
    resultado = controller.executar()
    assert resultado.estatisticas["k"] == 3
    assert visao.chamadas[-1] == ("resultado", "qui_quadrado")


def test_integracao_fisher_carregar_selecionar_executar(controller, visao, tmp_path):
    # Checklist §9, item 7, para o Teste exato de Fisher.
    arquivo = tmp_path / "estudo.csv"
    linhas = ["fuma;doente"] + ["Sim;Sim"] * 12 + ["Sim;Não"] * 5 + ["Não;Sim"] * 4
    linhas += ["Não;Não"] * 11
    arquivo.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    controller.carregar_arquivo(str(arquivo))
    controller.selecionar_teste("fisher")
    assert visao.ultima()[:2] == ("formulario", "fisher")
    visao.params = {
        "coluna1": "fuma",
        "evento1": "Sim",
        "coluna2": "doente",
        "evento2": "Sim",
        "alternativa": "OR > 1",
        "alfa": 0.05,
    }
    resultado = controller.executar()
    assert isinstance(resultado, ResultadoTeste) and resultado.teste_id == "fisher"
    assert resultado.estatisticas["odds_ratio_amostral"] == pytest.approx(6.6)
    assert resultado.comparacao is None
    assert visao.chamadas[-1] == ("resultado", "fisher")


def test_integracao_mcnemar_carregar_selecionar_executar(controller, visao, tmp_path):
    # Checklist §9, item 7, para o McNemar.
    arquivo = tmp_path / "campanha.csv"
    linhas = ["antes;depois"] + ["Sim;Sim"] * 20 + ["Sim;Não"] * 12 + ["Não;Sim"] * 3
    linhas += ["Não;Não"] * 15
    arquivo.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    controller.carregar_arquivo(str(arquivo))
    controller.selecionar_teste("mcnemar")
    assert visao.ultima()[:2] == ("formulario", "mcnemar")
    visao.params = {
        "coluna1": "antes",
        "coluna2": "depois",
        "evento": "Sim",
        "alternativa": "p₁ ≠ p₂",
        "alfa": 0.05,
    }
    resultado = controller.executar()
    assert isinstance(resultado, ResultadoTeste) and resultado.teste_id == "mcnemar"
    assert (resultado.estatisticas["b"], resultado.estatisticas["c"]) == (12, 3)
    assert resultado.comparacao.titulo_esquerda == "Exato (binomial)"
    assert visao.chamadas[-1] == ("resultado", "mcnemar")


def test_integracao_sinal_carregar_selecionar_executar(controller, visao):
    # Checklist §9, item 7, para o Teste do sinal, com a base do projeto (bases/).
    assert controller.carregar_arquivo(str(BASES / "atendimento_br.csv"))
    controller.selecionar_teste("teste_sinal")
    assert visao.ultima()[:2] == ("formulario", "teste_sinal")
    visao.params = {
        "modo": "Uma amostra",
        "coluna1": "minutos",
        "coluna2": None,
        "m0": 10.0,
        "alternativa": "M > M₀",
        "alfa": 0.05,
    }
    resultado = controller.executar()
    assert isinstance(resultado, ResultadoTeste) and resultado.teste_id == "teste_sinal"
    assert (resultado.estatisticas["positivos"], resultado.estatisticas["negativos"]) == (13, 4)
    assert resultado.comparacao is None
    visao.params = visao.params | {
        "modo": "Pareado",
        "coluna1": "antes",
        "coluna2": "depois",
        "m0": 0.0,
        "alternativa": "M ≠ M₀",
    }
    assert controller.carregar_arquivo(str(BASES / "pressao_br.csv"))
    assert controller.executar().estatisticas["positivos"] == 8


def test_integracao_wilcoxon_carregar_selecionar_executar(controller, visao):
    # Checklist §9, item 7, para o Wilcoxon, com as bases do projeto (bases/).
    assert controller.carregar_arquivo(str(BASES / "dieta_br.csv"))
    controller.selecionar_teste("wilcoxon")
    assert visao.ultima()[:2] == ("formulario", "wilcoxon")
    visao.params = {
        "modo": "Pareado",
        "coluna1": "antes",
        "coluna2": "depois",
        "m0": 0.0,
        "alternativa": "M ≠ M₀",
        "alfa": 0.05,
    }
    resultado = controller.executar()
    assert isinstance(resultado, ResultadoTeste) and resultado.teste_id == "wilcoxon"
    assert resultado.estatisticas["usou_exato"] == 1.0
    assert (resultado.estatisticas["w_mais"], resultado.estatisticas["w_menos"]) == (75, 3)
    assert resultado.comparacao is None
    assert visao.chamadas[-1] == ("resultado", "wilcoxon")


def test_integracao_mann_whitney_carregar_selecionar_executar(controller, visao):
    # Checklist §9, item 7, para o Mann-Whitney, com a base do projeto (bases/).
    assert controller.carregar_arquivo(str(BASES / "turmas_br.csv"))
    controller.selecionar_teste("mann_whitney")
    assert visao.ultima()[:2] == ("formulario", "mann_whitney")
    visao.params = {"coluna": "nota", "grupo": "turma", "alternativa": "G₁ ≠ G₂", "alfa": 0.05}
    resultado = controller.executar()
    assert isinstance(resultado, ResultadoTeste) and resultado.teste_id == "mann_whitney"
    assert (resultado.estatisticas["n1"], resultado.estatisticas["n2"]) == (6, 7)
    assert resultado.comparacao is None
    assert visao.chamadas[-1] == ("resultado", "mann_whitney")


def test_integracao_kruskal_wallis_carregar_selecionar_executar(controller, visao):
    # Checklist §9, item 7, para o Kruskal-Wallis (com Dunn), com a base do projeto (bases/).
    assert controller.carregar_arquivo(str(BASES / "entregas_br.csv"))
    controller.selecionar_teste("kruskal_wallis")
    assert visao.ultima()[:2] == ("formulario", "kruskal_wallis")
    visao.params = {"coluna": "prazo_dias", "grupo": "fornecedor", "dunn": True, "alfa": 0.05}
    resultado = controller.executar()
    assert isinstance(resultado, ResultadoTeste) and resultado.teste_id == "kruskal_wallis"
    assert resultado.estatisticas["k"] == 4 and resultado.estatisticas["comparacoes"] == 6
    assert "Comparações múltiplas (Dunn, Holm)" in resultado.tabelas
    assert visao.chamadas[-1] == ("resultado", "kruskal_wallis")


def test_integracao_friedman_carregar_selecionar_executar(controller, visao):
    # Checklist §9, item 7, para o Friedman, com a base do projeto (bases/).
    assert controller.carregar_arquivo(str(BASES / "provas_br.csv"))
    controller.selecionar_teste("friedman")
    assert visao.ultima()[:2] == ("formulario", "friedman")
    visao.params = {"colunas": ["prova1", "prova2", "prova3"], "comparacoes": True, "alfa": 0.05}
    resultado = controller.executar()
    assert isinstance(resultado, ResultadoTeste) and resultado.teste_id == "friedman"
    assert (resultado.estatisticas["n"], resultado.estatisticas["k"]) == (14, 3)
    assert "Comparações múltiplas (Wilcoxon, Holm)" in resultado.tabelas
    assert visao.chamadas[-1] == ("resultado", "friedman")
