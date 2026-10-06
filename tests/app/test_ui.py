"""Montagem da interface sem janela (página falsa) e integração com o controller."""

from typing import ClassVar
from unittest.mock import MagicMock

import flet as ft
import flet.canvas as cv
import numpy as np
import pandas as pd
import pytest
from ajudantes_ui import do_tipo, iterar_controles, textos

import main as app_main
from app.controller import Controller
from app.state import AppState
from app.ui import tema
from app.ui.componentes.card_comparacao import CardComparacaoTestes
from app.ui.painel_abas import ABA_ANALISE, PainelAbas
from app.ui.painel_parametros import PainelParametros
from app.ui.sidebar import Sidebar
from app.ui.tabela_dados import MAX_LINHAS, TabelaDados, formatar_celula
from app.ui.tela_principal import SEM_ARQUIVO, SEM_EXECUCAO, TelaPrincipal
from core import registry
from core.base import ComparacaoPValores, ErroValidacao, ParametroSpec, ResultadoTeste, TesteBase
from core.figuras import histograma
from core.testes.medias import TesteT1Amostra

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------


@pytest.fixture
def selecoes():
    return []


@pytest.fixture
def sidebar(selecoes):
    return Sidebar(on_arquivo=lambda e: None, on_selecionar=selecoes.append, on_executar=print)


def _visual(sidebar, teste_id):
    linha, icone, texto = sidebar._linhas[teste_id]
    return icone.icon, linha.bgcolor, texto.color


def test_sidebar_lista_todos_os_testes_com_grupos(sidebar):
    conteudo = textos(sidebar)
    nomes = [t.nome for t in registry.listar()]
    assert [t for t in conteudo if t in nomes] == nomes
    grupos = [g.upper() for g in registry.por_grupo()]
    assert [t for t in conteudo if t in grupos] == grupos
    assert sidebar.width == 260


def test_sidebar_primeiro_teste_marcado(sidebar):
    assert sidebar.selecionado == "teste_t_1am"
    assert _visual(sidebar, "teste_t_1am") == (
        ft.Icons.RADIO_BUTTON_CHECKED,
        tema.LARANJA_SUAVE,
        tema.LARANJA,
    )
    for info in registry.listar()[1:]:
        assert _visual(sidebar, info.id) == (
            ft.Icons.RADIO_BUTTON_UNCHECKED,
            tema.TRANSPARENTE,
            tema.TEXTO,
        )


def test_sidebar_selecionar_troca_marcacao_e_avisa(sidebar, selecoes):
    linha, _, _ = sidebar._linhas["wilcoxon"]
    linha.on_click(None)  # mesmo caminho do clique
    assert selecoes == ["wilcoxon"]
    assert sidebar.selecionado == "wilcoxon"
    assert _visual(sidebar, "wilcoxon")[0] == ft.Icons.RADIO_BUTTON_CHECKED
    assert _visual(sidebar, "teste_t_1am")[0] == ft.Icons.RADIO_BUTTON_UNCHECKED


def test_sidebar_botoes_conectados(sidebar):
    # Problema 2 (botão Executar sem on_click) e API nova (ft.Button, não ElevatedButton).
    for botao in (sidebar.botao_arquivo, sidebar.botao_executar):
        assert type(botao) is ft.Button
        assert botao.on_click is not None
    assert "Executar teste" in textos(sidebar.botao_executar)


# --------------------------------------------------------------------------
# Tabela de dados
# --------------------------------------------------------------------------


def _cabecalhos(tabela):
    return [(c.label.value, c.label.color) for c in tabela.tabela.columns]


def test_tabela_estado_vazio():
    tabela = TabelaDados()
    assert tabela.vazia
    assert _cabecalhos(tabela) == [(f"column{i}", tema.TEXTO_TERCIARIO) for i in range(1, 21)]
    assert len(tabela.tabela.rows) == 12
    assert tabela.height == 320


def test_tabela_com_dados_sem_colunas_fantasmas(df_exemplo):
    tabela = TabelaDados()
    df = df_exemplo.rename(columns={"users": "column_usuario"})
    tabela.mostrar(df)
    assert not tabela.vazia
    # Coluna real cujo nome começa com "column" não é mais pintada como fantasma.
    assert _cabecalhos(tabela) == [(str(c), tema.TEXTO_SECUNDARIO) for c in df.columns]
    assert len(tabela.tabela.rows) == len(df)
    assert all(len(r.cells) == df.shape[1] for r in tabela.tabela.rows)
    tabela.mostrar_vazio()
    assert tabela.vazia and len(tabela.tabela.columns) == 20


def test_tabela_limita_linhas_e_formata_nan():
    df = pd.DataFrame({"x": [np.nan, *range(1, MAX_LINHAS + 50)]})
    tabela = TabelaDados()
    tabela.mostrar(df)
    assert len(tabela.tabela.rows) == MAX_LINHAS
    assert tabela.tabela.rows[0].cells[0].content.value == ""


@pytest.mark.parametrize(
    ("valor", "esperado"),
    [
        (None, ""),
        (np.nan, ""),
        (pd.NaT, ""),
        (1.5, "1,5"),
        (np.float64(980.0), "980,0"),
        (np.int64(3), "3"),
        (True, "True"),
        ("a", "a"),
    ],
)
def test_formatar_celula(valor, esperado):
    assert formatar_celula(valor) == esperado


# --------------------------------------------------------------------------
# Painel de abas
# --------------------------------------------------------------------------


def test_painel_abas_api_sem_indices():
    painel = PainelAbas()
    rotulos = [t.label.value for barra in do_tipo(painel, ft.TabBar) for t in barra.tabs]
    assert rotulos == ["Parâmetros", "Análise", "Visualização"]
    a, b, c = ft.Text("a"), ft.Text("b"), ft.Text("c")
    painel.definir_parametros(a)
    painel.definir_analise(b)
    painel.definir_visualizacao(c)
    assert (painel.parametros, painel.analise, painel.visualizacao) == (a, b, c)
    painel.ir_para(ABA_ANALISE)
    assert painel.tabs.selected_index == ABA_ANALISE


# --------------------------------------------------------------------------
# Formulário gerado por ParametroSpec
# --------------------------------------------------------------------------


@pytest.fixture
def form_t1(df_exemplo):
    return PainelParametros(TesteT1Amostra(), df_exemplo)


def test_form_t1_igual_ao_formulario_atual(form_t1):
    variavel = form_t1.controle("coluna")
    mu0 = form_t1.controle("mu0")
    alfa = form_t1.controle("alfa")
    assert isinstance(variavel, ft.Dropdown) and variavel.label == "Variável"
    # Problema 6: só colunas numéricas ("users" é texto).
    assert [o.key for o in variavel.options] == ["id", "qtd", "valor", "total"]
    assert isinstance(mu0, ft.TextField) and mu0.label == "Média Hipotética"
    assert alfa.label == "Nível de significância (α)"
    assert [o.key for o in alfa.options] == ["0.01", "0.05", "0.10"]
    assert alfa.value == "0.05"
    assert "Parâmetros do teste" in textos(form_t1)
    assert isinstance(form_t1.card, CardComparacaoTestes)
    assert "Hₐ:  μ ≠ μ₀" in textos(form_t1.card)


def test_form_coletar_valores(form_t1):
    form_t1.controle("coluna").value = "qtd"
    form_t1.controle("mu0").value = "7,5"
    form_t1.controle("alfa").value = "0.10"
    assert form_t1.coletar_valores() == {
        "coluna": "qtd",
        "mu0": 7.5,
        "alternativa": "μ ≠ μ₀",  # padrão já selecionado
        "alfa": 0.10,
    }


def test_form_campos_vazios(form_t1):
    with pytest.raises(ErroValidacao) as erro:
        form_t1.coletar_valores()
    assert erro.value.mensagens == [
        "Preencha o campo 'Variável'.",
        "Preencha o campo 'Média Hipotética'.",
    ]


def test_form_numero_invalido(form_t1):
    form_t1.controle("coluna").value = "qtd"
    form_t1.controle("mu0").value = "abc"
    with pytest.raises(ErroValidacao, match="Informe um número válido em 'Média Hipotética'"):
        form_t1.coletar_valores()


def test_form_sem_colunas_compativeis():
    df = pd.DataFrame({"nome": ["a", "b"]})
    form = PainelParametros(TesteT1Amostra(), df)
    form.controle("mu0").value = "1"
    with pytest.raises(ErroValidacao, match="não tem colunas compatíveis com 'Variável'"):
        form.coletar_valores()


class _TesteTodosOsTipos(TesteBase):
    id = "todos"
    nome = "Todos os tipos"
    grupo = "Médias"
    specs: ClassVar[list[ParametroSpec]] = [
        ParametroSpec("y", "Resposta", "coluna_numerica"),
        ParametroSpec("g", "Grupo", "coluna_categorica"),
        ParametroSpec("b", "Binária", "coluna_binaria", obrigatorio=False),
        ParametroSpec("xs", "Preditores", "multi_coluna"),
        ParametroSpec("k", "Valor", "numero", padrao=2),
        ParametroSpec("alfa", "α", "alfa", padrao=0.01),
        ParametroSpec(
            "alt", "Alternativa", "opcao", padrao="bilateral", opcoes=["bilateral", "maior"]
        ),
        ParametroSpec("cc", "Correção", "booleano", padrao=True),
    ]

    def parametros(self):
        return self.specs

    def validar(self, df, params):
        return []

    def executar(self, df, params):
        raise NotImplementedError


def test_form_renderiza_todos_os_tipos():
    df = pd.DataFrame({"v": [1.5, 2.5, 3.5], "w": [4.5, 5.5, 6.5], "s": ["a", "b", "a"]})
    form = PainelParametros(_TesteTodosOsTipos(), df)
    assert form.card is None
    assert [o.key for o in form.controle("y").options] == ["v", "w"]
    assert [o.key for o in form.controle("g").options] == ["s"]
    assert [c.label for c in form.controle("xs").controls] == ["v", "w"]
    assert form.controle("alfa").value == "0.01"
    assert form.controle("alt").value == "bilateral"
    assert form.controle("cc").value is True

    form.controle("y").value = "v"
    form.controle("g").value = "s"
    with pytest.raises(ErroValidacao, match="Selecione ao menos uma coluna em 'Preditores'"):
        form.coletar_valores()
    form.controle("xs").controls[1].value = True
    assert form.coletar_valores() == {
        "y": "v",
        "g": "s",
        "b": None,
        "xs": ["w"],
        "k": 2.0,
        "alfa": 0.01,
        "alt": "bilateral",
        "cc": True,
    }


# --------------------------------------------------------------------------
# Tela principal + controller (página falsa)
# --------------------------------------------------------------------------


@pytest.fixture
def page():
    pagina = MagicMock()
    # run_thread executa na hora, para os testes serem determinísticos.
    pagina.run_thread.side_effect = lambda funcao, *args, **kwargs: funcao(*args, **kwargs)
    return pagina


@pytest.fixture
def tela_controller(page):
    tela = TelaPrincipal(page)
    controller = Controller(AppState(), tela)
    tela.conectar(controller)
    page.add(tela.raiz)
    controller.iniciar()
    return tela, controller


def _textos_aba(controle):
    return " ".join(t for t in textos(controle) if t)


def test_main_monta_pagina(page):
    app_main.main(page)
    assert page.title == "smartETL — Processamento de dados"
    assert (page.window.width, page.window.height) == (1440, 900)
    assert (page.window.min_width, page.window.min_height) == (1150, 720)
    page.run_task.assert_called_once_with(page.window.center)  # janela centralizada
    (servico,), _ = page.services.append.call_args
    assert isinstance(servico, ft.FilePicker)
    assert page.add.call_count == 1
    raiz = page.add.call_args.args[0]
    for esperado in ("Processamento de dados", "Arquivo", "Executar teste", "Parâmetros"):
        assert esperado in textos(raiz)


def test_estado_inicial_sem_processando(tela_controller):
    # Problema 1: as abas não ficam mais em "Processando...".
    tela, _ = tela_controller
    for aba in (tela.painel.parametros, tela.painel.analise, tela.painel.visualizacao):
        assert SEM_ARQUIVO in _textos_aba(aba)
        assert "Processando..." not in _textos_aba(aba)
    assert tela.tabela.vazia


def test_carregar_arquivo_mostra_formulario(tela_controller, csv_valido, page):
    tela, controller = tela_controller
    controller.carregar_arquivo(str(csv_valido))
    assert not tela.tabela.vazia
    assert tela.painel.parametros is tela.formulario
    assert SEM_EXECUCAO in _textos_aba(tela.painel.analise)
    assert page.show_dialog.call_count == 1  # aviso de arquivo carregado


def test_cards_de_parametros_e_analise_sao_instancias_distintas(tela_controller, csv_valido):
    # Problema 4: um controle Flet com dois pais.
    tela, controller = tela_controller
    controller.carregar_arquivo(str(csv_valido))
    assert isinstance(tela.formulario.card, CardComparacaoTestes)
    assert isinstance(tela.card_analise, CardComparacaoTestes)
    assert tela.formulario.card is not tela.card_analise
    assert tela.card_analise in list(iterar_controles(tela.painel.analise))


def test_teste_nao_implementado_nao_reusa_formulario_do_t_1am(tela_controller, csv_valido):
    # Problema 3: qualquer teste mostrava o formulário do t de uma amostra.
    tela, controller = tela_controller
    controller.carregar_arquivo(str(csv_valido))
    tela.sidebar.selecionar("wilcoxon")
    assert tela.formulario is None
    assert "Wilcoxon ainda não está disponível" in _textos_aba(tela.painel.parametros)
    assert not do_tipo(tela.painel.parametros, ft.TextField)


def test_executar_pelo_botao_notifica(tela_controller, csv_valido, page):
    tela, controller = tela_controller
    controller.carregar_arquivo(str(csv_valido))
    tela.formulario.controle("coluna").value = "qtd"
    tela.formulario.controle("mu0").value = "5"
    tela.sidebar.botao_executar.on_click(None)
    page.run_thread.assert_called_once_with(controller.executar)  # fora da thread da UI
    # Fluxo completo do t de uma amostra na interface: Análise, card e Visualização.
    resultado = controller.estado.ultimo_resultado
    assert resultado is not None and resultado.teste_id == "teste_t_1am"
    analise = _textos_aba(tela.painel.analise)
    assert resultado.decisao.replace("H0", "H₀") in analise
    assert "Resumo" in analise and "Estatística t" in analise
    assert tela.card_analise._textos_p_esquerda[0].value != "—"
    assert tela.formulario.card._textos_p_direita[0].value != "—"
    assert len(do_tipo(tela.painel.visualizacao, cv.Canvas)) == 1
    assert tela.painel.tabs.selected_index == ABA_ANALISE


def test_executar_com_campos_vazios_notifica_erro(tela_controller, csv_valido, page):
    tela, controller = tela_controller
    controller.carregar_arquivo(str(csv_valido))
    tela.sidebar.botao_executar.on_click(None)
    snack = page.show_dialog.call_args.args[0]
    assert isinstance(snack, ft.SnackBar)
    assert "Preencha o campo 'Variável'." in snack.content.value
    assert snack.bgcolor == tema.NOTIFICACAO_ERRO
    assert controller.estado.ultimo_resultado is None


def test_exibir_resultado_preenche_analise_e_cards(tela_controller, csv_valido):
    tela, controller = tela_controller
    controller.carregar_arquivo(str(csv_valido))
    comparacao = ComparacaoPValores("t Student", "Wilcoxon", ["a", "b", "c"], [(0.01, 0.02)] * 3)
    resultado = ResultadoTeste(
        "teste_t_1am",
        {"t": 2.0},
        0.01,
        0.05,
        "Rejeita H0",
        "Há evidência.",
        tabelas={"Resumo": pd.DataFrame({"Medida": ["t", "gl"], "Valor": ["2,0000", "9"]})},
        avisos=["Amostra pequena."],
        comparacao=comparacao,
    )
    tela.exibir_resultado(resultado)
    analise = _textos_aba(tela.painel.analise)
    for esperado in ("Rejeita H₀", "Há evidência.", "Amostra pequena.", "Resumo", "2,0000"):
        assert esperado in analise
    assert do_tipo(tela.painel.analise, ft.DataTable)
    assert tela.card_analise in list(iterar_controles(tela.painel.analise))
    assert tela.card_analise._textos_p_esquerda[0].value == "0,010"
    assert tela.formulario.card._textos_p_direita[0].value == "0,020"
    assert tela.painel.tabs.selected_index == ABA_ANALISE
    assert "Nenhum gráfico" in _textos_aba(tela.painel.visualizacao)


def test_exibir_resultado_sem_card_com_figura(tela_controller, csv_valido):
    tela, controller = tela_controller
    controller.carregar_arquivo(str(csv_valido))
    figura = histograma(
        [1.0, 2.0, 2.5, 4.0], "Distribuição de 'x'", "x", [("μ₀", 3.0, "tracejado")]
    )
    resultado = ResultadoTeste("x", {}, 0.5, 0.05, "Não rejeita H0", "ok", figuras=[figura])
    tela.exibir_resultado(resultado)
    assert tela.card_analise is None
    assert "Não rejeita H₀" in _textos_aba(tela.painel.analise)
    canvas = do_tipo(tela.painel.visualizacao, cv.Canvas)
    assert len(canvas) == 1
    assert "Distribuição de 'x'" in _textos_aba(tela.painel.visualizacao)


def test_processando_e_sem_resultado(tela_controller, csv_valido):
    tela, controller = tela_controller
    controller.carregar_arquivo(str(csv_valido))
    tela.exibir_processando()
    assert "Processando..." in _textos_aba(tela.painel.analise)
    tela.exibir_sem_resultado()
    assert SEM_EXECUCAO in _textos_aba(tela.painel.analise)


def test_csv_brasileiro_alimenta_formulario(tela_controller, tmp_path):
    tela, controller = tela_controller
    arquivo = tmp_path / "br.csv"
    arquivo.write_bytes("nome;nota;turma\nJoão;7,5;A\nAna;8,0;B\n".encode("latin-1"))
    controller.carregar_arquivo(str(arquivo))
    assert [c.label.value for c in tela.tabela.tabela.columns] == ["nome", "nota", "turma"]
    assert [o.key for o in tela.formulario.controle("coluna").options] == ["nota"]
    assert tela.tabela.tabela.rows[0].cells[0].content.value == "João"


def test_formulario_t_2am_filtra_grupo_binario(tela_controller, tmp_path, page):
    tela, controller = tela_controller
    arquivo = tmp_path / "t.csv"
    arquivo.write_text(
        "nota;turma;cidade\n7,5;A;x\n8,0;A;y\n6,5;A;z\n9,0;B;x\n8,5;B;y\n9,5;B;z\n",
        encoding="utf-8",
    )
    controller.carregar_arquivo(str(arquivo))
    tela.sidebar.selecionar("teste_t_2am")
    form = tela.formulario
    assert [o.key for o in form.controle("coluna").options] == ["nota"]
    assert [o.key for o in form.controle("grupo").options] == ["turma"]  # "cidade" tem 3 níveis
    assert form.controle("variancias").value == "Diferentes (Welch)"
    assert "Mann-Whitney" in textos(form.card)
    form.controle("coluna").value = "nota"
    form.controle("grupo").value = "turma"
    tela.sidebar.botao_executar.on_click(None)
    assert controller.estado.ultimo_resultado.teste_id == "teste_t_2am"
    assert len(do_tipo(tela.painel.visualizacao, cv.Canvas)) == 1


def test_formulario_t_pareado_e_execucao(tela_controller, tmp_path):
    tela, controller = tela_controller
    arquivo = tmp_path / "p.csv"
    arquivo.write_text("id;antes;depois;grupo\n1;10;8;a\n2;12;11;b\n3;9;9,5;a\n4;14;11;b\n")
    controller.carregar_arquivo(str(arquivo))
    tela.sidebar.selecionar("teste_t_pareado")
    form = tela.formulario
    assert form.controle("coluna1").label == "Medida 1 (ex.: antes)"
    assert [o.key for o in form.controle("coluna2").options] == ["id", "antes", "depois"]
    assert "Wilcoxon" in textos(form.card) and "Hₐ:  μ₁ ≠ μ₂" in textos(form.card)
    form.controle("coluna1").value = "antes"
    form.controle("coluna2").value = "depois"
    tela.sidebar.botao_executar.on_click(None)
    assert controller.estado.ultimo_resultado.teste_id == "teste_t_pareado"
    assert "Diferenças 'antes' − 'depois'" in _textos_aba(tela.painel.visualizacao)


def test_campo_sucesso_acompanha_a_coluna(tela_controller, tmp_path):
    tela, controller = tela_controller
    arquivo = tmp_path / "pesquisa.csv"
    arquivo.write_text(
        "comprou;resposta;nota\n1;Sim;3\n0;Não;4\n1;Sim;5\n1;Não;2\n0;Sim;1\n", encoding="utf-8"
    )
    controller.carregar_arquivo(str(arquivo))
    tela.sidebar.selecionar("teste_z_1prop")
    form = tela.formulario
    coluna, sucesso = form.controle("coluna"), form.controle("sucesso")
    assert [o.key for o in coluna.options] == ["comprou", "resposta"]  # binárias
    assert sucesso.options == [] and sucesso.value is None  # sem coluna escolhida
    coluna.value = "resposta"
    coluna.on_select(None)  # mesmo caminho da seleção na tela
    assert [o.key for o in sucesso.options] == ["Não", "Sim"]
    assert sucesso.value == "Sim"
    coluna.value = "comprou"
    coluna.on_select(None)
    assert [o.key for o in sucesso.options] == ["0", "1"] and sucesso.value == "1"
    assert form.controle("p0").value == "0,5"  # padrão com vírgula decimal
    assert "Binomial exato" in textos(form.card)
    tela.sidebar.botao_executar.on_click(None)
    resultado = controller.estado.ultimo_resultado
    assert resultado.teste_id == "teste_z_1prop"
    assert resultado.estatisticas["sucessos"] == 3
    assert len(do_tipo(tela.painel.visualizacao, cv.Canvas)) == 1


def test_campo_sucesso_vazio_gera_erro(tela_controller, tmp_path, page):
    tela, controller = tela_controller
    arquivo = tmp_path / "p.csv"
    arquivo.write_text("resposta\nSim\nNão\nSim\n", encoding="utf-8")
    controller.carregar_arquivo(str(arquivo))
    tela.sidebar.selecionar("teste_z_1prop")
    tela.sidebar.botao_executar.on_click(None)
    mensagem = page.show_dialog.call_args.args[0].content.value
    assert "Preencha o campo 'Variável (binária)'." in mensagem
    assert "Preencha o campo 'Valor que conta como sucesso'." in mensagem


def test_formulario_z_2prop_e_duas_tabelas(tela_controller, tmp_path):
    tela, controller = tela_controller
    arquivo = tmp_path / "lojas.csv"
    linhas = ["comprou;loja;cidade"] + [
        f"{c};{lj};{cid}" for c, lj, cid in zip("SNSSNSNN", "AABBAABB", "xyzxyzxy", strict=True)
    ]
    arquivo.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    controller.carregar_arquivo(str(arquivo))
    tela.sidebar.selecionar("teste_z_2prop")
    form = tela.formulario
    assert [o.key for o in form.controle("grupo").options] == ["comprou", "loja"]  # binárias
    form.controle("coluna").value = "comprou"
    form.controle("coluna").on_select(None)
    assert [o.key for o in form.controle("sucesso").options] == ["N", "S"]
    assert form.controle("sucesso").value == "S"
    form.controle("grupo").value = "loja"
    assert "Fisher exato" in textos(form.card)
    tela.sidebar.botao_executar.on_click(None)
    assert controller.estado.ultimo_resultado.teste_id == "teste_z_2prop"
    analise = _textos_aba(tela.painel.analise)
    assert "Resumo" in analise and "Tabela 2×2" in analise
    assert len(do_tipo(tela.painel.analise, ft.DataTable)) == 2
    assert len(do_tipo(tela.painel.visualizacao, cv.Canvas)) == 1


def test_qui_quadrado_sem_card_com_tres_tabelas(tela_controller, tmp_path):
    tela, controller = tela_controller
    arquivo = tmp_path / "t.csv"
    linhas = ["turno;conceito;nota"] + [
        f"{t};{c};{i},5" for i, (t, c) in enumerate(zip("MMMMNNNNMN", "ABABCCABCA", strict=True))
    ]
    arquivo.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    controller.carregar_arquivo(str(arquivo))
    tela.sidebar.selecionar("qui_quadrado")
    form = tela.formulario
    assert form.card is None and tela.card_analise is None  # teste sem card
    assert form.controle("modo").value == "Independência"
    assert form.controle("correcao").value is False
    assert [o.key for o in form.controle("coluna1").options] == ["turno", "conceito"]
    form.controle("coluna1").value = "turno"
    form.controle("coluna2").value = "conceito"
    tela.sidebar.botao_executar.on_click(None)
    assert controller.estado.ultimo_resultado.teste_id == "qui_quadrado"
    analise = _textos_aba(tela.painel.analise)
    for titulo in ("Resumo", "Frequências observadas", "Frequências esperadas"):
        assert titulo in analise
    assert len(do_tipo(tela.painel.analise, ft.DataTable)) == 3
    assert isinstance(tela.painel.analise, ft.Column)  # sem card: só a coluna de detalhes
    assert len(do_tipo(tela.painel.visualizacao, cv.Canvas)) == 1
