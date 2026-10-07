"""Interface da Regressão Linear: formulário, Análise por seções, Visualização e Simulação.

Base: bases/salarios_br.csv (120 funcionários; 2 com experiência ausente). Os valores
estatísticos são conferidos contra o R em tests/core; aqui se verifica o que a tela mostra.
"""

from unittest.mock import MagicMock

import flet as ft
import flet.canvas as cv
import pytest
from ajudantes_ui import do_tipo, iterar_controles, textos
from conftest import BASES

from app.controller import Controller
from app.state import AppState
from app.ui.painel_simulacao import PainelSimulacao
from app.ui.tela_principal import TelaPrincipal
from core.testes.regressao import VALORES_AJUSTADOS

FAIXAS = ("1 a < 2", "2 a < 5", "≥ 20", "0 a < 1,0", "1,5 a 2,5", "> 3,0 a 4", "≤ 0,05")


@pytest.fixture
def page():
    pagina = MagicMock()
    pagina.run_thread.side_effect = lambda funcao, *args, **kwargs: funcao(*args, **kwargs)
    return pagina


@pytest.fixture
def tela_controller(page):
    tela = TelaPrincipal(page)
    controller = Controller(AppState(), tela)
    tela.conectar(controller)
    page.add(tela.raiz)
    controller.iniciar()
    controller.carregar_arquivo(str(BASES / "salarios_br.csv"))
    tela.sidebar.selecionar("regres_linear")
    return tela, controller


def _marcar(form, *colunas):
    for caixa in form.controle("preditores").controls:
        if caixa.label in colunas:
            caixa.value = True
            caixa.on_change(None)


def _executar(tela, *preditores):
    form = tela.formulario
    form.controle("y").value = "salario"
    _marcar(form, *preditores)
    tela.sidebar.botao_executar.on_click(None)


def _texto(controle):
    return " ".join(t for t in textos(controle) if t)


def _tabela_com_coluna(controle, coluna):
    for tabela in do_tipo(controle, ft.DataTable):
        if coluna in [c.label.value.removesuffix(" ⓘ") for c in tabela.columns]:
            return tabela
    raise AssertionError(f"tabela com a coluna {coluna!r} não encontrada")


# ---------------------------------------------------------------------------
# Formulário
# ---------------------------------------------------------------------------


def test_formulario_y_numerica_com_explicacao(tela_controller):
    tela, _ = tela_controller
    form = tela.formulario
    assert [o.key for o in form.controle("y").options] == ["experiencia", "salario"]
    assert "Só colunas numéricas aparecem aqui" in _texto(form)
    # identificador (funcionario) fica de fora; numéricas e categóricas aparecem
    assert [c.label for c in form.controle("preditores").controls] == [
        "experiencia",
        "modalidade",
        "cargo",
        "salario",
    ]
    assert [o.key for o in form.controle("ordenar_por").options] == [
        VALORES_AJUSTADOS,
        "experiencia",
        "salario",
    ]
    assert form.controle("ordenar_por").value == VALORES_AJUSTADOS
    assert form.controle("confianca").value == "95%"


def test_niveis_de_referencia_por_categorica(tela_controller):
    tela, _ = tela_controller
    form = tela.formulario
    referencias = form.controle("referencias")
    assert referencias.controls == []
    _marcar(form, "experiencia", "cargo")
    (cargo,) = referencias.controls
    assert cargo.label == "Nível de referência de 'cargo'"
    assert [o.key for o in cargo.options] == ["Analista", "Assistente", "Coordenador", "Gerente"]
    assert cargo.value == "Assistente"  # o mais frequente
    cargo.value = "Gerente"
    _marcar(form, "modalidade")
    assert [d.label for d in referencias.controls] == [
        "Nível de referência de 'modalidade'",
        "Nível de referência de 'cargo'",
    ]
    assert referencias.controls[1].value == "Gerente"  # escolha mantida
    for caixa in form.controle("preditores").controls:
        if caixa.label == "cargo":
            caixa.value = False
            caixa.on_change(None)
    assert [d.label for d in referencias.controls] == ["Nível de referência de 'modalidade'"]
    form.controle("y").value = "salario"
    valores = form.coletar_valores()
    assert valores["preditores"] == ["experiencia", "modalidade"]
    assert valores["referencias"] == {"modalidade": "Home Office"}


def test_referencia_escolhida_muda_a_tabela_de_coeficientes(tela_controller):
    tela, controller = tela_controller
    form = tela.formulario
    _marcar(form, "cargo")
    form.controle("referencias").controls[0].value = "Gerente"
    _executar(tela, "experiencia")
    preditores = controller.estado.ultimo_resultado.tabelas["Coeficientes"]["Preditor"].tolist()
    assert "cargo (ref.: Gerente)" in preditores


# ---------------------------------------------------------------------------
# Análise
# ---------------------------------------------------------------------------


def test_analise_em_secoes_na_ordem_da_especificacao(tela_controller):
    tela, controller = tela_controller
    _executar(tela, "experiencia", "modalidade", "cargo")
    assert controller.estado.ultimo_resultado.teste_id == "regres_linear"
    titulos = [
        t.value
        for t in do_tipo(tela.painel.analise, ft.Text)
        if t.value
        in (
            "Pressupostos",
            "Heterocedasticidade",
            "Autocorrelação",
            "Colinearidade",
            "Normalidade dos resíduos",
            "Modelo",
            "Coeficientes",
        )
    ]
    assert titulos == [
        "Pressupostos",
        "Heterocedasticidade",
        "Autocorrelação",
        "Colinearidade",
        "Normalidade dos resíduos",
        "Modelo",
        "Coeficientes",
    ]
    texto = _texto(tela.painel.analise)
    assert "Regra prática; os limites formais dependem de n e do número de preditores." in texto
    assert "Só é interpretável se as linhas tiverem ordem significativa" in texto
    assert "2 linha(s) com valor ausente" in texto


def test_dw_uma_linha_e_vif_uma_linha_por_preditor_sem_faixas(tela_controller):
    tela, _ = tela_controller
    _executar(tela, "experiencia", "modalidade", "cargo")
    analise = tela.painel.analise
    dw = _tabela_com_coluna(analise, "DW")
    assert len(dw.rows) == 1
    vif = _tabela_com_coluna(analise, "Tolerância")
    assert [r.cells[0].content.value for r in vif.rows] == ["experiencia", "modalidade", "cargo"]
    texto = _texto(analise)
    for faixa in FAIXAS:
        assert faixa not in texto


def test_dica_do_rmse_e_cabecalhos_em_negrito(tela_controller):
    tela, _ = tela_controller
    _executar(tela, "experiencia", "modalidade", "cargo")
    modelo = _tabela_com_coluna(tela.painel.analise, "RMSE")
    rmse = next(c for c in modelo.columns if c.label.value.startswith("RMSE"))
    assert rmse.label.value == "RMSE ⓘ" and "√(SQE / (n − p − 1))" in rmse.tooltip
    coeficientes = _tabela_com_coluna(tela.painel.analise, "Estimativa")
    negrito = [
        r.cells[0].content.value
        for r in coeficientes.rows
        if r.cells[0].content.weight == ft.FontWeight.BOLD
    ]
    assert negrito == ["modalidade (ref.: Home Office)", "cargo (ref.: Assistente)"]
    referencia = next(r for r in coeficientes.rows if r.cells[0].content.value == "    Assistente")
    assert [c.content.value for c in referencia.cells[1:]] == ["—"] * 6


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------


def test_residuos_com_lista_de_eixo_x_e_qq(tela_controller):
    tela, _ = tela_controller
    _executar(tela, "experiencia", "cargo")
    visual = tela.painel.visualizacao
    (lista,) = [d for d in do_tipo(visual, ft.Dropdown) if d.label == "Eixo X"]
    assert [o.key for o in lista.options] == [VALORES_AJUSTADOS, "experiencia", "cargo"]
    assert lista.value == VALORES_AJUSTADOS
    assert len(do_tipo(visual, cv.Canvas)) == 2  # resíduos + Q-Q
    assert "Resíduos × valores ajustados" in _texto(visual)
    assert "Gráfico Q-Q dos resíduos" in _texto(visual)
    lista.value = "cargo"
    lista.on_select(None)
    assert "Resíduos por 'cargo'" in _texto(tela.painel.visualizacao)
    assert "Resíduos × valores ajustados" not in _texto(tela.painel.visualizacao)


# ---------------------------------------------------------------------------
# Simulação
# ---------------------------------------------------------------------------


def test_aba_simulacao_so_na_regressao(tela_controller):
    tela, _ = tela_controller
    assert tela.painel.rotulos == ["Parâmetros", "Análise", "Visualização"]
    _executar(tela, "experiencia", "modalidade", "cargo")
    assert tela.painel.rotulos == ["Parâmetros", "Análise", "Visualização", "Simulação"]
    assert isinstance(tela.painel.simulacao, PainelSimulacao)
    tela.sidebar.selecionar("anova_1fator")
    assert tela.painel.rotulos == ["Parâmetros", "Análise", "Visualização"]
    assert tela.simulacao is None


def test_simulacao_previsao_inicial_e_tempo_real(tela_controller):
    tela, controller = tela_controller
    _executar(tela, "experiencia", "modalidade", "cargo")
    sim = tela.simulacao
    simulador = controller.estado.ultimo_resultado.simulacao
    inicial = simulador.prever(simulador.valores_iniciais())
    from core.testes.regressao import formatar_coeficiente

    assert sim.valor.value == formatar_coeficiente(inicial.valor)
    assert any("IC 95% da média prevista" in t for t in textos(sim) if t)
    assert any("Intervalo de predição 95%" in t for t in textos(sim) if t)
    assert set(sim.sliders) == {"experiencia"} and set(sim.listas) == {"modalidade", "cargo"}
    # slider: atualiza o campo, a previsão e destaca o termo na equação
    sim.sliders["experiencia"].value = 10.0
    sim.sliders["experiencia"].on_change(None)
    assert sim.textos["experiencia"].value == "10,00"
    esperado = simulador.prever(simulador.valores_iniciais() | {"experiencia": 10.0})
    assert sim.valor.value == formatar_coeficiente(esperado.valor)
    destacados = [s.text for s in sim.equacao.spans if s.style.weight == ft.FontWeight.BOLD]
    assert destacados == [" + 35,25·(experiencia)"]
    # lista: troca o nível e destaca os termos da categórica
    sim.listas["cargo"].value = "Gerente"
    sim.listas["cargo"].on_select(None)
    destacados = [s.text for s in sim.equacao.spans if s.style.weight == ft.FontWeight.BOLD]
    assert len(destacados) == 3 and all("·(" in d for d in destacados)
    assert sim.grafico.content is not None and len(do_tipo(sim.grafico, cv.Canvas)) == 1


def test_simulacao_extrapolacao_sinalizada(tela_controller):
    tela, _ = tela_controller
    _executar(tela, "experiencia", "modalidade", "cargo")
    sim = tela.simulacao
    assert sim.avisos.controls == []
    sim.textos["experiencia"].value = "40"
    sim.textos["experiencia"].on_change(None)
    assert sim.sliders["experiencia"].value == sim.sliders["experiencia"].max  # slider na borda
    assert "a previsão é uma extrapolação" in _texto(sim.avisos)
    sim.textos["experiencia"].value = "-"  # digitação incompleta: nada muda
    sim.textos["experiencia"].on_change(None)
    assert "a previsão é uma extrapolação" in _texto(sim.avisos)


def test_equacao_no_topo_da_simulacao(tela_controller):
    tela, _ = tela_controller
    _executar(tela, "experiencia", "modalidade", "cargo")
    sim = tela.simulacao
    equacao = "".join(s.text for s in sim.equacao.spans)
    assert equacao.startswith("salario = 3.264,37 + 35,25·(experiencia) + 638,37·(Presencial)")
    assert any(c is sim.equacao for c in iterar_controles(sim.controls[0]))


def _caixas(form):
    return {c.label: c for c in form.controle("preditores").controls}


def test_variavel_dependente_some_dos_preditores(tela_controller):
    tela, _ = tela_controller
    form = tela.formulario
    y = form.controle("y")
    assert all(c.visible for c in _caixas(form).values())  # nada escolhido ainda
    _marcar(form, "salario", "cargo")
    assert form.controle("referencias").controls[0].label == "Nível de referência de 'cargo'"
    y.value = "salario"
    y.on_select(None)
    caixas = _caixas(form)
    assert not caixas["salario"].visible and caixas["salario"].value is False
    assert caixas["cargo"].value is True  # as outras escolhas ficam
    y.value = "experiencia"
    y.on_select(None)
    caixas = _caixas(form)
    assert caixas["salario"].visible and not caixas["experiencia"].visible
    _marcar(form, "salario")
    assert form.coletar_valores()["preditores"] == ["cargo", "salario"]


@pytest.mark.parametrize(
    ("alvo", "decisao", "cor"),
    [
        ("salario", "Rejeita H₀", "DECISAO_REJEITA"),
        ("experiencia", "Não rejeita H₀", "DECISAO_NAO_REJEITA"),
    ],
)
def test_cor_da_decisao_no_modelo(tela_controller, alvo, decisao, cor):
    from app.ui import tema

    tela, _ = tela_controller
    form = tela.formulario
    form.controle("y").value = alvo
    form.controle("y").on_select(None)
    _marcar(form, "modalidade")  # salário depende da modalidade; a experiência, não
    tela.sidebar.botao_executar.on_click(None)
    (texto,) = [t for t in do_tipo(tela.painel.analise, ft.Text) if t.value == decisao]
    assert texto.color == getattr(tema, cor)
