"""Interface da Regressão Logística: formulário, Análise, Visualização e Simulação.

Base: bases/credito_br.csv (400 clientes; 2 com renda ausente). Os valores estatísticos são
conferidos contra o R em tests/core/test_regressao_logit.py; aqui se verifica o que a tela mostra.
"""

from unittest.mock import MagicMock

import flet as ft
import flet.canvas as cv
import pytest
from ajudantes_ui import do_tipo, textos
from conftest import BASES

from app.controller import Controller
from app.state import AppState
from app.ui import tema
from app.ui.tela_principal import TelaPrincipal

PREDITORES = ("renda", "idade", "comprometimento_pct", "vinculo")


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
    controller.carregar_arquivo(str(BASES / "credito_br.csv"))
    tela.sidebar.selecionar("regres_logit")
    return tela, controller


def _caixas(form):
    return {c.label: c for c in form.controle("preditores").controls}


def _executar(tela, *preditores):
    form = tela.formulario
    y = form.controle("y")
    y.value = "inadimplente"
    y.on_select(None)
    for nome in preditores:
        caixa = _caixas(form)[nome]
        caixa.value = True
        caixa.on_change(None)
    tela.sidebar.botao_executar.on_click(None)


def _texto(controle):
    return " ".join(t for t in textos(controle) if t)


def test_formulario_y_binaria_evento_e_limiar(tela_controller):
    tela, _ = tela_controller
    form = tela.formulario
    assert [o.key for o in form.controle("y").options] == ["inadimplente"]  # vinculo tem 3
    assert "Só colunas com exatamente 2 valores" in _texto(form)
    y = form.controle("y")
    y.value = "inadimplente"
    y.on_select(None)
    assert [o.key for o in form.controle("evento").options] == ["Não", "Sim"]
    assert form.controle("evento").value == "Sim"  # sugestão automática
    caixas = _caixas(form)
    assert not caixas["inadimplente"].visible  # y fora dos preditores
    assert form.controle("limiar").value == "0,5"


def test_analise_com_secoes_cor_e_odds_ratio(tela_controller):
    tela, controller = tela_controller
    _executar(tela, *PREDITORES)
    resultado = controller.estado.ultimo_resultado
    assert resultado.teste_id == "regres_logit"
    texto = _texto(tela.painel.analise)
    for titulo in (
        "Pressupostos",
        "Linearidade do logit",
        "Qualidade do ajuste",
        "Tamanho da amostra e separação",
        "Modelo",
        "Classificação",
        "Coeficientes",
    ):
        assert titulo in texto
    assert "Limiar de classificação: 0,5" in texto
    (decisao,) = [t for t in do_tipo(tela.painel.analise, ft.Text) if t.value == "Rejeita H₀"]
    assert decisao.color == tema.DECISAO_REJEITA
    colunas = [
        c.label.value.removesuffix(" ⓘ")
        for tabela in do_tipo(tela.painel.analise, ft.DataTable)
        for c in tabela.columns
    ]
    for coluna in ("LR χ²", "R² Nagelkerke", "AUC", "Odds ratio", "LI OR (95%)"):
        assert coluna in colunas


def test_visualizacao_roc_classes_e_residuos(tela_controller):
    tela, _ = tela_controller
    _executar(tela, *PREDITORES)
    visual = tela.painel.visualizacao
    assert len(do_tipo(visual, cv.Canvas)) == 3
    texto = _texto(visual)
    assert "Curva ROC (AUC = 0,785)" in texto
    assert "Probabilidade prevista por classe observada" in texto
    (lista,) = [d for d in do_tipo(visual, ft.Dropdown) if d.label == "Eixo X"]
    assert [o.key for o in lista.options] == ["Valores ajustados", *PREDITORES]


def test_simulacao_probabilidade_e_classe(tela_controller):
    tela, controller = tela_controller
    _executar(tela, *PREDITORES)
    assert tela.painel.rotulos[-1] == "Simulação"
    sim = tela.simulacao
    assert sim.valor.value == "27,7%"
    texto = _texto(sim)
    assert "Probabilidade prevista de 'inadimplente' = 'Sim'" in texto
    assert "P = P('inadimplente' = 'Sim') = 1 / (1 + e^(−logit))" in texto
    assert "IC 95% da probabilidade" in texto and "Classe prevista (limiar 0,5): 'Não'" in texto
    assert "Sem intervalo de predição" in texto
    assert "".join(s.text for s in sim.equacao.spans).startswith("logit(P) = ")
    sim.listas["vinculo"].value = "Autônomo"
    sim.listas["vinculo"].on_select(None)
    sim.sliders["comprometimento_pct"].value = 65.0
    sim.sliders["comprometimento_pct"].on_change(None)
    simulador = controller.estado.ultimo_resultado.simulacao
    esperado = simulador.prever(
        simulador.valores_iniciais() | {"vinculo": "Autônomo", "comprometimento_pct": 65.0}
    )
    assert esperado.classe == "Sim"
    assert sim.valor.value == simulador.textos_previsao(esperado)[0]
    assert "Classe prevista (limiar 0,5): 'Sim'" in _texto(sim)
