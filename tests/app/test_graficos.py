"""Desenho das figuras com flet.canvas."""

import flet as ft
import flet.canvas as cv
import pytest
from ajudantes_ui import do_tipo, textos

from app.ui import tema
from app.ui.graficos import ALTURA, LARGURA, cor_serie, desenhar_figura
from core.base import Figura
from core.figuras import barras, barras_agrupadas, boxplot, histograma


@pytest.fixture
def figura():
    return histograma(
        [1.0, 1.5, 2.0, 2.2, 3.0, 3.1, 3.9],
        "Distribuição de 'x'",
        "x",
        [("x̄ = 2,39", 2.39, "destaque"), ("μ₀ = 5", 5.0, "tracejado")],
    )


def _canvas(controle) -> cv.Canvas:
    (canvas,) = do_tipo(controle, cv.Canvas)
    return canvas


def test_histograma_desenha_barras_eixos_e_referencias(figura):
    controle = desenhar_figura(figura)
    canvas = _canvas(controle)
    assert (canvas.width, canvas.height) == (LARGURA, ALTURA)
    barras = [s for s in canvas.shapes if isinstance(s, cv.Rect)]
    nao_vazias = sum(1 for c in figura.dados["contagens"] if c)
    assert len(barras) == 2 * nao_vazias  # preenchimento + contorno
    linhas = [s for s in canvas.shapes if isinstance(s, cv.Line)]
    assert len(linhas) == 2 + 2  # 2 eixos + 2 referências
    tracejadas = [s for s in linhas if s.paint.stroke_dash_pattern]
    assert len(tracejadas) == 1
    assert tracejadas[0].paint.color == tema.GRAFICO_REFERENCIA


def test_formas_dentro_do_canvas(figura):
    canvas = _canvas(desenhar_figura(figura))
    for forma in canvas.shapes:
        if isinstance(forma, cv.Rect):
            assert 0 <= forma.x and forma.x + forma.width <= LARGURA
            assert 0 <= forma.y and forma.y + forma.height <= ALTURA
        if isinstance(forma, cv.Line):
            assert 0 <= min(forma.x1, forma.x2) and max(forma.x1, forma.x2) <= LARGURA


def test_referencia_na_borda_direita(figura):
    canvas = _canvas(desenhar_figura(figura))
    mu0 = next(s for s in canvas.shapes if isinstance(s, cv.Line) and s.paint.stroke_dash_pattern)
    barras = [s for s in canvas.shapes if isinstance(s, cv.Rect)]
    assert mu0.x1 > max(b.x + b.width for b in barras)  # μ₀ = 5 fica à direita dos dados


def test_titulo_e_legenda(figura):
    controle = desenhar_figura(figura)
    conteudo = textos(controle)
    assert conteudo[0] == "Distribuição de 'x'"
    assert {"x̄ = 2,39", "μ₀ = 5"} <= set(conteudo)
    rotulos_eixo = [s.value for s in _canvas(controle).shapes if isinstance(s, cv.Text)]
    assert "0" in rotulos_eixo and "5,00" in rotulos_eixo


def test_tipo_desconhecido():
    with pytest.raises(ValueError):
        desenhar_figura(Figura("pizza", "x", {}))  # type: ignore[arg-type]


def test_retorna_controle_flet(figura):
    assert isinstance(desenhar_figura(figura), ft.Column)


def test_boxplot_desenha_um_por_grupo():
    fig = boxplot([("A", [1, 2, 3, 4, 100]), ("B", [5, 6, 7])], "'y' por 'g'", "y")
    controle = desenhar_figura(fig)
    canvas = _canvas(controle)
    caixas = [s for s in canvas.shapes if isinstance(s, cv.Rect)]
    assert len(caixas) == 2 * 2  # preenchimento + contorno por grupo
    circulos = [s for s in canvas.shapes if isinstance(s, cv.Circle)]
    assert len(circulos) == 2 + 1  # uma média por grupo + o outlier 100
    rotulos = [s.value for s in canvas.shapes if isinstance(s, cv.Text)]
    assert "A (n = 5)" in rotulos and "B (n = 3)" in rotulos
    assert {"Mediana", "Média"} <= set(textos(controle))
    for forma in canvas.shapes:
        if isinstance(forma, cv.Rect):
            assert 0 <= forma.y and forma.y + forma.height <= ALTURA
        if isinstance(forma, cv.Circle):
            assert 0 <= forma.y <= ALTURA


def test_boxplot_grupo_constante():
    canvas = _canvas(desenhar_figura(boxplot([("A", [5, 5, 5])], "t", "y")))
    assert any(isinstance(s, cv.Rect) and s.height >= 1 for s in canvas.shapes)


def test_barras_desenha_valores_em_percentual_e_referencia():
    fig = barras(
        [("Sim (sucesso)", 0.62), ("Não", 0.38)],
        "Proporções em 'r'",
        "Proporção",
        maximo=1.0,
        referencias=[("p₀ = 0,5", 0.5, "tracejado")],
        percentual=True,
    )
    controle = desenhar_figura(fig)
    canvas = _canvas(controle)
    barras_ = [s for s in canvas.shapes if isinstance(s, cv.Rect)]
    assert len(barras_) == 2 * 2
    alturas = sorted({round(b.height, 6) for b in barras_})
    assert alturas[1] / alturas[0] == pytest.approx(0.62 / 0.38)
    rotulos = [s.value for s in canvas.shapes if isinstance(s, cv.Text)]
    assert {"62,0%", "38,0%", "Sim (sucesso)", "Não", "0,0%", "100,0%"} <= set(rotulos)
    tracejadas = [
        s for s in canvas.shapes if isinstance(s, cv.Line) and s.paint.stroke_dash_pattern
    ]
    assert len(tracejadas) == 1 and tracejadas[0].y1 == tracejadas[0].y2  # linha horizontal
    assert "p₀ = 0,5" in textos(controle)


def test_barras_agrupadas_uma_cor_por_serie():
    fig = barras_agrupadas(
        [("manhã", [0.5, 0.375, 0.125]), ("noite", [0.2, 0.3, 0.5])],
        ["A", "B", "C"],
        "'nota' por 'turno'",
        "Proporção",
        maximo=1.0,
        percentual=True,
    )
    controle = desenhar_figura(fig)
    barras_ = [s for s in _canvas(controle).shapes if isinstance(s, cv.Rect)]
    assert len(barras_) == 6
    assert [b.paint.color for b in barras_] == [cor_serie(j) for j in range(3)] * 2
    assert cor_serie(0) == tema.LARANJA and cor_serie(1) == tema.ROXO
    rotulos = [s.value for s in _canvas(controle).shapes if isinstance(s, cv.Text)]
    assert {"manhã", "noite", "50,0%", "12,5%"} <= set(rotulos)
    assert {"A", "B", "C"} <= set(textos(controle))  # legenda
    for b in barras_:
        assert 0 <= b.x and b.x + b.width <= LARGURA and b.y + b.height <= ALTURA


def test_paleta_de_series_cicla():
    assert cor_serie(len(tema.GRAFICO_SERIES)) == cor_serie(0)


@pytest.mark.parametrize("agrupadas", [False, True])
def test_barras_negativas_descem_a_partir_do_zero(agrupadas):
    if agrupadas:
        fig = barras_agrupadas([("g", [-40.0, 20.0])], ["A", "B"], "t", "y")
    else:
        fig = barras([("A", -40.0), ("B", 20.0)], "t", "y")
    canvas = _canvas(desenhar_figura(fig))
    retangulos = [s for s in canvas.shapes if isinstance(s, cv.Rect)]
    negativa, positiva = retangulos[0], retangulos[-1]
    zero = positiva.y + positiva.height  # base da barra positiva = linha do zero
    assert negativa.y == pytest.approx(zero)  # a negativa começa no zero e desce
    assert negativa.height == pytest.approx(2 * positiva.height)  # |−40| = 2·20
    assert negativa.y + negativa.height <= ALTURA
    linhas_zero = [s for s in canvas.shapes if isinstance(s, cv.Line) and s.y1 == s.y2 == zero]
    assert linhas_zero  # eixo horizontal na altura do zero


def test_rotulos_longos_de_faixas_quebram_em_duas_linhas():
    # 10 faixas do IV ("5,00 – 11,80"...) não cabem lado a lado: quebram no " – ".
    categorias = [(f"{i},00 – {i + 6},80", 0.5 - i / 10) for i in range(10)]
    canvas = _canvas(desenhar_figura(barras(categorias, "WoE", "WoE")))
    rotulos = [s.value for s in canvas.shapes if isinstance(s, cv.Text) and "–" in s.value]
    assert rotulos and all(r.count("\n") == 1 and r.split("\n")[0].endswith("–") for r in rotulos)
    curtas = _canvas(desenhar_figura(barras([("A", 1.0), ("B", 2.0)], "t", "y")))
    assert not any("\n" in s.value for s in curtas.shapes if isinstance(s, cv.Text))
