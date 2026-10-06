"""Desenho das figuras com flet.canvas."""

import flet as ft
import flet.canvas as cv
import pytest
from ajudantes_ui import do_tipo, textos

from app.ui import tema
from app.ui.graficos import ALTURA, LARGURA, desenhar_figura
from core.base import Figura
from core.figuras import histograma


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
