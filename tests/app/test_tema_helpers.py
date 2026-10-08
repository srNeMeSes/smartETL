"""Paleta única e helpers de layout."""

import re
from pathlib import Path

import flet as ft
import pytest

from app.ui import helpers, tema

RAIZ_APP = Path(__file__).resolve().parents[2] / "app"
COR_HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")


def test_paleta_da_identidade():
    assert tema.LARANJA == "#FF6A1A"
    assert tema.LARANJA_SUAVE == "#FFF1E6"
    assert tema.FUNDO == "#F4F5F7"
    assert tema.CARTAO == "#FFFFFF"
    assert tema.BORDA == "#E7E8EC"
    assert tema.TEXTO == "#232529"
    assert (tema.TEXTO_SECUNDARIO, tema.TEXTO_TERCIARIO) == ("#8A8D93", "#B4B6BC")
    assert tema.ROXO == "#7C3AED"


def test_nenhuma_cor_hardcoded_fora_do_tema():
    ofensores = {
        str(arquivo.relative_to(RAIZ_APP)): COR_HEX.findall(arquivo.read_text(encoding="utf-8"))
        for arquivo in RAIZ_APP.rglob("*.py")
        if arquivo.name != "tema.py"
    }
    assert {nome: cores for nome, cores in ofensores.items() if cores} == {}


@pytest.mark.parametrize(
    ("kwargs", "esperado"),
    [
        ({"all": 5}, (5, 5, 5, 5)),
        ({"horizontal": 18, "vertical": 9}, (18, 9, 18, 9)),
        ({"vertical": 4}, (0, 4, 0, 4)),
        ({"left": 1, "top": 2, "right": 3, "bottom": 4}, (1, 2, 3, 4)),
    ],
)
def test_pad(kwargs, esperado):
    p = helpers.pad(**kwargs)
    assert (p.left, p.top, p.right, p.bottom) == esperado


def test_border_all():
    borda = helpers.border_all(2, tema.BORDA)
    for lado in (borda.top, borda.right, borda.bottom, borda.left):
        assert (lado.width, lado.color) == (2, tema.BORDA)


def test_border_only():
    lado = ft.BorderSide(1, tema.BORDA)
    borda = helpers.border_only(right=lado)
    assert borda.right == lado
    assert borda.left != lado


def test_cores_do_relatorio_pdf_iguais_as_do_tema():
    from core.relatorio import CORES, SERIES

    assert CORES == {
        "laranja": tema.LARANJA,
        "laranja_suave": tema.LARANJA_SUAVE,
        "roxo": tema.ROXO,
        "fundo": tema.FUNDO,
        "borda": tema.BORDA,
        "texto": tema.TEXTO,
        "texto_secundario": tema.TEXTO_SECUNDARIO,
        "rejeita": tema.DECISAO_REJEITA,
        "nao_rejeita": tema.DECISAO_NAO_REJEITA,
    }
    assert SERIES == tema.GRAFICO_SERIES


def test_cores_da_decisao():
    # Rejeitar H₀ (o teste encontrou o efeito) em verde; não rejeitar em terracota suave.
    assert tema.DECISAO_REJEITA == "#3F9B6B"
    assert tema.DECISAO_NAO_REJEITA == "#C8705F"


def test_indicador_da_aba_ativa_em_laranja():
    from app.ui.painel_abas import PainelAbas

    assert PainelAbas()._barra.indicator_color == tema.LARANJA
