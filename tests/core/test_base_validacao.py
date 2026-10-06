"""Contratos de core/base.py, o formulário do t de uma amostra e core/validacao.py."""

import math

import pandas as pd
import pytest

from core.base import ErroValidacao, TesteBase, TesteNaoImplementado
from core.testes.medias import TesteT1Amostra
from core.tipos import detectar_tipos
from core.validacao import colunas_por_tipo, converter_numero


def test_teste_base_e_abstrata():
    with pytest.raises(TypeError):
        TesteBase()  # type: ignore[abstract]


def test_erro_validacao_aceita_texto_ou_lista():
    assert ErroValidacao("a").mensagens == ["a"]
    erro = ErroValidacao(["a", "b"])
    assert erro.mensagens == ["a", "b"]
    assert str(erro) == "a\nb"


def test_t_1am_parametros_do_formulario_atual():
    specs = TesteT1Amostra().parametros()
    assert [(s.nome, s.rotulo, s.tipo) for s in specs] == [
        ("coluna", "Variável", "coluna_numerica"),
        ("mu0", "Média Hipotética", "numero"),
        ("alfa", "Nível de significância (α)", "alfa"),
    ]
    assert specs[2].padrao == 0.05


def test_t_1am_comparacao_inicial_sem_p_valores():
    comp = TesteT1Amostra().comparacao_inicial()
    assert (comp.titulo_esquerda, comp.titulo_direita) == ("t Student", "Wilcoxon")
    assert comp.hipoteses == ["μ ≠ μ₀", "μ > μ₀", "μ < μ₀"]
    assert comp.linhas == [(None, None)] * 3


def test_t_1am_calculo_ainda_nao_implementado(df_exemplo):
    teste = TesteT1Amostra()
    with pytest.raises(TesteNaoImplementado, match="ainda não foi implementado"):
        teste.validar(df_exemplo, {})
    with pytest.raises(TesteNaoImplementado):
        teste.executar(df_exemplo, {})


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [("10", 10.0), (" 1,5 ", 1.5), ("1.5", 1.5), ("1.234,5", 1234.5), ("-0,25", -0.25), (3, 3.0)],
)
def test_converter_numero(texto, esperado):
    assert converter_numero(texto) == pytest.approx(esperado)


@pytest.mark.parametrize("texto", ["", "   ", None, "abc", "1,2,3", "nan", "inf", math.inf])
def test_converter_numero_invalido(texto):
    with pytest.raises(ValueError):
        converter_numero(texto)


@pytest.fixture
def df_tipos() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "valor": [1.5, 2.5, 3.5, 4.5],
            "grupo": ["a", "b", "a", "b"],
            "nota": [1, 2, 3, 4],
            "flag": [True, False, True, True],
            "cidade": ["x", "y", "z", "x"],
        }
    )


def test_colunas_por_tipo(df_tipos):
    assert colunas_por_tipo(df_tipos, "coluna_numerica") == ["valor", "nota"]
    assert colunas_por_tipo(df_tipos, "multi_coluna") == ["valor", "nota"]
    assert colunas_por_tipo(df_tipos, "coluna_binaria") == ["grupo", "flag"]
    assert colunas_por_tipo(df_tipos, "coluna_categorica") == [
        "grupo",
        "nota",
        "flag",
        "cidade",
    ]


def test_colunas_por_tipo_usa_perfis_prontos(df_tipos):
    perfis = detectar_tipos(df_tipos[["valor"]])  # só "valor": prova que não recalcula
    assert colunas_por_tipo(df_tipos, "coluna_numerica", perfis) == ["valor"]


def test_colunas_por_tipo_desconhecido(df_tipos):
    with pytest.raises(ValueError):
        colunas_por_tipo(df_tipos, "numero")
