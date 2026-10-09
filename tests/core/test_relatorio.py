"""Relatório em PDF (core/relatorio.py): conteúdo, figuras e formatação.

O PDF é lido de volta com pypdf (texto e imagens de cada página). Os 19 testes do catálogo são
exportados sobre uma base sintética (semente fixa) e o texto extraído é conferido contra o
`ResultadoTeste`: decisão, interpretação, avisos, tabelas, card e uma imagem por figura.
"""

import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from fontTools.ttLib import TTFont
from pypdf import PdfReader

from core import registry
from core.base import ComparacaoPValores, Figura, GrupoFiguras, ParametroSpec, ResultadoTeste
from core.figuras import barras, barras_agrupadas, boxplot, cascata, dispersao, histograma
from core.relatorio import (
    RODAPE,
    ContextoRelatorio,
    _caminho_fontes,
    _celula,
    _figuras,
    descrever_parametros,
    figura_png,
    garantir_extensao,
    gerar_pdf,
    nome_sugerido,
    tabela_comparacao,
)

QUANDO = datetime(2026, 10, 8, 14, 5, 9)
PNG = b"\x89PNG\r\n\x1a\n"


def _base(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    grupo = rng.choice(["A", "B", "C"], n)
    x1, x2 = rng.normal(50, 10, n), rng.uniform(0, 100, n)
    y = 10 + 0.5 * x1 - 0.2 * x2 + np.where(grupo == "B", 3, 0) + rng.normal(0, 5, n)
    return pd.DataFrame(
        {
            "y": np.round(y, 3),
            "x1": np.round(x1, 3),
            "x2": np.round(x2, 3),
            "x3": np.round(x1 + rng.normal(0, 8, n), 3),
            "grupo": grupo,
            "fator": rng.choice(["p", "q"], n),
            "binaria": rng.choice(["Sim", "Não"], n, p=[0.4, 0.6]),
            "binaria2": rng.choice(["Sim", "Não"], n),
        }
    )


PARAMS = {
    "teste_t_1am": {"coluna": "y", "mu0": 40.0},
    "teste_t_2am": {"coluna": "y", "grupo": "binaria"},
    "teste_t_pareado": {"coluna1": "x1", "coluna2": "x3"},
    "teste_z_1prop": {"coluna": "binaria", "sucesso": "Sim", "p0": 0.4},
    "teste_z_2prop": {"coluna": "binaria", "sucesso": "Sim", "grupo": "binaria2"},
    "qui_quadrado": {"coluna1": "grupo", "coluna2": "fator"},
    "fisher": {"coluna1": "binaria", "evento1": "Sim", "coluna2": "binaria2", "evento2": "Sim"},
    "mcnemar": {"coluna1": "binaria", "coluna2": "binaria2", "evento": "Sim"},
    "teste_sinal": {"coluna1": "y", "m0": 40.0},
    "wilcoxon": {"coluna1": "y", "m0": 40.0},
    "mann_whitney": {"coluna": "y", "grupo": "binaria"},
    "kruskal_wallis": {"coluna": "y", "grupo": "grupo", "dunn": True},
    "friedman": {"colunas": ["x1", "x2", "x3"], "comparacoes": True},
    "anova_1fator": {"coluna": "y", "grupo": "grupo", "tukey": True},
    "anova_2fator": {"coluna": "y", "fator_a": "grupo", "fator_b": "fator"},
    "correlacao_pearson": {"x": "x1", "y": "y"},
    "correlacao_spearman": {"x": "x1", "y": "y"},
    "regres_linear": {"y": "y", "preditores": ["x1", "x2", "grupo"]},
    "regres_logit": {"y": "binaria", "evento": "Sim", "preditores": ["x1", "grupo"]},
    "correlacao_parcial": {"x": "x1", "y": "y", "controles": ["x2"]},
    "information_value": {"y": "binaria", "evento": "Sim", "preditores": ["x1", "grupo", "fator"]},
}


def _params(info: registry.TesteInfo) -> dict:
    params = {"alfa": 0.05} | PARAMS[info.id]
    for spec in info.criar().parametros():
        if spec.nome not in params and spec.padrao is not None:
            params[spec.nome] = spec.padrao
    return params


def _contexto(info, params, df) -> ContextoRelatorio:
    specs = info.criar().parametros()
    return ContextoRelatorio(
        info.nome, "base.csv", len(df), QUANDO, descrever_parametros(specs, params)
    )


def _ler(conteudo: bytes) -> tuple[list[str], int]:
    """Texto de cada página (espaços normalizados) e o número total de imagens."""
    leitor = PdfReader(io.BytesIO(conteudo))
    textos = [re.sub(r"\s+", " ", pagina.extract_text()) for pagina in leitor.pages]
    return textos, sum(len(pagina.images) for pagina in leitor.pages)


def _corpo(paginas: list[str], nome_teste: str) -> str:
    """Texto contínuo do relatório, sem o cabeçalho e o rodapé de cada página (um parágrafo que
    passa de uma página para a outra fica inteiro)."""
    rodape = re.compile(rf"\s*{RODAPE}\s*Página \d+ de \d+\s*$")
    partes = []
    for pagina in paginas:
        texto = pagina[len(nome_teste) :] if pagina.startswith(nome_teste) else pagina
        partes.append(rodape.sub("", texto))
    return " ".join(partes)


def _compacto(texto: str) -> str:
    return re.sub(r"\s+", "", texto)


@pytest.fixture(scope="module")
def relatorios() -> dict[str, tuple[ResultadoTeste, ContextoRelatorio, bytes]]:
    df = _base()
    saida = {}
    for info in registry.listar():
        params = _params(info)
        resultado = info.criar().executar(df, params)
        contexto = _contexto(info, params, df)
        saida[info.id] = (resultado, contexto, gerar_pdf(resultado, contexto))
    return saida


# ---------------------------------------------------------------------------
# Os 19 testes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("teste_id", [info.id for info in registry.listar()])
def test_conteudo_do_relatorio(relatorios, teste_id):
    resultado, contexto, conteudo = relatorios[teste_id]
    assert conteudo.startswith(b"%PDF")
    paginas, imagens = _ler(conteudo)
    tudo = _compacto(_corpo(paginas, contexto.nome_teste))

    for i, pagina in enumerate(paginas, start=1):
        # Cabeçalho: só o nome do teste; "smartETL" só no rodapé ("by smartETL").
        assert pagina.startswith(contexto.nome_teste)
        assert pagina.count("smartETL") == 1 and RODAPE in pagina
        assert f"Página {i} de {len(paginas)}" in pagina
    assert "Arquivo: base.csv (150 linhas)" in paginas[0]
    assert "Gerado em: 08/10/2026 às 14:05" in paginas[0]
    for rotulo, valor in contexto.parametros:
        assert _compacto(rotulo) in tudo and _compacto(valor) in tudo

    if resultado.secoes:
        for secao in resultado.secoes:
            assert _compacto(secao.titulo) in tudo
            for texto in [*secao.textos, *secao.avisos, *secao.notas]:
                assert _compacto(texto) in tudo
        tabelas = [df for secao in resultado.secoes for df in secao.tabelas.values()]
    else:
        assert _compacto(resultado.decisao.replace("H0", "H₀")) in tudo
        assert _compacto(resultado.interpretacao) in tudo
        for aviso in resultado.avisos:
            assert _compacto(aviso) in tudo
        tabelas = list(resultado.tabelas.values())
    for df in tabelas:
        for coluna in df.columns:
            assert _compacto(str(coluna)) in tudo
        # Células longas quebram linha e a extração mistura as colunas: confere palavra a palavra.
        palavras = [p for v in df.iloc[0].tolist() for p in _celula(v).split()]
        assert all(_compacto(p) in tudo for p in palavras)

    if resultado.comparacao is not None:
        assert _compacto("Comparação de p-valores") in tudo
        for hipotese in resultado.comparacao.hipoteses:
            assert _compacto(hipotese) in tudo
    assert imagens == len(_figuras(resultado.figuras))
    assert "Simulação" not in " ".join(paginas)


def test_fontes_cobrem_todos_os_caracteres(relatorios):
    """A DejaVu Sans tem um glifo para cada caractere escrito (μ, ρₛ, H₀, ≠, η², ⚠...)."""
    mapas = [
        set(TTFont(_caminho_fontes() / nome).getBestCmap())
        for nome in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
    ]
    cobertos = set.intersection(*mapas)
    textos = []
    for resultado, contexto, _ in relatorios.values():
        textos += [
            contexto.nome_teste,
            resultado.decisao,
            resultado.interpretacao,
            *resultado.avisos,
        ]
        textos += [t for par in contexto.parametros for t in par]
        for secao in resultado.secoes:
            textos += [secao.titulo, *secao.textos, *secao.avisos, *secao.notas]
        for df in [
            *resultado.tabelas.values(),
            *(t for s in resultado.secoes for t in s.tabelas.values()),
        ]:
            textos += [str(c) for c in df.columns] + [str(v) for v in df.to_numpy().ravel()]
            textos += list(df.attrs.get("dicas", {}).values())
        if resultado.comparacao is not None:
            textos += resultado.comparacao.hipoteses
    faltando = {c for c in "".join([*textos, "⚠•Hₐ"]) if ord(c) not in cobertos}
    assert not faltando


def test_regressao_tem_secoes_tabelas_largas_e_grupo_de_figuras(relatorios):
    resultado, _, conteudo = relatorios["regres_linear"]
    paginas, imagens = _ler(conteudo)
    grupos = [f for f in resultado.figuras if isinstance(f, GrupoFiguras)]
    assert grupos and imagens > len(resultado.figuras)  # todas as opções do "Eixo X"
    texto = _corpo(paginas, "Regressão Linear")
    assert texto.index("Pressupostos") < texto.index("Coeficientes")  # ordem das seções
    # Dicas das colunas (tooltips na interface) viram notas no PDF.
    dicas = [
        d
        for s in resultado.secoes
        for t in s.tabelas.values()
        for d in t.attrs.get("dicas", {}).values()
    ]
    assert dicas and all(_compacto(d) in _compacto(texto) for d in dicas)


# ---------------------------------------------------------------------------
# Partes isoladas
# ---------------------------------------------------------------------------


def test_descrever_parametros():
    specs = [
        ParametroSpec("coluna", "Variável", "coluna_numerica"),
        ParametroSpec("mu0", "Média Hipotética", "numero"),
        ParametroSpec("alfa", "Nível de significância (α)", "alfa"),
        ParametroSpec("tukey", "Tukey", "booleano"),
        ParametroSpec("colunas", "Colunas", "multi_coluna"),
        ParametroSpec("refs", "Referências", "niveis_referencia"),
        ParametroSpec("opcional", "Opcional", "opcao", obrigatorio=False),
    ]
    params = {
        "coluna": "nota",
        "mu0": 7.25,
        "alfa": 0.1,
        "tukey": False,
        "colunas": ["a", "b"],
        "refs": {"sexo": "F", "turno": "Manhã"},
        "opcional": None,
    }
    assert descrever_parametros(specs, params) == [
        ("Variável", "nota"),
        ("Média Hipotética", "7,25"),
        ("Nível de significância (α)", "0,10"),
        ("Tukey", "Não"),
        ("Colunas", "a, b"),
        ("Referências", "sexo: F; turno: Manhã"),
        ("Opcional", "—"),
    ]
    assert descrever_parametros(specs[:2], {"coluna": "x", "mu0": 40.0}) == [
        ("Variável", "x"),
        ("Média Hipotética", "40"),
    ]


def test_nomes_de_arquivo():
    assert nome_sugerido("anova_1fator", QUANDO) == "anova_1fator_2026-10-08_140509.pdf"
    assert garantir_extensao(r"C:\saida\relatorio") == r"C:\saida\relatorio.pdf"
    assert garantir_extensao("relatorio.PDF") == "relatorio.PDF"


def test_tabela_comparacao():
    comparacao = ComparacaoPValores(
        "t Student", "Wilcoxon", ["μ ≠ μ₀", "μ > μ₀"], [(0.0004, 0.2), (None, 0.5)]
    )
    df = tabela_comparacao(comparacao)
    assert list(df.columns) == ["Hipótese", "p-valor (t Student)", "p-valor (Wilcoxon)"]
    assert df.values.tolist() == [["Hₐ: μ ≠ μ₀", "< 0,001", "0,200"], ["Hₐ: μ > μ₀", "—", "0,500"]]


def test_titulos_de_grupo_de_figuras():
    a = histograma([1.0, 2.0, 3.0], "Resíduos × x", "x")
    b = histograma([1.0, 2.0, 3.0], "Resíduos × z", "z")
    igual = histograma([1.0, 2.0], "Mesmo título", "v")
    grupo = GrupoFiguras("Eixo X", {"x": a, "z": b}, "x")
    repetido = GrupoFiguras("Escala", {"Valores": igual, "Postos": igual}, "Valores")
    assert [t for t, _ in _figuras([grupo, repetido, a])] == [
        "Resíduos × x",
        "Resíduos × z",
        "Mesmo título — Escala: Valores",
        "Mesmo título — Escala: Postos",
        "Resíduos × x",
    ]


@pytest.mark.parametrize(
    "figura",
    [
        histograma(
            [1.0, 2.0, 2.5, 4.0], "h", "x", [("Média", 2.4, "destaque"), ("μ₀", 3, "tracejado")]
        ),
        boxplot(
            [("A", [1.0, 2.0, 3.0, 10.0]), ("B", [2.0, 3.0])], "b", "y", [("ref", 2, "tracejado")]
        ),
        barras([("Sim", 0.4), ("Não", 0.6)], "p", "Proporção", 1, [("p₀", 0.5, "tracejado")], True),
        barras_agrupadas([("A", [1.0, 2.0]), ("B", [3.0, 4.0])], ["s1", "s2"], "g", "Média"),
        barras_agrupadas([("A", [-50.0, -48.0]), ("B", [-49.0, 5.0])], ["s1", "s2"], "n", "Média"),
        dispersao(
            np.arange(3000.0),
            np.arange(3000.0) ** 0.5,
            "d",
            "x",
            "y",
            [("reta", 0, 0, 1, 1, "destaque")],
        ),
        dispersao(
            [0, 0.5, 1], [0, 0.8, 1], "ROC", "1 − especificidade", "sensibilidade", conectar=True
        ),
        cascata(("Intercepto", 10.0), [("x1", 2.5), ("grupo", -1.0)], "Previsão", "y", "c"),
    ],
    ids=lambda f: f.tipo,
)
def test_figura_png(figura: Figura):
    png = figura_png(figura)
    assert png.startswith(PNG) and len(png) > 5000
