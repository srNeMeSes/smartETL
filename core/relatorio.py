"""Relatório em PDF de um resultado: a aba Análise inteira e os gráficos da Visualização.

Sem Flet. O PDF é montado com fpdf2 e os gráficos com matplotlib (backend Agg, importado só
quando um relatório é gerado). A fonte é a DejaVu Sans que acompanha o matplotlib: cobre os
símbolos usados nos textos (μ, ρₛ, H₀, ≠, ≤, η², −...).

Conteúdo, na ordem: cabeçalho com o nome do teste (em todas as páginas); arquivo, data/hora e
parâmetros; decisão (com a cor da interface), interpretação e avisos — ou as seções, quando o
teste as define (regressões); tabelas (com as dicas das colunas em nota); o card de comparação
como tabela; todas as figuras (cada opção de um `GrupoFiguras`). A aba Simulação fica de fora.
Rodapé: "by smartETL" e o número da página.
"""

import io
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.base import (
    ComparacaoPValores,
    Figura,
    GrupoFiguras,
    ParametroSpec,
    ResultadoTeste,
    Secao,
)
from core.interpretacao import (
    NAO_REJEITA_H0,
    REJEITA_H0,
    SEM_VALOR,
    formatar_alfa,
    formatar_numero,
    formatar_p_valor,
)

# Cores do relatório: os mesmos valores de app/ui/tema.py (conferido em tests/app).
CORES = {
    "laranja": "#FF6A1A",
    "laranja_suave": "#FFF1E6",
    "roxo": "#7C3AED",
    "fundo": "#F4F5F7",
    "borda": "#E7E8EC",
    "texto": "#232529",
    "texto_secundario": "#8A8D93",
    "rejeita": "#D0605A",
    "nao_rejeita": "#3F9B6B",
}
SERIES = ("#FF6A1A", "#7C3AED", "#0F9D8A", "#E0A100", "#5B6B7F", "#D9467A", "#3B82C4", "#8C6D46")

RODAPE = "by smartETL"
FONTE = "DejaVu"
MARGEM = 15  # mm
TITULO_COMPARACAO = "Comparação de p-valores"
FONTE_TABELA, FONTE_TABELA_MIN = 8.5, 6.0


@dataclass(frozen=True)
class ContextoRelatorio:
    """O que o relatório mostra além do resultado."""

    nome_teste: str
    arquivo: str | None
    linhas: int | None
    gerado_em: datetime
    parametros: list[tuple[str, str]]  # (rótulo, valor já formatado)


# ---------------------------------------------------------------------------
# Textos auxiliares
# ---------------------------------------------------------------------------


def nome_sugerido(teste_id: str, quando: datetime) -> str:
    """Nome padrão do arquivo: <teste>_<data-hora>.pdf."""
    return f"{teste_id}_{quando:%Y-%m-%d_%H%M%S}.pdf"


def _formatar_valor(spec: ParametroSpec, valor: object) -> str:
    if valor is None or valor == "" or valor == [] or valor == {}:
        return SEM_VALOR
    if spec.tipo == "alfa":
        return formatar_alfa(float(valor))
    if spec.tipo == "booleano":
        return "Sim" if valor else "Não"
    if isinstance(valor, dict):
        return "; ".join(f"{chave}: {nivel}" for chave, nivel in valor.items())
    if isinstance(valor, (list, tuple)):
        return ", ".join(str(v) for v in valor)
    if isinstance(valor, float):
        return f"{valor:.10g}".replace(".", ",")
    return str(valor)


def descrever_parametros(specs: Sequence[ParametroSpec], params: dict) -> list[tuple[str, str]]:
    """(rótulo, valor) de cada campo do formulário, na ordem do formulário."""
    return [(spec.rotulo, _formatar_valor(spec, params.get(spec.nome))) for spec in specs]


def _decisao(texto: str) -> tuple[str, str]:
    """Texto da decisão com H₀ e a cor (vermelho/verde suaves, como na interface)."""
    normalizado = texto.replace("H₀", "H0")
    cor = {REJEITA_H0: "rejeita", NAO_REJEITA_H0: "nao_rejeita"}.get(normalizado, "texto")
    return texto.replace("H0", "H₀"), CORES[cor]


def _celula(valor: object) -> str:
    """Como a tabela da interface: NaN vazio e vírgula decimal."""
    try:
        if pd.isna(valor):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(valor, float):
        return str(valor).replace(".", ",")
    return str(valor)


def tabela_comparacao(comparacao: ComparacaoPValores) -> pd.DataFrame:
    """O card de comparação como tabela: uma linha por hipótese alternativa."""
    linhas = [
        (f"Hₐ: {h}", formatar_p_valor(esq), formatar_p_valor(dir_))
        for h, (esq, dir_) in zip(comparacao.hipoteses, comparacao.linhas, strict=True)
    ]
    colunas = ["Hipótese", f"p-valor ({comparacao.titulo_esquerda})"]
    return pd.DataFrame(linhas, columns=[*colunas, f"p-valor ({comparacao.titulo_direita})"])


# ---------------------------------------------------------------------------
# Gráficos (matplotlib)
# ---------------------------------------------------------------------------


def _numero_eixo(valor: float, _posicao: int = 0) -> str:
    texto = f"{valor:.6g}".replace(".", ",")
    return texto.replace("-", "−")


def _percentual_eixo(valor: float, _posicao: int = 0) -> str:
    return f"{formatar_numero(valor * 100, 0)}%"


def _rotulo_valor(valor: float, percentual: bool) -> str:
    return f"{formatar_numero(valor * 100, 1)}%" if percentual else formatar_numero(valor, 2)


def _estilo_linha(estilo: str) -> dict:
    if estilo == "tracejado":
        return {"color": CORES["texto_secundario"], "linestyle": (0, (5, 3)), "linewidth": 1.4}
    return {"color": CORES["texto"], "linestyle": "-", "linewidth": 1.8}


def _histograma(ax, dados: dict) -> None:
    bordas, contagens = dados["bordas"], dados["contagens"]
    larguras = [b - a for a, b in itertools.pairwise(bordas)]
    ax.bar(
        bordas[:-1],
        contagens,
        width=larguras,
        align="edge",
        color=CORES["laranja_suave"],
        edgecolor=CORES["laranja"],
        linewidth=1,
    )
    for ref in dados.get("referencias", []):
        ax.axvline(ref["valor"], label=ref["rotulo"], **_estilo_linha(ref["estilo"]))
    ax.set_xlabel(dados["rotulo_x"])
    ax.set_ylabel("Frequência")


def _boxplot(ax, dados: dict) -> None:
    caixas = [
        {
            "label": f"{g['rotulo']} (n = {g['n']})",
            "q1": g["q1"],
            "med": g["mediana"],
            "q3": g["q3"],
            "whislo": g["bigode_inf"],
            "whishi": g["bigode_sup"],
            "mean": g["media"],
            "fliers": g["outliers"],
        }
        for g in dados["grupos"]
    ]
    ax.bxp(
        caixas,
        showmeans=True,
        patch_artist=True,
        boxprops={"facecolor": CORES["laranja_suave"], "edgecolor": CORES["laranja"]},
        medianprops={"color": CORES["texto"], "linewidth": 2},
        whiskerprops={"color": CORES["texto_secundario"]},
        capprops={"color": CORES["texto_secundario"]},
        meanprops={"marker": "o", "markerfacecolor": CORES["texto"], "markeredgecolor": "none"},
        flierprops={"marker": "o", "markerfacecolor": "none", "markeredgecolor": CORES["laranja"]},
    )
    ax.plot([], [], color=CORES["texto"], linewidth=2, label="Mediana")
    ax.plot([], [], "o", color=CORES["texto"], label="Média")
    for ref in dados.get("referencias", []):
        ax.axhline(ref["valor"], label=ref["rotulo"], **_estilo_linha(ref["estilo"]))
    ax.set_ylabel(dados["rotulo_y"])


def _limites_barras(ax, dados: dict) -> None:
    """Eixo de `minimo` (0, ou negativo) a `maximo`, com folga para os rótulos das barras."""
    minimo, maximo = float(dados.get("minimo", 0.0)), float(dados["maximo"])
    folga = (maximo - minimo) * 0.12
    ax.set_ylim(minimo - (folga if minimo < 0 else 0.0), maximo + folga)
    if minimo < 0:
        ax.axhline(0, color=CORES["borda"], linewidth=1)


def _barras(ax, dados: dict) -> None:
    percentual = bool(dados.get("percentual"))
    rotulos = [c["rotulo"] for c in dados["categorias"]]
    valores = [c["valor"] for c in dados["categorias"]]
    barras = ax.bar(
        rotulos, valores, color=CORES["laranja_suave"], edgecolor=CORES["laranja"], width=0.6
    )
    ax.bar_label(barras, labels=[_rotulo_valor(v, percentual) for v in valores], padding=2)
    for ref in dados.get("referencias", []):
        ax.axhline(ref["valor"], label=ref["rotulo"], **_estilo_linha(ref["estilo"]))
    _limites_barras(ax, dados)
    ax.set_ylabel(dados["rotulo_y"])
    if percentual:
        ax.yaxis.set_major_formatter(_percentual_eixo)


def _barras_agrupadas(ax, dados: dict) -> None:
    percentual = bool(dados.get("percentual"))
    grupos, series = dados["grupos"], dados["series"]
    largura = 0.8 / max(len(series), 1)
    for j, serie in enumerate(series):
        posicoes = [i - 0.4 + largura * (j + 0.5) for i in range(len(grupos))]
        valores = [g["valores"][j] for g in grupos]
        barras = ax.bar(
            posicoes, valores, width=largura * 0.95, color=SERIES[j % len(SERIES)], label=serie
        )
        if len(grupos) * len(series) <= 24:
            ax.bar_label(
                barras,
                labels=[_rotulo_valor(v, percentual) for v in valores],
                padding=2,
                fontsize=7,
            )
    ax.set_xticks(range(len(grupos)), [g["rotulo"] for g in grupos])
    _limites_barras(ax, dados)
    ax.set_ylabel(dados["rotulo_y"])
    if percentual:
        ax.yaxis.set_major_formatter(_percentual_eixo)


def _dispersao(ax, dados: dict) -> None:
    if dados.get("conectar"):
        ax.plot(dados["x"], dados["y"], color=CORES["laranja"], linewidth=2)
    else:
        ax.scatter(dados["x"], dados["y"], s=10, color=CORES["laranja"], alpha=0.8, linewidths=0)
    for linha in dados.get("linhas", []):
        ax.plot(
            [linha["x1"], linha["x2"]],
            [linha["y1"], linha["y2"]],
            label=linha["rotulo"],
            **_estilo_linha(linha["estilo"]),
        )
    if dados["n_total"] > len(dados["x"]):
        ax.plot(
            [],
            [],
            "o",
            color=CORES["laranja"],
            label=f"{len(dados['x'])} de {dados['n_total']} pontos",
        )
    ax.set_xlabel(dados["rotulo_x"])
    ax.set_ylabel(dados["rotulo_y"])


def _cascata(ax, dados: dict) -> None:
    inicio, etapas, final = dados["inicio"], dados["etapas"], dados["final"]
    barras = [(inicio["rotulo"], 0.0, inicio["valor"], CORES["laranja"])]
    barras += [
        (e["rotulo"], e["de"], e["ate"], SERIES[1] if e["valor"] < 0 else SERIES[2]) for e in etapas
    ]
    barras.append((final["rotulo"], 0.0, final["valor"], CORES["texto"]))
    for i, (_rotulo, de, ate, cor) in enumerate(barras):
        ax.bar(i, ate - de, bottom=de, color=cor, width=0.6)
        meio = 0 < i < len(barras) - 1
        valor = ate - de if meio else ate
        texto = ("+" if meio and valor > 0 else "") + formatar_numero(valor, 2)
        ax.annotate(texto, (i, max(de, ate)), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(barras)), [b[0] for b in barras])
    ax.axhline(0, color=CORES["borda"], linewidth=1)
    ax.set_ylabel(dados["rotulo_y"])


_DESENHOS = {
    "histograma": _histograma,
    "boxplot": _boxplot,
    "barras": _barras,
    "barras_agrupadas": _barras_agrupadas,
    "dispersao": _dispersao,
    "cascata": _cascata,
}


def figura_png(figura: Figura, largura_pol: float = 7.0, altura_pol: float = 3.6) -> bytes:
    """PNG do gráfico (o título fica no PDF, fora da imagem)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.edgecolor": CORES["borda"],
            "axes.labelcolor": CORES["texto"],
            "xtick.color": CORES["texto_secundario"],
            "ytick.color": CORES["texto_secundario"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": True,
        }
    ):
        fig, ax = plt.subplots(figsize=(largura_pol, altura_pol), dpi=200)
        try:
            _DESENHOS[figura.tipo](ax, figura.dados)
            if not (
                figura.tipo in ("barras", "barras_agrupadas") and figura.dados.get("percentual")
            ):
                ax.yaxis.set_major_formatter(_numero_eixo)
            if figura.tipo in ("histograma", "dispersao"):
                ax.xaxis.set_major_formatter(_numero_eixo)
            rotulos = [t.get_text() for t in ax.get_xticklabels()]
            if figura.tipo != "histograma" and figura.tipo != "dispersao":
                if sum(len(r) for r in rotulos) > 70:
                    ax.tick_params(axis="x", labelrotation=30)
                    for rotulo in ax.get_xticklabels():
                        rotulo.set_horizontalalignment("right")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False, fontsize=7, loc="best")
            fig.tight_layout()
            saida = io.BytesIO()
            fig.savefig(saida, format="png", facecolor="white")
            return saida.getvalue()
        finally:
            plt.close(fig)


def _figuras(itens: Sequence[Figura | GrupoFiguras]) -> list[tuple[str, Figura]]:
    """(título, figura) de cada gráfico, abrindo todas as opções de um GrupoFiguras."""
    saida: list[tuple[str, Figura]] = []
    for item in itens:
        if isinstance(item, GrupoFiguras):
            titulos = [figura.titulo for figura in item.opcoes.values()]
            distintos = len(set(titulos)) == len(titulos)
            for opcao, figura in item.opcoes.items():
                sufixo = "" if distintos else f" — {item.rotulo}: {opcao}"
                saida.append((figura.titulo + sufixo, figura))
        else:
            saida.append((item.titulo, item))
    return saida


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _rgb(cor: str) -> tuple[int, int, int]:
    return tuple(int(cor[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def _caminho_fontes() -> Path:
    import matplotlib

    return Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def _criar_pdf(nome_teste: str):
    from fpdf import FPDF

    class _Relatorio(FPDF):
        def header(self) -> None:
            self.set_font(FONTE, "B", 14)
            self.set_text_color(*_rgb(CORES["texto"]))
            self.cell(0, 8, nome_teste, align="L", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(*_rgb(CORES["laranja"]))
            self.set_line_width(0.6)
            self.line(MARGEM, self.get_y() + 1, self.w - MARGEM, self.get_y() + 1)
            self.ln(5)

        def footer(self) -> None:
            self.set_y(-12)
            self.set_font(FONTE, "", 8)
            self.set_text_color(*_rgb(CORES["texto_secundario"]))
            self.cell(0, 6, RODAPE, align="L")
            self.set_x(MARGEM)
            self.cell(0, 6, f"Página {self.page_no()} de {{nb}}", align="R")

    pdf = _Relatorio(format="A4")
    pdf.set_margins(MARGEM, MARGEM, MARGEM)
    pdf.set_auto_page_break(True, margin=18)
    pasta = _caminho_fontes()
    pdf.add_font(FONTE, "", str(pasta / "DejaVuSans.ttf"))
    pdf.add_font(FONTE, "B", str(pasta / "DejaVuSans-Bold.ttf"))
    pdf.add_font(FONTE, "I", str(pasta / "DejaVuSans-Oblique.ttf"))
    pdf.set_title(nome_teste)
    pdf.set_creator("smartETL")
    pdf.alias_nb_pages()
    return pdf


class _Escritor:
    """Escreve os blocos do relatório numa página A4."""

    def __init__(self, pdf):
        self.pdf = pdf
        self.largura = pdf.w - 2 * MARGEM

    def titulo(self, texto: str, nivel: int = 1) -> None:
        pdf = self.pdf
        if pdf.will_page_break(14):
            pdf.add_page()
        pdf.ln(3 if nivel == 1 else 1)
        pdf.set_font(FONTE, "B", 12 if nivel == 1 else 10.5)
        pdf.set_text_color(*_rgb(CORES["texto"] if nivel == 1 else CORES["laranja"]))
        pdf.multi_cell(0, 6, texto, align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def paragrafo(
        self, texto: str, cor: str = "texto", estilo: str = "", tamanho: float = 9.5
    ) -> None:
        pdf = self.pdf
        pdf.set_font(FONTE, estilo, tamanho)
        pdf.set_text_color(*_rgb(CORES.get(cor, cor)))
        pdf.multi_cell(0, tamanho * 0.5, texto, align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1.5)

    def marcador(self, texto: str, simbolo: str, cor_simbolo: str) -> None:
        pdf = self.pdf
        pdf.set_font(FONTE, "", 9)
        pdf.set_text_color(*_rgb(cor_simbolo))
        pdf.cell(5, 4.5, simbolo)
        pdf.set_text_color(*_rgb(CORES["texto_secundario"]))
        pdf.multi_cell(0, 4.5, texto, align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def aviso(self, texto: str) -> None:
        self.marcador(texto, "⚠", CORES["laranja"])

    def nota(self, texto: str) -> None:
        self.marcador(texto, "•", CORES["texto_secundario"])

    def tabela(self, nome: str | None, df: pd.DataFrame) -> None:
        from fpdf.fonts import FontFace

        pdf = self.pdf
        cabecalho = [str(c) for c in df.columns]
        corpo = [[_celula(v) for v in linha] for linha in df.itertuples(index=False, name=None)]
        destaques = set(df.attrs.get("destaques", []))
        tamanho, larguras = self._larguras(cabecalho, corpo, destaques)
        # Tabela que cabe numa página não é partida: começa na página seguinte.
        altura = (len(corpo) + 1) * (tamanho * 0.55 + 2.4) + (7 if nome else 0)
        if pdf.will_page_break(altura) and altura < pdf.h - pdf.t_margin - 40:
            pdf.add_page()
        if nome:
            pdf.set_font(FONTE, "B", 9.5)
            pdf.set_text_color(*_rgb(CORES["texto"]))
            pdf.multi_cell(0, 5, nome, align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        pdf.set_font(FONTE, "", tamanho)
        pdf.set_text_color(*_rgb(CORES["texto"]))
        pdf.set_draw_color(*_rgb(CORES["borda"]))
        pdf.set_line_width(0.2)
        with pdf.table(
            col_widths=larguras,
            width=sum(larguras),
            align="LEFT",
            text_align="LEFT",
            line_height=tamanho * 0.55,
            padding=(1.2, 1.5),
            headings_style=FontFace(
                emphasis="BOLD",
                color=_rgb(CORES["texto_secundario"]),
                fill_color=_rgb(CORES["fundo"]),
            ),
            borders_layout="HORIZONTAL_LINES",
        ) as tabela:
            linha = tabela.row()
            for texto in cabecalho:
                linha.cell(texto)
            negrito = FontFace(emphasis="BOLD")
            for i, valores in enumerate(corpo):
                linha = tabela.row()
                for texto in valores:
                    linha.cell(texto, style=negrito if i in destaques else None)
        pdf.ln(2)
        for coluna, dica in df.attrs.get("dicas", {}).items():
            self.nota(f"{coluna}: {dica}")

    def _larguras(
        self, cabecalho: list[str], corpo: list[list[str]], destaques: set[int]
    ) -> tuple[float, list[float]]:
        """Fonte e larguras naturais das colunas; reduz a fonte (até 6 pt) e, se ainda não
        couber, divide a largura da página na proporção do conteúdo (o texto quebra)."""
        pdf = self.pdf
        tamanho = FONTE_TABELA
        while True:
            pdf.set_font(FONTE, "B", tamanho)
            larguras = [pdf.get_string_width(c) for c in cabecalho]
            for i, valores in enumerate(corpo):
                pdf.set_font(FONTE, "B" if i in destaques else "", tamanho)
                larguras = [
                    max(atual, pdf.get_string_width(texto))
                    for atual, texto in zip(larguras, valores, strict=True)
                ]
            larguras = [w + 4 for w in larguras]
            if sum(larguras) <= self.largura or tamanho <= FONTE_TABELA_MIN:
                break
            tamanho -= 0.5
        total = sum(larguras)
        if total > self.largura:
            larguras = [w * self.largura / total for w in larguras]
        return tamanho, larguras

    def imagem(self, titulo: str, png: bytes, altura_mm: float) -> None:
        pdf = self.pdf
        if pdf.will_page_break(altura_mm + 10):
            pdf.add_page()
        pdf.set_font(FONTE, "B", 9.5)
        pdf.set_text_color(*_rgb(CORES["texto"]))
        pdf.multi_cell(0, 5, titulo, align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.image(io.BytesIO(png), x=MARGEM, w=self.largura, h=altura_mm)
        pdf.ln(4)


def _cabecalho(escritor: _Escritor, contexto: ContextoRelatorio) -> None:
    arquivo = contexto.arquivo or SEM_VALOR
    if contexto.linhas is not None:
        arquivo += f" ({contexto.linhas} linhas)"
    escritor.paragrafo(f"Arquivo: {arquivo}", cor="texto_secundario", tamanho=9)
    escritor.paragrafo(
        f"Gerado em: {contexto.gerado_em:%d/%m/%Y às %H:%M}", cor="texto_secundario", tamanho=9
    )
    if contexto.parametros:
        escritor.titulo("Parâmetros")
        escritor.tabela(None, pd.DataFrame(contexto.parametros, columns=["Campo", "Valor"]))


def _resultado_padrao(escritor: _Escritor, resultado: ResultadoTeste) -> None:
    escritor.titulo("Resultado")
    texto, cor = _decisao(resultado.decisao)
    escritor.paragrafo(texto, cor=cor, estilo="B", tamanho=12)
    escritor.paragrafo(resultado.interpretacao)
    for aviso in resultado.avisos:
        escritor.aviso(aviso)
    for nome, df in resultado.tabelas.items():
        escritor.tabela(nome, df)


def _secoes(escritor: _Escritor, secoes: Sequence[Secao]) -> None:
    for secao in secoes:
        escritor.titulo(secao.titulo, secao.nivel)
        if secao.destaque:
            texto, cor = _decisao(secao.destaque)
            escritor.paragrafo(texto, cor=cor, estilo="B", tamanho=11)
        for texto in secao.textos:
            escritor.paragrafo(texto)
        for aviso in secao.avisos:
            escritor.aviso(aviso)
        for nome, df in secao.tabelas.items():
            escritor.tabela(None if nome == secao.titulo else nome, df)
        for nota in secao.notas:
            escritor.nota(nota)


def gerar_pdf(resultado: ResultadoTeste, contexto: ContextoRelatorio) -> bytes:
    """Bytes do PDF com a análise completa de `resultado`."""
    pdf = _criar_pdf(contexto.nome_teste)
    pdf.add_page()
    escritor = _Escritor(pdf)
    _cabecalho(escritor, contexto)
    if resultado.secoes:
        _secoes(escritor, resultado.secoes)
    else:
        _resultado_padrao(escritor, resultado)
    if resultado.comparacao is not None:
        escritor.titulo(TITULO_COMPARACAO)
        escritor.tabela(None, tabela_comparacao(resultado.comparacao))
        escritor.nota("p-valores da última execução do teste de hipótese.")
    figuras = _figuras(resultado.figuras)
    if figuras:
        escritor.titulo("Visualização")
        altura = escritor.largura * 3.6 / 7.0
        for titulo, figura in figuras:
            escritor.imagem(titulo, figura_png(figura), altura)
    return bytes(pdf.output())


def garantir_extensao(caminho: str) -> str:
    """Acrescenta ".pdf" quando o diálogo devolve o nome sem extensão."""
    return caminho if caminho.lower().endswith(".pdf") else f"{caminho}.pdf"


__all__ = [
    "ContextoRelatorio",
    "descrever_parametros",
    "figura_png",
    "garantir_extensao",
    "gerar_pdf",
    "nome_sugerido",
    "tabela_comparacao",
]
