"""Montagem da tela principal e implementação da `Visao` usada pelo controller."""

from typing import TYPE_CHECKING

import flet as ft
import pandas as pd

from app.ui import tema
from app.ui.componentes import campos
from app.ui.componentes.card_comparacao import CardComparacaoTestes
from app.ui.graficos import desenhar_figura
from app.ui.helpers import border_all, esta_na_pagina, pad
from app.ui.painel_abas import ABA_ANALISE, PainelAbas, mensagem, processando
from app.ui.painel_parametros import PainelParametros
from app.ui.painel_simulacao import PainelSimulacao
from app.ui.sidebar import Sidebar
from app.ui.tabela_dados import TabelaDados, formatar_celula
from core import registry
from core.base import Figura, GrupoFiguras, ResultadoTeste, Secao, TesteBase
from core.interpretacao import NAO_REJEITA_H0, REJEITA_H0
from core.io import DadosCarregados

if TYPE_CHECKING:
    from app.controller import Controller


def cor_da_decisao(texto: str) -> str:
    """Vermelho suave para "Rejeita H₀", verde suave para "Não rejeita H₀"; senão, o texto."""
    normalizado = texto.replace("H₀", "H0")
    if normalizado == NAO_REJEITA_H0:
        return tema.DECISAO_NAO_REJEITA
    if normalizado == REJEITA_H0:
        return tema.DECISAO_REJEITA
    return tema.TEXTO


SEM_ARQUIVO = "Carregue um arquivo para começar."
SEM_EXECUCAO = "Configure os parâmetros e clique em Executar teste."
EXTENSOES = ["xlsx", "csv"]
ROTULO_EXPORTAR = "Exportar PDF"


class TelaPrincipal:
    """Sidebar + cabeçalho + prévia dos dados + painel de abas."""

    def __init__(self, page: ft.Page):
        self.page = page
        self._controller: Controller | None = None
        self._teste: TesteBase | None = None
        self.formulario: PainelParametros | None = None
        self.card_analise: CardComparacaoTestes | None = None
        self.simulacao: PainelSimulacao | None = None
        self.botao_pdf: ft.Button | None = None

        self.file_picker = ft.FilePicker()
        page.services.append(self.file_picker)

        self.sidebar = Sidebar(
            on_arquivo=self._ao_clicar_arquivo,
            on_selecionar=self._ao_selecionar_teste,
            on_executar=self._ao_clicar_executar,
        )
        self.tabela = TabelaDados()
        self.painel = PainelAbas()
        self.raiz = ft.Row([self.sidebar, self._area_principal()], expand=True, spacing=0)

    def conectar(self, controller: "Controller") -> None:
        self._controller = controller

    def _area_principal(self) -> ft.Container:
        cabecalho = ft.Column(
            [
                ft.Text(
                    "Processamento de dados", size=24, weight=ft.FontWeight.BOLD, color=tema.TEXTO
                ),
                ft.Text(
                    "Testes de Hipótese  |  paramétricos e não paramétricos",
                    size=13,
                    color=tema.TEXTO_SECUNDARIO,
                ),
            ],
            spacing=4,
        )
        return ft.Container(
            content=ft.Column(
                controls=[
                    cabecalho,
                    ft.Divider(color=tema.TRANSPARENTE, height=7),
                    self.tabela,
                    self.painel,
                ],
                expand=True,
            ),
            expand=True,
            padding=pad(left=32, right=32, top=28, bottom=28),
            bgcolor=tema.FUNDO,
        )

    # ---------------- Callbacks (só repassam ao controller) ----------------
    async def _ao_clicar_arquivo(self, e: ft.Event) -> None:
        await self._controller.abrir_arquivo()

    def _ao_selecionar_teste(self, teste_id: str) -> None:
        self._controller.selecionar_teste(teste_id)

    async def _ao_clicar_exportar(self, e: ft.Event) -> None:
        await self._controller.exportar_pdf()

    def _ao_clicar_executar(self, e: ft.Event) -> None:
        # Fora da thread da UI: "Processando..." aparece e a janela não congela.
        self.page.run_thread(self._controller.executar)

    # ---------------- Visao ----------------
    async def escolher_arquivo(self) -> str | None:
        arquivos = await self.file_picker.pick_files(
            allow_multiple=False, allowed_extensions=EXTENSOES
        )
        return arquivos[0].path if arquivos else None

    async def escolher_destino_pdf(self, nome_sugerido: str) -> str | None:
        return await self.file_picker.save_file(
            dialog_title="Exportar a análise em PDF",
            file_name=nome_sugerido,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["pdf"],
        )

    def exibir_dados(self, df: pd.DataFrame) -> None:
        self.tabela.mostrar(df)
        self._atualizar()

    def exibir_sem_arquivo(self) -> None:
        self._limpar_teste()
        for definir in (
            self.painel.definir_parametros,
            self.painel.definir_analise,
            self.painel.definir_visualizacao,
        ):
            definir(mensagem(SEM_ARQUIVO))
        self._atualizar()

    def exibir_indisponivel(self, info: registry.TesteInfo) -> None:
        self._limpar_teste()
        texto = f"O {info.nome} ainda não está disponível nesta versão."
        self.painel.definir_parametros(mensagem(texto, titulo=info.nome))
        self.painel.definir_analise(mensagem(texto))
        self.painel.definir_visualizacao(mensagem(texto))
        self._atualizar()

    def exibir_formulario(self, teste: TesteBase, dados: DadosCarregados) -> None:
        self._teste = teste
        self.formulario = PainelParametros(teste, dados.df, dados.perfis)
        self.painel.definir_parametros(self.formulario)
        self._definir_abas_sem_resultado()
        self._atualizar()

    def coletar_parametros(self) -> dict:
        return self.formulario.coletar_valores()

    def exibir_processando(self) -> None:
        self.painel.definir_analise(processando())
        self.painel.definir_visualizacao(processando())
        self._atualizar()

    def exibir_sem_resultado(self) -> None:
        self._definir_abas_sem_resultado()
        self._atualizar()

    def exibir_resultado(self, resultado: ResultadoTeste) -> None:
        if resultado.secoes:
            self.card_analise = None
            conteudo: ft.Control = self._analise_por_secoes(resultado.secoes)
        else:
            conteudo = self._analise_padrao(resultado)
        self.painel.definir_analise(conteudo)
        self.painel.definir_acao(self._botao_exportar())
        self.painel.definir_visualizacao(self._visualizacao(resultado))
        self.simulacao = None
        if resultado.simulacao is not None:
            self.simulacao = PainelSimulacao(resultado.simulacao, self._ao_alterar_simulacao)
        quantas = len(self.painel.rotulos)
        self.painel.definir_simulacao(self.simulacao)
        if len(self.painel.rotulos) != quantas:
            # Mudar o número de abas reinicia a aba selecionada no Flet: primeiro a nova
            # estrutura, depois a seleção.
            self._atualizar()
        self.painel.ir_para(ABA_ANALISE)
        self._atualizar()
        if self.simulacao is not None:
            self.simulacao.iniciar()  # previsão inicial (médias e níveis de referência)

    def exibir_previsao(self, previsao: object, variavel: str | None) -> None:
        if self.simulacao is not None:
            self.simulacao.mostrar(previsao, variavel)

    def _ao_alterar_simulacao(self, valores: dict, variavel: str | None) -> None:
        self._controller.simular(valores, variavel)

    def _analise_padrao(self, resultado: ResultadoTeste) -> ft.Control:
        detalhes: list[ft.Control] = [self._resumo(resultado)]
        detalhes += [self._tabela(nome, df) for nome, df in resultado.tabelas.items()]
        if resultado.comparacao is None:
            self.card_analise = None
            analise: ft.Control = ft.Column(
                detalhes, spacing=16, scroll=ft.ScrollMode.AUTO, expand=True
            )
        else:
            self.card_analise = CardComparacaoTestes.de_comparacao(resultado.comparacao)
            if self.formulario is not None and self.formulario.card is not None:
                self.formulario.card.aplicar(resultado.comparacao, atualizar_pagina=False)
            analise = ft.Row(
                [
                    ft.Column(detalhes, spacing=16, scroll=ft.ScrollMode.AUTO, width=440),
                    ft.Column(width=12),
                    self.card_analise,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            )
        return analise

    def notificar(self, mensagem: str, erro: bool = False) -> None:
        self.page.show_dialog(
            ft.SnackBar(
                content=ft.Text(mensagem, color=tema.TEXTO_SOBRE_DESTAQUE),
                bgcolor=tema.NOTIFICACAO_ERRO if erro else tema.NOTIFICACAO,
                show_close_icon=True,
            )
        )

    # ---------------- Internos ----------------
    def _botao_exportar(self) -> ft.Button:
        """Botão "Exportar PDF" na linha das abas (só existe depois de uma execução)."""
        self.botao_pdf = ft.Button(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.PICTURE_AS_PDF_OUTLINED, size=18, color=tema.LARANJA),
                    ft.Text(
                        ROTULO_EXPORTAR, size=13, color=tema.LARANJA, weight=ft.FontWeight.W_600
                    ),
                ],
                spacing=8,
                tight=True,
            ),
            on_click=self._ao_clicar_exportar,
            bgcolor=tema.LARANJA_SUAVE,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=tema.RAIO_PEQUENO),
                padding=pad(horizontal=14, vertical=8),
                elevation=0,
            ),
        )
        return self.botao_pdf

    def _limpar_teste(self) -> None:
        self._teste = None
        self.botao_pdf = None
        self.painel.definir_acao(None)
        self.formulario = None
        self.card_analise = None
        self.simulacao = None
        self.painel.definir_simulacao(None)

    def _definir_abas_sem_resultado(self) -> None:
        """Análise com o card zerado (2ª instância) e Visualização com estado vazio."""
        self.botao_pdf = None
        self.painel.definir_acao(None)
        comparacao = self._teste.comparacao_inicial() if self._teste else None
        analise: list[ft.Control] = [mensagem(SEM_EXECUCAO)]
        self.card_analise = None
        if comparacao is not None:
            self.card_analise = CardComparacaoTestes.de_comparacao(comparacao)
            analise.append(self.card_analise)
        self.painel.definir_analise(ft.Column(controls=analise, expand=True, spacing=12))
        self.painel.definir_visualizacao(mensagem(SEM_EXECUCAO, titulo="Visualização"))
        self.simulacao = None
        self.painel.definir_simulacao(None)

    @staticmethod
    def _resumo(resultado: ResultadoTeste) -> ft.Column:
        decisao = resultado.decisao.replace("H0", "H₀")
        linhas: list[ft.Control] = []
        if decisao:  # sem decisão em técnicas que não são testes de hipótese (ex.: IV)
            linhas.append(
                ft.Text(decisao, size=16, weight=ft.FontWeight.BOLD, color=cor_da_decisao(decisao))
            )
        linhas.append(ft.Text(resultado.interpretacao, size=14, color=tema.TEXTO))
        linhas += [
            ft.Row(
                [
                    ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, size=16, color=tema.LARANJA),
                    ft.Text(aviso, size=13, color=tema.TEXTO_SECUNDARIO, expand=True),
                ],
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.START,
            )
            for aviso in resultado.avisos
        ]
        return ft.Column(controls=linhas, spacing=6)

    @staticmethod
    def _linha_icone(texto: str, icone: str, cor: str) -> ft.Row:
        return ft.Row(
            [
                ft.Icon(icone, size=16, color=cor),
                ft.Text(texto, size=13, color=tema.TEXTO_SECUNDARIO, expand=True),
            ],
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )

    def _analise_por_secoes(self, secoes: list[Secao]) -> ft.Column:
        """Análise na ordem definida pelo teste (ex.: regressão: pressupostos primeiro)."""
        controles: list[ft.Control] = []
        for secao in secoes:
            principal = secao.nivel == 1
            controles.append(
                ft.Text(
                    secao.titulo,
                    size=17 if principal else 15,
                    weight=ft.FontWeight.BOLD if principal else ft.FontWeight.W_600,
                    color=tema.TEXTO if principal else tema.LARANJA,
                )
            )
            if secao.destaque:
                controles.append(
                    ft.Text(
                        secao.destaque,
                        size=16,
                        weight=ft.FontWeight.BOLD,
                        color=cor_da_decisao(secao.destaque),
                    )
                )
            controles += [ft.Text(texto, size=14, color=tema.TEXTO) for texto in secao.textos]
            controles += [
                self._linha_icone(a, ft.Icons.WARNING_AMBER_ROUNDED, tema.LARANJA)
                for a in secao.avisos
            ]
            for nome, df in secao.tabelas.items():
                # Título da tabela só quando acrescenta algo ao título da seção.
                controles.append(self._tabela(None if nome == secao.titulo else nome, df))
            controles += [
                self._linha_icone(n, ft.Icons.INFO_OUTLINE, tema.TEXTO_SECUNDARIO)
                for n in secao.notas
            ]
        return ft.Column(controles, spacing=12, scroll=ft.ScrollMode.AUTO, expand=True)

    @staticmethod
    def _tabela(nome: str | None, df: pd.DataFrame) -> ft.Column:
        """DataTable; `df.attrs` pode trazer "dicas" (tooltip por coluna, marcada com ⓘ) e
        "destaques" (posições das linhas em negrito)."""
        dicas: dict[str, str] = df.attrs.get("dicas", {})
        destaques = set(df.attrs.get("destaques", []))

        def celula(valor: object, negrito: bool) -> ft.DataCell:
            peso = ft.FontWeight.BOLD if negrito else None
            return ft.DataCell(
                ft.Text(formatar_celula(valor), size=13, color=tema.TEXTO, weight=peso)
            )

        tabela = ft.DataTable(
            columns=[
                ft.DataColumn(
                    ft.Text(
                        str(c) + (" ⓘ" if str(c) in dicas else ""),
                        size=13,
                        weight=ft.FontWeight.W_600,
                        color=tema.TEXTO_SECUNDARIO,
                    ),
                    tooltip=dicas.get(str(c)),
                )
                for c in df.columns
            ],
            rows=[
                ft.DataRow(cells=[celula(v, i in destaques) for v in linha])
                for i, linha in enumerate(df.itertuples(index=False, name=None))
            ],
            heading_row_color=tema.FUNDO,
            heading_row_height=36,
            data_row_min_height=32,
            data_row_max_height=32,
            column_spacing=28,
            border=border_all(1, tema.BORDA),
            border_radius=tema.RAIO_PEQUENO,
            horizontal_lines=ft.BorderSide(1, tema.BORDA),
        )
        # Rolagem horizontal: a tabela mantém a largura natural em vez de quebrar o texto das
        # células (cortado pela altura fixa da linha) quando a coluna é estreita (ao lado do card).
        controles: list[ft.Control] = [ft.Row([tabela], scroll=ft.ScrollMode.AUTO)]
        if nome:
            controles.insert(
                0, ft.Text(nome, size=14, weight=ft.FontWeight.W_600, color=tema.TEXTO)
            )
        return ft.Column(controles, spacing=8)

    @staticmethod
    def _grupo_figuras(grupo: GrupoFiguras) -> ft.Column:
        """Lista suspensa acima do gráfico; trocar a opção só troca a figura exibida."""
        area = ft.Container(content=desenhar_figura(grupo.opcoes[grupo.padrao]))
        lista = campos.dropdown_campo(grupo.rotulo, list(grupo.opcoes), valor=grupo.padrao)

        def trocar(_evento=None) -> None:
            area.content = desenhar_figura(grupo.opcoes[lista.value])
            if esta_na_pagina(area):
                area.update()

        lista.on_select = trocar
        return ft.Column([lista, area], spacing=10)

    def _visualizacao(self, resultado: ResultadoTeste) -> ft.Control:
        if not resultado.figuras:
            return mensagem("Nenhum gráfico gerado para este teste.", titulo="Visualização")

        def desenhar(item: Figura | GrupoFiguras) -> ft.Control:
            if isinstance(item, GrupoFiguras):
                return self._grupo_figuras(item)
            return desenhar_figura(item)

        return ft.Column(
            controls=[desenhar(item) for item in resultado.figuras],
            spacing=24,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _atualizar(self) -> None:
        self.page.update()
