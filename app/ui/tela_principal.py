"""Montagem da tela principal e implementação da `Visao` usada pelo controller."""

from typing import TYPE_CHECKING

import flet as ft
import pandas as pd

from app.ui import tema
from app.ui.componentes.card_comparacao import CardComparacaoTestes
from app.ui.graficos import desenhar_figura
from app.ui.helpers import border_all, pad
from app.ui.painel_abas import ABA_ANALISE, PainelAbas, mensagem, processando
from app.ui.painel_parametros import PainelParametros
from app.ui.sidebar import Sidebar
from app.ui.tabela_dados import TabelaDados, formatar_celula
from core import registry
from core.base import ResultadoTeste, TesteBase
from core.io import DadosCarregados

if TYPE_CHECKING:
    from app.controller import Controller

SEM_ARQUIVO = "Carregue um arquivo para começar."
SEM_EXECUCAO = "Configure os parâmetros e clique em Executar teste."
EXTENSOES = ["xlsx", "csv"]


class TelaPrincipal:
    """Sidebar + cabeçalho + prévia dos dados + painel de abas."""

    def __init__(self, page: ft.Page):
        self.page = page
        self._controller: Controller | None = None
        self._teste: TesteBase | None = None
        self.formulario: PainelParametros | None = None
        self.card_analise: CardComparacaoTestes | None = None

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

    def _ao_clicar_executar(self, e: ft.Event) -> None:
        # Fora da thread da UI: "Processando..." aparece e a janela não congela.
        self.page.run_thread(self._controller.executar)

    # ---------------- Visao ----------------
    async def escolher_arquivo(self) -> str | None:
        arquivos = await self.file_picker.pick_files(
            allow_multiple=False, allowed_extensions=EXTENSOES
        )
        return arquivos[0].path if arquivos else None

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
        self.painel.definir_analise(analise)
        self.painel.definir_visualizacao(self._visualizacao(resultado))
        self.painel.ir_para(ABA_ANALISE)
        self._atualizar()

    def notificar(self, mensagem: str, erro: bool = False) -> None:
        self.page.show_dialog(
            ft.SnackBar(
                content=ft.Text(mensagem, color=tema.TEXTO_SOBRE_DESTAQUE),
                bgcolor=tema.NOTIFICACAO_ERRO if erro else tema.NOTIFICACAO,
                show_close_icon=True,
            )
        )

    # ---------------- Internos ----------------
    def _limpar_teste(self) -> None:
        self._teste = None
        self.formulario = None
        self.card_analise = None

    def _definir_abas_sem_resultado(self) -> None:
        """Análise com o card zerado (2ª instância) e Visualização com estado vazio."""
        comparacao = self._teste.comparacao_inicial() if self._teste else None
        analise: list[ft.Control] = [mensagem(SEM_EXECUCAO)]
        self.card_analise = None
        if comparacao is not None:
            self.card_analise = CardComparacaoTestes.de_comparacao(comparacao)
            analise.append(self.card_analise)
        self.painel.definir_analise(ft.Column(controls=analise, expand=True, spacing=12))
        self.painel.definir_visualizacao(mensagem(SEM_EXECUCAO, titulo="Visualização"))

    @staticmethod
    def _resumo(resultado: ResultadoTeste) -> ft.Column:
        decisao = resultado.decisao.replace("H0", "H₀")
        linhas: list[ft.Control] = [
            ft.Text(decisao, size=16, weight=ft.FontWeight.BOLD, color=tema.TEXTO),
            ft.Text(resultado.interpretacao, size=14, color=tema.TEXTO),
        ]
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
    def _tabela(nome: str, df: pd.DataFrame) -> ft.Column:
        def celula(valor: object) -> ft.DataCell:
            return ft.DataCell(ft.Text(formatar_celula(valor), size=13, color=tema.TEXTO))

        tabela = ft.DataTable(
            columns=[
                ft.DataColumn(
                    ft.Text(
                        str(c), size=13, weight=ft.FontWeight.W_600, color=tema.TEXTO_SECUNDARIO
                    )
                )
                for c in df.columns
            ],
            rows=[
                ft.DataRow(cells=[celula(v) for v in linha])
                for linha in df.itertuples(index=False, name=None)
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
        titulo = ft.Text(nome, size=14, weight=ft.FontWeight.W_600, color=tema.TEXTO)
        return ft.Column([titulo, tabela], spacing=8)

    @staticmethod
    def _visualizacao(resultado: ResultadoTeste) -> ft.Control:
        if not resultado.figuras:
            return mensagem("Nenhum gráfico gerado para este teste.", titulo="Visualização")
        return ft.Column(
            controls=[desenhar_figura(fig) for fig in resultado.figuras],
            spacing=24,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _atualizar(self) -> None:
        self.page.update()
