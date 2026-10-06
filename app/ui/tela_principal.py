"""Montagem da tela principal e implementação da `Visao` usada pelo controller."""

from typing import TYPE_CHECKING

import flet as ft
import pandas as pd

from app.ui import tema
from app.ui.componentes.card_comparacao import CardComparacaoTestes
from app.ui.helpers import pad
from app.ui.painel_abas import ABA_ANALISE, PainelAbas, mensagem, processando
from app.ui.painel_parametros import PainelParametros
from app.ui.sidebar import Sidebar
from app.ui.tabela_dados import TabelaDados
from core import registry
from core.base import ResultadoTeste, TesteBase

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
        self._controller.executar()

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

    def exibir_formulario(self, teste: TesteBase, df: pd.DataFrame) -> None:
        self._teste = teste
        self.formulario = PainelParametros(teste, df)
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
        resumo = self._resumo(resultado)
        controles: list[ft.Control] = [resumo]
        if resultado.comparacao is not None:
            self.card_analise = CardComparacaoTestes.de_comparacao(resultado.comparacao)
            controles.append(self.card_analise)
            if self.formulario is not None and self.formulario.card is not None:
                self.formulario.card.aplicar(resultado.comparacao, atualizar_pagina=False)
        self.painel.definir_analise(ft.Column(controls=controles, expand=True, spacing=12))
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
        linhas = [
            ft.Text(resultado.decisao, size=16, weight=ft.FontWeight.BOLD, color=tema.TEXTO),
            ft.Text(resultado.interpretacao, size=14, color=tema.TEXTO),
        ]
        linhas += [
            ft.Text(f"Aviso: {aviso}", size=13, color=tema.TEXTO_SECUNDARIO)
            for aviso in resultado.avisos
        ]
        return ft.Column(controls=linhas, spacing=6)

    @staticmethod
    def _visualizacao(resultado: ResultadoTeste) -> ft.Control:
        if not resultado.figuras:
            return mensagem("Nenhum gráfico gerado para este teste.", titulo="Visualização")
        return ft.Column(
            controls=[
                ft.Column(
                    [
                        ft.Text(fig.titulo, size=14, weight=ft.FontWeight.W_600, color=tema.TEXTO),
                        ft.Image(src=fig.png),
                    ],
                    spacing=6,
                )
                for fig in resultado.figuras
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _atualizar(self) -> None:
        self.page.update()
