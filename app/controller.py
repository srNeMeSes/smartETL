"""Liga a interface ao core. Sem lógica estatística e sem Flet."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pandas as pd

from app.state import AppState
from core import registry
from core.base import ErroExecucao, ErroValidacao, ResultadoTeste, TesteBase
from core.io import FormatoNaoSuportado, importar_dados

log = logging.getLogger(__name__)


class Visao(Protocol):
    """O que o controller precisa da interface."""

    async def escolher_arquivo(self) -> str | None: ...
    def exibir_dados(self, df: pd.DataFrame) -> None: ...
    def exibir_sem_arquivo(self) -> None: ...
    def exibir_indisponivel(self, info: registry.TesteInfo) -> None: ...
    def exibir_formulario(self, teste: TesteBase, df: pd.DataFrame) -> None: ...
    def coletar_parametros(self) -> dict: ...
    def exibir_processando(self) -> None: ...
    def exibir_sem_resultado(self) -> None: ...
    def exibir_resultado(self, resultado: ResultadoTeste) -> None: ...
    def notificar(self, mensagem: str, erro: bool = False) -> None: ...


class Controller:
    def __init__(
        self,
        estado: AppState,
        visao: Visao,
        obter_teste: Callable[[str], registry.TesteInfo] = registry.obter,
    ):
        self.estado = estado
        self.visao = visao
        self._obter_teste = obter_teste

    def iniciar(self) -> None:
        """Renderiza o estado inicial (inclusive o teste já marcado na sidebar)."""
        self._renderizar_painel()

    # ---------------- Arquivo ----------------
    async def abrir_arquivo(self) -> None:
        try:
            caminho = await self.visao.escolher_arquivo()
        except Exception:
            log.exception("Falha ao abrir o seletor de arquivos")
            self.visao.notificar("Não foi possível abrir o seletor de arquivos.", erro=True)
            return
        if caminho:
            self.carregar_arquivo(caminho)

    def carregar_arquivo(self, caminho: str) -> bool:
        nome = Path(caminho).name
        try:
            df = importar_dados(caminho)
        except FormatoNaoSuportado as erro:
            self.visao.notificar(str(erro), erro=True)
            return False
        except pd.errors.EmptyDataError:
            self.visao.notificar(f"O arquivo '{nome}' está vazio.", erro=True)
            return False
        except Exception:
            log.exception("Falha ao ler %s", caminho)
            self.visao.notificar(
                f"Não foi possível ler o arquivo '{nome}'. Verifique se ele não está vazio "
                "ou corrompido.",
                erro=True,
            )
            return False
        if len(df.columns) == 0:
            self.visao.notificar(f"O arquivo '{nome}' não contém dados.", erro=True)
            return False

        log.info("Arquivo carregado: %s (%d linhas, %d colunas)", caminho, *df.shape)
        self.estado.df = df
        self.estado.caminho = caminho
        self.estado.ultimo_resultado = None
        self.visao.exibir_dados(df)
        self._renderizar_painel()
        self.visao.notificar(f"{nome}: {df.shape[0]} linhas e {df.shape[1]} colunas carregadas.")
        return True

    # ---------------- Teste ----------------
    def selecionar_teste(self, teste_id: str) -> None:
        self._obter_teste(teste_id)  # valida o id
        self.estado.teste_id = teste_id
        self.estado.params = {}
        self.estado.ultimo_resultado = None
        self._renderizar_painel()

    def _renderizar_painel(self) -> None:
        info = self._obter_teste(self.estado.teste_id)
        if self.estado.df is None:
            self.visao.exibir_sem_arquivo()
        elif not info.disponivel:
            self.visao.exibir_indisponivel(info)
        else:
            self.visao.exibir_formulario(info.criar(), self.estado.df)

    def executar(self) -> ResultadoTeste | None:
        info = self._obter_teste(self.estado.teste_id)
        df = self.estado.df
        if df is None:
            self.visao.notificar("Carregue um arquivo para começar.", erro=True)
            return None
        if not info.disponivel:
            self.visao.notificar(f"O {info.nome} ainda não está disponível nesta versão.")
            return None

        teste = info.criar()
        processando = False
        try:
            params = self.visao.coletar_parametros()
            erros = teste.validar(df, params)
            if erros:
                raise ErroValidacao(erros)
            self.visao.exibir_processando()
            processando = True
            resultado = teste.executar(df, params)
        except ErroValidacao as erro:
            self.visao.notificar("\n".join(erro.mensagens), erro=True)
            return None
        except ErroExecucao as erro:
            self._falhou(processando)
            self.visao.notificar(str(erro), erro=True)
            return None
        except Exception:
            log.exception("Erro inesperado ao executar %s", info.id)
            self._falhou(processando)
            self.visao.notificar(
                f"Erro inesperado ao executar o {info.nome}. Detalhes no log.", erro=True
            )
            return None

        self.estado.params = params
        self.estado.ultimo_resultado = resultado
        self.visao.exibir_resultado(resultado)
        return resultado

    def _falhou(self, processando: bool) -> None:
        if processando:
            self.visao.exibir_sem_resultado()
