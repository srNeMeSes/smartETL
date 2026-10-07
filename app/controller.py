"""Liga a interface ao core. Sem lógica estatística e sem Flet."""

import asyncio
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from app.state import AppState
from core import registry
from core.base import ErroExecucao, ErroValidacao, ResultadoTeste, TesteBase
from core.io import DadosCarregados, ErroLeitura, carregar_dados

log = logging.getLogger(__name__)


class Visao(Protocol):
    """O que o controller precisa da interface."""

    async def escolher_arquivo(self) -> str | None: ...
    def exibir_dados(self, df: pd.DataFrame) -> None: ...
    def exibir_sem_arquivo(self) -> None: ...
    def exibir_indisponivel(self, info: registry.TesteInfo) -> None: ...
    def exibir_formulario(self, teste: TesteBase, dados: DadosCarregados) -> None: ...
    def coletar_parametros(self) -> dict: ...
    def exibir_processando(self) -> None: ...
    def exibir_sem_resultado(self) -> None: ...
    def exibir_resultado(self, resultado: ResultadoTeste) -> None: ...
    def exibir_previsao(self, previsao: Any, variavel: str | None) -> None: ...
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
        self._executando = threading.Lock()  # executar() roda fora da thread da UI

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
            # Fora da thread da UI: arquivos grandes (XLSX, sobretudo) não congelam a janela.
            self.visao.notificar(f"Lendo '{Path(caminho).name}'...")
            dados = await asyncio.to_thread(self._ler, caminho)
            if dados is not None:
                self._aplicar_dados(dados)

    def carregar_arquivo(self, caminho: str) -> bool:
        """Lê e aplica o arquivo (na thread atual). Devolve se a leitura deu certo."""
        dados = self._ler(caminho)
        if dados is None:
            return False
        self._aplicar_dados(dados)
        return True

    def _ler(self, caminho: str) -> DadosCarregados | None:
        nome = Path(caminho).name
        try:
            return carregar_dados(caminho)
        except ErroLeitura as erro:
            self.visao.notificar(str(erro), erro=True)
        except Exception:
            log.exception("Falha ao ler %s", caminho)
            self.visao.notificar(
                f"Não foi possível ler o arquivo '{nome}'. Verifique se ele não está vazio "
                "ou corrompido.",
                erro=True,
            )
        return None

    def _aplicar_dados(self, dados: DadosCarregados) -> None:
        linhas, colunas = dados.df.shape
        log.info(
            "Arquivo carregado: %s (%d linhas, %d colunas; encoding=%s, separador=%r, decimal=%r)",
            dados.caminho,
            linhas,
            colunas,
            dados.encoding,
            dados.separador,
            dados.decimal,
        )
        for aviso in dados.avisos:
            log.warning("%s: %s", dados.nome_arquivo, aviso)
        self.estado.dados = dados
        self.estado.ultimo_resultado = None
        self.visao.exibir_dados(dados.df)
        self._renderizar_painel()
        self.visao.notificar(self._resumo_carga(dados))

    @staticmethod
    def _resumo_carga(dados: DadosCarregados) -> str:
        linhas, colunas = dados.df.shape
        resumo = f"{dados.nome_arquivo}: {linhas} linhas e {colunas} colunas carregadas."
        return "\n".join([resumo, *(f"Atenção: {aviso}" for aviso in dados.avisos)])

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
            self.visao.exibir_formulario(info.criar(), self.estado.dados)

    def executar(self) -> ResultadoTeste | None:
        """Executa o teste selecionado; ignora cliques enquanto uma execução está em andamento."""
        if not self._executando.acquire(blocking=False):
            return None
        try:
            return self._executar()
        finally:
            self._executando.release()

    def _executar(self) -> ResultadoTeste | None:
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

    # ---------------- Simulação ----------------
    def simular(self, valores: dict, variavel: str | None = None) -> Any:
        """Previsão do último resultado para `valores` (aba Simulação); `variavel` é o campo
        alterado, que a interface destaca na equação. Sem simulação disponível, devolve None."""
        simulador = getattr(self.estado.ultimo_resultado, "simulacao", None)
        if simulador is None:
            return None
        try:
            previsao = simulador.prever(valores)
        except (KeyError, TypeError, ValueError):
            log.exception("Valores inválidos na simulação: %r", valores)
            return None
        self.visao.exibir_previsao(previsao, variavel)
        return previsao

    def _falhou(self, processando: bool) -> None:
        if processando:
            self.visao.exibir_sem_resultado()
