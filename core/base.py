"""Contratos dos testes de hipótese: parâmetros, resultado e classe base."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

TipoParametro = Literal[
    "coluna_numerica",
    "coluna_categorica",
    "coluna_binaria",
    "multi_coluna",
    "numero",
    "alfa",
    "opcao",
    "booleano",
]

TIPOS_COLUNA: frozenset[str] = frozenset(
    {"coluna_numerica", "coluna_categorica", "coluna_binaria", "multi_coluna"}
)

ALFAS = ("0.01", "0.05", "0.10")


@dataclass(frozen=True)
class ParametroSpec:
    """Descreve um campo do formulário de um teste."""

    nome: str
    rotulo: str  # texto exibido na UI (pt-BR)
    tipo: TipoParametro
    padrao: Any = None
    opcoes: list[str] | None = None
    obrigatorio: bool = True


@dataclass
class ComparacaoPValores:
    """Alimenta o card que compara o teste paramétrico com o não paramétrico."""

    titulo_esquerda: str  # ex.: "t Student"
    titulo_direita: str  # ex.: "Wilcoxon"
    hipoteses: list[str]  # texto de H1 de cada linha, ex.: "μ ≠ μ₀"
    linhas: list[tuple[float | None, float | None]]  # (p_param, p_nao_param); None = não calculado


@dataclass
class Figura:
    """Gráfico para a aba Visualização (formato definitivo decidido na Fase 3)."""

    titulo: str
    png: bytes


@dataclass
class ResultadoTeste:
    teste_id: str
    estatisticas: dict[str, float]  # ex.: {"t": 2.31, "gl": 29}
    p_valor: float | None
    alfa: float
    decisao: str  # "Rejeita H0" / "Não rejeita H0"
    interpretacao: str  # texto em pt-BR
    tabelas: dict[str, pd.DataFrame] = field(default_factory=dict)
    figuras: list[Figura] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    comparacao: ComparacaoPValores | None = None


class ErroValidacao(Exception):
    """Entrada inválida; as mensagens são exibidas ao usuário."""

    def __init__(self, mensagens: str | list[str]):
        self.mensagens = [mensagens] if isinstance(mensagens, str) else list(mensagens)
        super().__init__("\n".join(self.mensagens))


class ErroExecucao(Exception):
    """Falha ao executar um teste; a mensagem é exibida ao usuário."""


class TesteNaoImplementado(ErroExecucao):
    """O cálculo do teste ainda não foi implementado."""

    def __init__(self, nome: str):
        super().__init__(f"O cálculo do {nome} ainda não foi implementado.")


class TesteBase(ABC):
    """Classe base de todos os testes. Não conhece Flet."""

    id: str
    nome: str
    grupo: str

    @abstractmethod
    def parametros(self) -> list[ParametroSpec]: ...

    @abstractmethod
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        """Lista de erros (vazia quando os parâmetros são válidos)."""

    @abstractmethod
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste: ...

    def comparacao_inicial(self) -> ComparacaoPValores | None:
        """Estrutura do card antes da execução (p-valores None), ou None se não houver card."""
        return None
