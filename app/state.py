"""Estado da aplicação (sem Flet)."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core import registry
from core.base import ResultadoTeste
from core.io import DadosCarregados
from core.tipos import PerfilColuna


@dataclass
class AppState:
    dados: DadosCarregados | None = None  # o arquivo é lido uma vez e guardado aqui
    teste_id: str = field(default_factory=lambda: registry.primeiro().id)
    params: dict[str, Any] = field(default_factory=dict)
    ultimo_resultado: ResultadoTeste | None = None

    @property
    def df(self) -> pd.DataFrame | None:
        return self.dados.df if self.dados else None

    @property
    def perfis(self) -> dict[str, PerfilColuna]:
        return self.dados.perfis if self.dados else {}

    @property
    def caminho(self) -> str | None:
        return self.dados.caminho if self.dados else None

    @property
    def nome_arquivo(self) -> str | None:
        return self.dados.nome_arquivo if self.dados else None
