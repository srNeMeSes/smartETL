"""Estado da aplicação (sem Flet)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from core import registry
from core.base import ResultadoTeste


@dataclass
class AppState:
    df: pd.DataFrame | None = None
    caminho: str | None = None
    teste_id: str = field(default_factory=lambda: registry.primeiro().id)
    params: dict[str, Any] = field(default_factory=dict)
    ultimo_resultado: ResultadoTeste | None = None

    @property
    def nome_arquivo(self) -> str | None:
        return Path(self.caminho).name if self.caminho else None
