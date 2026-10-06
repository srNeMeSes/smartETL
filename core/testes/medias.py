"""Testes de médias."""

import pandas as pd

from core.base import (
    ComparacaoPValores,
    ParametroSpec,
    ResultadoTeste,
    TesteBase,
    TesteNaoImplementado,
)


class TesteT1Amostra(TesteBase):
    """Teste t de uma amostra. Na Fase 1 só define o formulário; o cálculo vem na Fase 3."""

    id = "teste_t_1am"
    nome = "Teste t (uma amostra)"
    grupo = "Médias"

    def parametros(self) -> list[ParametroSpec]:
        return [
            ParametroSpec("coluna", "Variável", "coluna_numerica"),
            ParametroSpec("mu0", "Média Hipotética", "numero"),
            ParametroSpec("alfa", "Nível de significância (α)", "alfa", padrao=0.05),
        ]

    def validar(self, df: pd.DataFrame, params: dict) -> list[str]:
        raise TesteNaoImplementado(self.nome)

    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste:
        raise TesteNaoImplementado(self.nome)

    def comparacao_inicial(self) -> ComparacaoPValores:
        return ComparacaoPValores(
            titulo_esquerda="t Student",
            titulo_direita="Wilcoxon",
            hipoteses=["μ ≠ μ₀", "μ > μ₀", "μ < μ₀"],
            linhas=[(None, None)] * 3,
        )
