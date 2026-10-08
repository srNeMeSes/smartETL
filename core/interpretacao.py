"""Decisão e textos de interpretação em pt-BR a partir do p-valor e de α (sem Flet)."""

import math

REJEITA_H0 = "Rejeita H0"
NAO_REJEITA_H0 = "Não rejeita H0"
SEM_VALOR = "—"


def formatar_numero(valor: float | None, casas: int = 4) -> str:
    """Número com vírgula decimal e ponto de milhar ("1.234,5000"); "—" se ausente."""
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return SEM_VALOR
    if math.isinf(valor):
        return "+∞" if valor > 0 else "-∞"
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def formatar_p_valor(p: float | None) -> str:
    """p-valor com vírgula decimal; "< 0,001" abaixo de 0,001; "—" quando não calculado."""
    if p is None or math.isnan(p):
        return SEM_VALOR
    if p < 0.001:
        return "< 0,001"
    return f"{p:.3f}".replace(".", ",")


def formatar_alfa(alfa: float) -> str:
    return f"{alfa:.2f}".replace(".", ",")


def p_indefinido(p_valor: float | None) -> bool:
    """p-valor que não pôde ser calculado (None ou NaN)."""
    return p_valor is None or math.isnan(p_valor)


def decidir(p_valor: float, alfa: float) -> str:
    """Rejeita H0 quando p ≤ α (convenção de livro-texto).

    p indefinido (NaN, None) nunca rejeita: sem evidência calculada, a decisão conservadora é
    não rejeitar — e `interpretar` diz que o p-valor não pôde ser calculado.
    """
    if p_indefinido(p_valor):
        return NAO_REJEITA_H0
    return REJEITA_H0 if p_valor <= alfa else NAO_REJEITA_H0


def interpretar(
    p_valor: float,
    alfa: float,
    h0: str,
    h1: str,
    conclusao_rejeita: str,
    conclusao_nao_rejeita: str,
) -> str:
    """Texto que cita α, o p-valor, H0 e H1 e termina com a conclusão no contexto do teste."""
    a, p = formatar_alfa(alfa), formatar_p_valor(p_valor)
    if p_indefinido(p_valor):
        return (
            f"Com α = {a}, o p-valor não pôde ser calculado (dados insuficientes ou "
            f"degenerados): não se rejeita H₀ ({h0}). {conclusao_nao_rejeita}"
        )
    if decidir(p_valor, alfa) == REJEITA_H0:
        return (
            f"Com α = {a}, o p-valor ({p}) é menor ou igual a α: rejeita-se H₀ ({h0}) "
            f"em favor de H₁ ({h1}). {conclusao_rejeita}"
        )
    return (
        f"Com α = {a}, o p-valor ({p}) é maior que α: não se rejeita H₀ ({h0}). "
        f"{conclusao_nao_rejeita}"
    )
