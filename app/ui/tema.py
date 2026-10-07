"""Paleta, raios e tipografia do smartETL. Única fonte de cores da interface."""

import flet as ft

# Cores da identidade
LARANJA = "#FF6A1A"  # destaque (identidade smartETL e 1º teste do card)
LARANJA_SUAVE = "#FFF1E6"  # fundo de itens selecionados, ícones e rodapé do card
ROXO = "#7C3AED"  # 2º teste do card (não paramétrico)
ROXO_SUAVE = "#EDE9FE"

# Superfícies
FUNDO = "#F4F5F7"  # fundo geral; também o "chip" da hipótese no card
CARTAO = "#FFFFFF"
BORDA = "#E7E8EC"
TRANSPARENTE = "transparent"

# Texto
TEXTO = "#232529"
TEXTO_SECUNDARIO = "#8A8D93"
TEXTO_TERCIARIO = "#B4B6BC"  # também o rádio desmarcado da sidebar
TEXTO_SOBRE_DESTAQUE = "#FFFFFF"

# Campos de formulário
CAMPO_TEXTO = ft.Colors.BLACK
CAMPO_ROTULO = ft.Colors.BLACK_54
CAMPO_BORDA = ft.Colors.BLACK_54

# Notificações (SnackBar)
NOTIFICACAO = TEXTO
NOTIFICACAO_ERRO = "#B42318"

# Decisão do teste (texto "Rejeita H₀" / "Não rejeita H₀"): tons suaves, legíveis no fundo branco
DECISAO_REJEITA = "#D0605A"  # vermelho suave
DECISAO_NAO_REJEITA = "#3F9B6B"  # verde suave

# Gráficos
GRAFICO_BARRA = LARANJA_SUAVE
GRAFICO_BARRA_BORDA = LARANJA
GRAFICO_DESTAQUE = TEXTO  # linha cheia (ex.: média amostral)
GRAFICO_REFERENCIA = TEXTO_SECUNDARIO  # linha tracejada (ex.: μ₀)
# Séries (barras agrupadas): começa pela identidade (laranja, roxo) e segue com tons sóbrios.
GRAFICO_SERIES = (LARANJA, ROXO, "#0F9D8A", "#E0A100", "#5B6B7F", "#D9467A", "#3B82C4", "#8C6D46")

SOMBRA_CARTAO = ft.Colors.with_opacity(0.06, ft.Colors.BLACK)

# Tipografia
FONTE = "Segoe UI, Roboto, Arial, sans-serif"

# Raios
RAIO_CARTAO = 16
RAIO_PEQUENO = 10
