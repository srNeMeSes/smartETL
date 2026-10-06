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

SOMBRA_CARTAO = ft.Colors.with_opacity(0.06, ft.Colors.BLACK)

# Tipografia
FONTE = "Segoe UI, Roboto, Arial, sans-serif"

# Raios
RAIO_CARTAO = 16
RAIO_PEQUENO = 10
