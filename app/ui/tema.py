"""Paleta, raios e tipografia do Signa. Única fonte de cores da interface."""

import flet as ft

# Cores da identidade
LARANJA = "#FF6A1A"  # destaque (identidade Signa e 1º teste do card)
LARANJA_SUAVE = "#FFF1E6"  # fundo de itens selecionados, ícones e rodapé do card
ROXO = "#7C3AED"  # 2º teste do card (não paramétrico)
ROXO_SUAVE = "#EDE9FE"

# Superfícies
FUNDO = "#F4F5F7"  # fundo geral; também o "chip" da hipótese no card
CARTAO = "#FFFFFF"
BORDA = "#E7E8EC"
TRANSPARENTE = "transparent"

# Prévia dos dados: cabeçalho e linhas alternadas, bem próximos do branco
TABELA_CABECALHO = "#F9FAFB"
TABELA_ZEBRA = "#FBFBFC"

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
# Rejeitar H₀ é o teste encontrar o efeito (verde); não rejeitar, não encontrar (terracota). As
# duas têm contraste ≥ 3,4:1 no branco (decisão do autor, 2026-10-08).
DECISAO_REJEITA = "#3F9B6B"  # verde suave
DECISAO_NAO_REJEITA = "#C8705F"  # vermelho terracota suave

# Gráficos
GRAFICO_BARRA = LARANJA_SUAVE
GRAFICO_BARRA_BORDA = LARANJA
GRAFICO_DESTAQUE = TEXTO  # linha cheia (ex.: média amostral)
GRAFICO_REFERENCIA = TEXTO_SECUNDARIO  # linha tracejada (ex.: μ₀)
# Séries (barras agrupadas): começa pela identidade (laranja, roxo) e segue com tons sóbrios.
GRAFICO_SERIES = (LARANJA, ROXO, "#0F9D8A", "#E0A100", "#5B6B7F", "#D9467A", "#3B82C4", "#8C6D46")

SOMBRA_CARTAO = ft.Colors.with_opacity(0.06, ft.Colors.BLACK)

# Tipografia
# Inter (SIL Open Font License, assets/fonts/OFL.txt), embutida: a interface fica igual em
# qualquer máquina. Arquivo variável (eixos de peso e tamanho óptico): um só arquivo cobre
# do regular ao negrito. O caminho é relativo à pasta de assets do Flet (main.py).
FONTE = "Inter"
ARQUIVOS_FONTE = {FONTE: "fonts/Inter.ttf"}

# Barras de rolagem: cinza quase transparente, um pouco mais forte sob o mouse
BARRA_ROLAGEM = ft.Colors.with_opacity(0.18, TEXTO_SECUNDARIO)
BARRA_ROLAGEM_ATIVA = ft.Colors.with_opacity(0.40, TEXTO_SECUNDARIO)

# Raios
RAIO_CARTAO = 16
RAIO_PEQUENO = 10


def tema_da_pagina() -> ft.Theme:
    """Fonte e barras de rolagem do app inteiro.

    Parada, a barra é um cinza quase transparente (não disputa atenção com o conteúdo); sob o
    mouse ou arrastada, fica um pouco mais visível (pedido do autor, 2026-10-11).
    """
    return ft.Theme(
        font_family=FONTE,
        scrollbar_theme=ft.ScrollbarTheme(
            thumb_color={
                ft.ControlState.HOVERED: BARRA_ROLAGEM_ATIVA,
                ft.ControlState.DRAGGED: BARRA_ROLAGEM_ATIVA,
                ft.ControlState.DEFAULT: BARRA_ROLAGEM,
            },
        ),
    )
