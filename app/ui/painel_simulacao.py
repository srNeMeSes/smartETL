"""Aba Simulação: equação do modelo, campos de entrada e previsão em tempo real.

A previsão é calculada no core (`SimuladorRegressao.prever`), pedida pelo controller: os
callbacks daqui só leem os campos e repassam os valores.
"""

from collections.abc import Callable

import flet as ft

from app.ui import tema
from app.ui.componentes import campos
from app.ui.graficos import desenhar_figura
from app.ui.helpers import border_all, esta_na_pagina
from core.interpretacao import formatar_numero
from core.testes.regressao import CampoSimulacao, Previsao, SimuladorRegressao, formatar_coeficiente
from core.validacao import converter_numero

LARGURA_SLIDER = 230
LARGURA_VALOR = 130
MAX_PASSOS = 500


def _formatar(campo: CampoSimulacao, valor: float) -> str:
    """Inteiros sem casas decimais ("3", "1.200"); demais com 4 algarismos significativos."""
    if campo.inteiro and float(valor).is_integer():
        return formatar_numero(valor, 0)
    return formatar_coeficiente(valor)


class PainelSimulacao(ft.Column):
    def __init__(
        self,
        simulador: SimuladorRegressao,
        on_alterar: Callable[[dict, str | None], None],
    ):
        self._simulador = simulador
        self._on_alterar = on_alterar
        self.valores: dict[str, float | str] = simulador.valores_iniciais()
        self.textos: dict[str, ft.TextField] = {}
        self.sliders: dict[str, ft.Slider] = {}
        self.listas: dict[str, ft.Dropdown] = {}
        self._campos: dict[str, CampoSimulacao] = {}

        self.equacao = ft.Text(size=15, selectable=True)
        self.valor = ft.Text(size=28, weight=ft.FontWeight.BOLD, color=tema.LARANJA)
        self.intervalos = ft.Column(spacing=4)
        self.notas = ft.Column(spacing=2)
        self.avisos = ft.Column(spacing=4)
        self.grafico = ft.Container()

        entradas = ft.Column(
            [self._criar_campo(c) for c in simulador.campos], spacing=14, width=420
        )
        resultado = ft.Column(
            [
                ft.Text(
                    simulador.titulo_resultado,
                    size=14,
                    weight=ft.FontWeight.W_600,
                    color=tema.TEXTO,
                ),
                self.valor,
                self.intervalos,
                self.notas,
                self.avisos,
                self.grafico,
            ],
            spacing=8,
            expand=True,
        )
        cartao_equacao = ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        "Equação do modelo",
                        size=14,
                        weight=ft.FontWeight.W_600,
                        color=tema.TEXTO,
                    ),
                    self.equacao,
                    *(
                        [ft.Text(simulador.rodape_equacao, size=13, color=tema.TEXTO_SECUNDARIO)]
                        if simulador.rodape_equacao
                        else []
                    ),
                ],
                spacing=6,
            ),
            padding=14,
            border=border_all(1, tema.BORDA),
            border_radius=tema.RAIO_PEQUENO,
            bgcolor=tema.FUNDO,
        )
        super().__init__(
            [
                cartao_equacao,
                ft.Row(
                    [entradas, resultado],
                    spacing=32,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
            ],
            spacing=18,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )
        self._desenhar_equacao(None)

    # ---------------- Montagem ----------------
    def _criar_campo(self, campo: CampoSimulacao) -> ft.Control:
        if campo.categorica:
            lista = campos.dropdown_campo(
                campo.nome,
                list(campo.niveis),
                valor=campo.inicial,
                largura=LARGURA_VALOR + LARGURA_SLIDER,
            )
            lista.on_select = lambda _e, nome=campo.nome: self._ao_escolher(nome)
            self.listas[campo.nome] = lista
            return lista
        self._campos[campo.nome] = campo
        texto = campos.campo_texto(
            campo.nome, valor=_formatar(campo, float(campo.inicial)), largura=LARGURA_VALOR
        )
        texto.on_change = lambda _e, nome=campo.nome: self._ao_digitar(nome)
        passos = int(campo.maximo - campo.minimo)
        slider = ft.Slider(
            min=campo.minimo,
            max=campo.maximo,
            value=float(campo.inicial),
            # Inteiros: o slider anda de 1 em 1 (até MAX_PASSOS posições; acima, contínuo).
            divisions=passos if campo.inteiro and 1 <= passos <= MAX_PASSOS else None,
            active_color=tema.LARANJA,
            width=LARGURA_SLIDER,
        )
        slider.on_change = lambda _e, nome=campo.nome: self._ao_deslizar(nome)
        self.textos[campo.nome] = texto
        self.sliders[campo.nome] = slider
        faixa = ft.Text(
            f"Faixa observada: {_formatar(campo, campo.minimo)} a {_formatar(campo, campo.maximo)}",
            size=12,
            color=tema.TEXTO_SECUNDARIO,
        )
        return ft.Column(
            [
                ft.Row([texto, slider], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                faixa,
            ],
            spacing=2,
        )

    # ---------------- Callbacks (só repassam os valores) ----------------
    def iniciar(self) -> None:
        """Pede a previsão dos valores iniciais (médias e níveis de referência)."""
        self._on_alterar(dict(self.valores), None)

    def _ao_deslizar(self, nome: str) -> None:
        campo = self._campos[nome]
        valor = float(self.sliders[nome].value)
        if campo.inteiro:
            valor = float(round(valor))
        self.valores[nome] = valor
        self.textos[nome].value = _formatar(campo, valor)
        if esta_na_pagina(self.textos[nome]):
            self.textos[nome].update()
        self._on_alterar(dict(self.valores), nome)

    def _ao_digitar(self, nome: str) -> None:
        try:
            valor = converter_numero(self.textos[nome].value)
        except ValueError:
            return  # texto incompleto ("1,", "-"): espera o próximo caractere
        self.valores[nome] = valor
        slider = self.sliders[nome]
        slider.value = min(max(valor, slider.min), slider.max)  # o slider fica na borda da faixa
        if esta_na_pagina(slider):
            slider.update()
        self._on_alterar(dict(self.valores), nome)

    def _ao_escolher(self, nome: str) -> None:
        self.valores[nome] = self.listas[nome].value
        self._on_alterar(dict(self.valores), nome)

    # ---------------- Exibição ----------------
    def _desenhar_equacao(self, destaque: str | None) -> None:
        estilo_normal = ft.TextStyle(color=tema.TEXTO)
        estilo_destaque = ft.TextStyle(
            color=tema.LARANJA, weight=ft.FontWeight.BOLD, bgcolor=tema.LARANJA_SUAVE
        )
        self.equacao.spans = [
            ft.TextSpan(
                termo.texto,
                style=estilo_destaque
                if destaque is not None and termo.variavel == destaque
                else estilo_normal,
            )
            for termo in self._simulador.equacao()
        ]

    def mostrar(self, previsao: Previsao, variavel: str | None) -> None:
        self._desenhar_equacao(variavel)
        valor, linhas, notas = self._simulador.textos_previsao(previsao)
        self.valor.value = valor
        self.intervalos.controls = [
            ft.Text(texto, size=14, color=tema.TEXTO, tooltip=dica) for texto, dica in linhas
        ]
        self.notas.controls = [
            ft.Text(nota, size=12, color=tema.TEXTO_SECUNDARIO) for nota in notas
        ]
        self.avisos.controls = [
            ft.Row(
                [
                    ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, size=16, color=tema.LARANJA),
                    ft.Text(aviso, size=13, color=tema.TEXTO_SECUNDARIO, expand=True),
                ],
                spacing=6,
            )
            for aviso in previsao.extrapolacoes
        ]
        self.grafico.content = desenhar_figura(previsao.figura, largura=560, altura=230)
        if esta_na_pagina(self):
            self.update()
