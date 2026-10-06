"""Formulário de parâmetros gerado a partir dos `ParametroSpec` do teste."""

import flet as ft
import pandas as pd

from app.ui import tema
from app.ui.componentes import campos
from app.ui.componentes.card_comparacao import CardComparacaoTestes
from core.base import ALFAS, TIPOS_COLUNA, ErroValidacao, ParametroSpec, TesteBase
from core.tipos import PerfilColuna, detectar_tipos
from core.validacao import colunas_por_tipo, converter_numero


def _alfa_texto(padrao: float | None) -> str:
    return f"{padrao:.2f}" if padrao is not None else "0.05"


class PainelParametros(ft.Row):
    """Aba Parâmetros: formulário à esquerda e card de comparação (se houver) à direita."""

    def __init__(
        self,
        teste: TesteBase,
        df: pd.DataFrame,
        perfis: dict[str, PerfilColuna] | None = None,
    ):
        self.specs = teste.parametros()
        self._df = df
        self._perfis = perfis if perfis is not None else detectar_tipos(df)
        self._controles: dict[str, ft.Control] = {}
        self._opcoes_coluna: dict[str, list[str]] = {}

        formulario = ft.Column(
            controls=[
                ft.Column(
                    spacing=0.9,
                    controls=[
                        ft.Text(
                            "Parâmetros do teste",
                            size=16,
                            weight=ft.FontWeight.BOLD,
                            color=tema.TEXTO,
                        ),
                        ft.Text("Selecione a variável e informe os parâmetros.", color=tema.TEXTO),
                    ],
                ),
                ft.Divider(color=tema.TRANSPARENTE, height=1),
                *(self._criar_campo(spec) for spec in self.specs),
            ],
            spacing=15,
        )

        comparacao = teste.comparacao_inicial()
        self.card = CardComparacaoTestes.de_comparacao(comparacao) if comparacao else None
        controles: list[ft.Control] = [formulario]
        if self.card is not None:
            controles += [ft.Column(width=12), self.card]
        super().__init__(controles, alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

    # ---------------- Montagem ----------------
    def _criar_campo(self, spec: ParametroSpec) -> ft.Control:
        if spec.tipo in TIPOS_COLUNA:
            opcoes = colunas_por_tipo(self._df, spec.tipo, self._perfis)
            self._opcoes_coluna[spec.nome] = opcoes
            if spec.tipo == "multi_coluna":
                caixas = [campos.caixa_selecao(c) for c in opcoes]
                self._controles[spec.nome] = ft.Column(controls=caixas, spacing=0)
                return ft.Column(
                    controls=[
                        ft.Text(spec.rotulo, weight=ft.FontWeight.BOLD, color=tema.CAMPO_ROTULO),
                        self._controles[spec.nome],
                    ],
                    spacing=4,
                )
            controle = campos.dropdown_campo(spec.rotulo, opcoes, valor=spec.padrao)
        elif spec.tipo == "numero":
            valor = None if spec.padrao is None else str(spec.padrao)
            controle = campos.campo_texto(spec.rotulo, valor=valor)
        elif spec.tipo == "alfa":
            controle = campos.dropdown_campo(spec.rotulo, ALFAS, valor=_alfa_texto(spec.padrao))
        elif spec.tipo == "opcao":
            controle = campos.dropdown_campo(spec.rotulo, spec.opcoes or [], valor=spec.padrao)
        elif spec.tipo == "booleano":
            controle = campos.caixa_selecao(spec.rotulo, bool(spec.padrao))
        else:
            raise ValueError(f"Tipo de parâmetro desconhecido: {spec.tipo}")
        self._controles[spec.nome] = controle
        return controle

    def controle(self, nome: str) -> ft.Control:
        """Campo do parâmetro `nome` (por referência, nunca por índice)."""
        return self._controles[nome]

    # ---------------- Leitura ----------------
    def coletar_valores(self) -> dict:
        """Valores digitados/selecionados, já convertidos. Lança ErroValidacao."""
        valores: dict = {}
        erros: list[str] = []
        for spec in self.specs:
            controle = self._controles[spec.nome]
            if spec.tipo == "multi_coluna":
                bruto = [c.label for c in controle.controls if c.value]
                vazio = not bruto
            elif spec.tipo == "booleano":
                bruto, vazio = bool(controle.value), False
            else:
                bruto = controle.value
                vazio = bruto is None or str(bruto).strip() == ""

            if vazio:
                if spec.obrigatorio:
                    erros.append(self._mensagem_vazio(spec))
                valores[spec.nome] = None
                continue

            if spec.tipo == "numero":
                try:
                    valores[spec.nome] = converter_numero(bruto)
                except ValueError:
                    erros.append(f"Informe um número válido em '{spec.rotulo}'.")
            elif spec.tipo == "alfa":
                valores[spec.nome] = float(bruto)
            else:
                valores[spec.nome] = bruto
        if erros:
            raise ErroValidacao(erros)
        return valores

    def _mensagem_vazio(self, spec: ParametroSpec) -> str:
        if spec.tipo in TIPOS_COLUNA and not self._opcoes_coluna.get(spec.nome):
            return f"O arquivo não tem colunas compatíveis com '{spec.rotulo}'."
        if spec.tipo == "multi_coluna":
            return f"Selecione ao menos uma coluna em '{spec.rotulo}'."
        return f"Preencha o campo '{spec.rotulo}'."
