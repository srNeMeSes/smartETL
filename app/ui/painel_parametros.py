"""Formulário de parâmetros gerado a partir dos `ParametroSpec` do teste."""

import flet as ft
import pandas as pd

from app.ui import tema
from app.ui.componentes import campos
from app.ui.componentes.card_comparacao import CardComparacaoTestes
from app.ui.helpers import esta_na_pagina
from core.base import ALFAS, TIPOS_COLUNA, ErroValidacao, ParametroSpec, TesteBase
from core.testes.regressao import e_categorica, niveis_ordenados, nivel_referencia_padrao
from core.tipos import PerfilColuna, detectar_tipos, niveis_coluna, nivel_sucesso_padrao
from core.validacao import NumeroAmbiguo, colunas_por_tipo, converter_numero


def _alfa_texto(padrao: float | None) -> str:
    return f"{padrao:.2f}".replace(".", ",") if padrao is not None else "0,05"


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
        self._referencias: dict[str, dict[str, ft.Dropdown]] = {}  # spec → {coluna: dropdown}

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
                *(self._com_ajuda(spec, self._criar_campo(spec)) for spec in self.specs),
            ],
            spacing=15,
            # Rola quando há mais campos do que cabem na aba (ex.: 4 campos no t de uma amostra).
            scroll=ft.ScrollMode.AUTO,
        )

        comparacao = teste.comparacao_inicial()
        self.card = CardComparacaoTestes.de_comparacao(comparacao) if comparacao else None
        controles: list[ft.Control] = [formulario]
        if self.card is not None:
            controles += [ft.Column(width=12), self.card]
        super().__init__(controles, alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

    # ---------------- Montagem ----------------
    @staticmethod
    def _com_ajuda(spec: ParametroSpec, campo: ft.Control) -> ft.Control:
        """Campo com o texto explicativo do spec logo abaixo (quando houver)."""
        if not spec.ajuda:
            return campo
        texto = ft.Text(
            spec.ajuda, size=12, color=tema.TEXTO_SECUNDARIO, width=campos.LARGURA_CAMPO
        )
        return ft.Column([campo, texto], spacing=4)

    def _criar_campo(self, spec: ParametroSpec) -> ft.Control:
        if spec.tipo in TIPOS_COLUNA:
            opcoes = colunas_por_tipo(self._df, spec.tipo, self._perfis)
            self._opcoes_coluna[spec.nome] = opcoes
            if spec.tipo in ("multi_coluna", "preditores"):
                caixas = [campos.caixa_selecao(c) for c in opcoes]
                self._controles[spec.nome] = ft.Column(controls=caixas, spacing=0)
                origem = self._controles.get(spec.depende_de or "")
                if spec.tipo == "preditores" and origem is not None:
                    self._excluir_alvo(origem, caixas)
                elif spec.tipo == "multi_coluna" and origem is not None:
                    self._so_marcadas(origem, caixas)
                return ft.Column(
                    controls=[
                        ft.Text(spec.rotulo, weight=ft.FontWeight.BOLD, color=tema.CAMPO_ROTULO),
                        self._controles[spec.nome],
                    ],
                    spacing=4,
                )
            controle = campos.dropdown_campo(spec.rotulo, opcoes, valor=spec.padrao)
        elif spec.tipo == "numero":
            # Padrão exibido com vírgula decimal (pt-BR); converter_numero aceita os dois formatos.
            valor = None if spec.padrao is None else str(spec.padrao).replace(".", ",")
            controle = campos.campo_texto(spec.rotulo, valor=valor)
        elif spec.tipo == "alfa":
            controle = campos.dropdown_campo(spec.rotulo, ALFAS, valor=_alfa_texto(spec.padrao))
        elif spec.tipo == "opcao":
            controle = campos.dropdown_campo(spec.rotulo, spec.opcoes or [], valor=spec.padrao)
        elif spec.tipo == "booleano":
            controle = campos.caixa_selecao(spec.rotulo, bool(spec.padrao))
        elif spec.tipo == "ordenacao":
            numericas = colunas_por_tipo(self._df, "coluna_numerica", self._perfis)
            controle = campos.dropdown_campo(
                spec.rotulo, [*(spec.opcoes or []), *numericas], valor=spec.padrao
            )
        elif spec.tipo == "niveis_referencia":
            controle = ft.Column(spacing=12)
            self._referencias[spec.nome] = {}
            origem = self._controles.get(spec.depende_de or "")
            if origem is not None:
                self._ligar_referencias(spec, origem, controle)
        elif spec.tipo == "nivel":
            controle = campos.dropdown_campo(spec.rotulo, [], icone=ft.Icons.CHECK_CIRCLE_OUTLINE)
            origem = self._controles.get(spec.depende_de or "")
            if origem is not None:
                self._ligar_niveis(origem, controle)
        else:
            raise ValueError(f"Tipo de parâmetro desconhecido: {spec.tipo}")
        self._controles[spec.nome] = controle
        return controle

    def _ligar_niveis(self, origem: ft.Dropdown, destino: ft.Dropdown) -> None:
        """Quando a coluna de `origem` muda, `destino` passa a listar os valores dessa coluna."""

        def atualizar(_evento=None) -> None:
            coluna = origem.value
            niveis = niveis_coluna(self._df[coluna]) if coluna in self._df.columns else []
            campos.definir_opcoes(destino, niveis)
            if destino.value not in niveis:
                destino.value = nivel_sucesso_padrao(niveis)
            if esta_na_pagina(destino):
                destino.update()

        origem.on_select = atualizar
        atualizar()

    @staticmethod
    def _encadear(caixa: ft.Checkbox, funcao) -> None:
        """Acrescenta `funcao` ao on_change da caixa sem apagar o que já estava ligado."""
        anterior = caixa.on_change

        def ambos(evento=None) -> None:
            if anterior is not None:
                anterior(evento)
            funcao(evento)

        caixa.on_change = ambos

    def _so_marcadas(self, origem: ft.Column, caixas: list[ft.Checkbox]) -> None:
        """Só as colunas marcadas em `origem` (os preditores) aparecem; desmarcar um preditor
        desmarca a caixa dele aqui (ex.: "Tratar como categóricas")."""

        def atualizar(_evento=None) -> None:
            marcados = {c.label for c in origem.controls if c.value}
            for caixa in caixas:
                caixa.visible = caixa.label in marcados
                if not caixa.visible and caixa.value:
                    caixa.value = False
                    if caixa.on_change is not None:
                        caixa.on_change(None)
                if esta_na_pagina(caixa):
                    caixa.update()

        for caixa in origem.controls:
            self._encadear(caixa, atualizar)
        atualizar()

    def _forcadas(self, depende_de: str | None) -> ft.Column | None:
        """Caixas "Tratar como categóricas" ligadas aos mesmos preditores, se o teste as tiver."""
        for spec in self.specs:
            if spec.tipo == "multi_coluna" and spec.depende_de == depende_de:
                return self._controles.get(spec.nome)
        return None

    def _excluir_alvo(self, origem: ft.Dropdown, caixas: list[ft.Checkbox]) -> None:
        """A coluna escolhida em `origem` (a variável dependente) some da lista de preditores;
        se estava marcada, é desmarcada (e o que depende das caixas é atualizado)."""
        anterior = origem.on_select

        def atualizar(evento=None) -> None:
            if anterior is not None:
                anterior(evento)
            for caixa in caixas:
                alvo = caixa.label == origem.value
                caixa.visible = not alvo
                if alvo and caixa.value:
                    caixa.value = False
                    if caixa.on_change is not None:
                        caixa.on_change(None)
                if esta_na_pagina(caixa):
                    caixa.update()

        origem.on_select = atualizar
        atualizar()

    def _ligar_referencias(
        self, spec: ParametroSpec, origem: ft.Column, destino: ft.Column
    ) -> None:
        """Uma lista de níveis por coluna categórica marcada em `origem` (padrão: a mais frequente),
        inclusive as numéricas marcadas em "Tratar como categóricas".

        Escolhas já feitas são mantidas quando outras caixas são marcadas ou desmarcadas.
        """
        listas = self._referencias[spec.nome]
        forcadas = self._forcadas(spec.depende_de)

        def atualizar(_evento=None) -> None:
            marcadas = [c.label for c in origem.controls if c.value]
            extra = {c.label for c in forcadas.controls if c.value} if forcadas else set()
            categoricas = [c for c in marcadas if e_categorica(self._df[c]) or c in extra]
            for coluna in list(listas):
                if coluna not in categoricas:
                    del listas[coluna]
            for coluna in categoricas:
                if coluna not in listas:
                    listas[coluna] = campos.dropdown_campo(
                        f"{spec.rotulo} de '{coluna}'",
                        niveis_ordenados(self._df[coluna]),
                        valor=nivel_referencia_padrao(self._df[coluna]),
                        icone=ft.Icons.FLAG_OUTLINED,
                    )
            destino.controls = [listas[c] for c in categoricas]
            if esta_na_pagina(destino):
                destino.update()

        for caixa in [*origem.controls, *(forcadas.controls if forcadas else [])]:
            self._encadear(caixa, atualizar)
        atualizar()

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
            if spec.tipo in ("multi_coluna", "preditores"):
                bruto = [c.label for c in controle.controls if c.value]
                vazio = not bruto
            elif spec.tipo == "niveis_referencia":
                bruto = {c: d.value for c, d in self._referencias[spec.nome].items() if d.value}
                vazio = False
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
                except NumeroAmbiguo as erro:
                    erros.append(f"{spec.rotulo}: {erro}")
                except ValueError:
                    erros.append(f"Informe um número válido em '{spec.rotulo}'.")
            elif spec.tipo == "alfa":
                valores[spec.nome] = converter_numero(bruto)
            else:
                valores[spec.nome] = bruto
        if erros:
            raise ErroValidacao(erros)
        return valores

    def _mensagem_vazio(self, spec: ParametroSpec) -> str:
        if spec.tipo in TIPOS_COLUNA and not self._opcoes_coluna.get(spec.nome):
            return f"O arquivo não tem colunas compatíveis com '{spec.rotulo}'."
        if spec.tipo in ("multi_coluna", "preditores"):
            return f"Selecione ao menos uma coluna em '{spec.rotulo}'."
        return f"Preencha o campo '{spec.rotulo}'."
