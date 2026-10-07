"""Regressão linear (especificação em docs/regressao_linear.md)."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.tipos import e_numerica, ordenar_niveis, rotulo_nivel

INTERCEPTO = "(Intercepto)"
MIN_POR_NIVEL = 5


@dataclass(frozen=True)
class VariavelModelo:
    """Preditor original do modelo e as colunas que ele gera na matriz X."""

    nome: str
    categorica: bool
    colunas: list[int]  # índices em X (o intercepto é a coluna 0)
    niveis: list[str] = field(default_factory=list)  # categórica: todos, inclusive a referência
    referencia: str | None = None
    contagens: dict[str, int] = field(default_factory=dict)


@dataclass
class DadosModelo:
    """y e matriz X (com intercepto) prontos para o ajuste, após descartar linhas incompletas."""

    nome_y: str
    y: np.ndarray
    x: np.ndarray
    colunas: list[str]  # nome de cada coluna de X ("(Intercepto)", "x", "cargo[Gerente]"...)
    variaveis: list[VariavelModelo]
    linhas: pd.DataFrame  # dados originais das linhas usadas (y e preditores)
    descartadas: int


def e_categorica(serie: pd.Series) -> bool:
    """Preditor categórico = coluna não numérica (texto, booleana); numéricas entram como número."""
    return not e_numerica(serie)


def niveis_ordenados(serie: pd.Series) -> list[str]:
    """Níveis de uma coluna categórica como texto, em ordem crescente."""
    return [rotulo_nivel(v) for v in ordenar_niveis(list(pd.unique(serie.dropna())))]


def nivel_referencia_padrao(serie: pd.Series) -> str | None:
    """Nível mais frequente (empate: o primeiro em ordem crescente)."""
    textos = serie.dropna().map(rotulo_nivel)
    if textos.empty:
        return None
    contagens = textos.value_counts()
    maximo = contagens.max()
    return next(n for n in niveis_ordenados(serie) if contagens.get(n, 0) == maximo)


def preparar_dados(
    df: pd.DataFrame, nome_y: str, preditores: list[str], referencias: dict[str, str] | None = None
) -> DadosModelo:
    """Monta y e X (intercepto + numéricas + k − 1 dummies por categórica).

    Linhas com valor ausente em y ou em algum preditor são descartadas. `referencias` dá o nível
    de referência de cada categórica (ausente ou inválido → o mais frequente). A ordem das
    dummies segue a ordem crescente dos níveis, sem a referência.
    """
    referencias = referencias or {}
    linhas = df[[nome_y, *preditores]].dropna()
    descartadas = len(df) - len(linhas)
    colunas = [INTERCEPTO]
    blocos = [np.ones(len(linhas))]
    variaveis = []
    for nome in preditores:
        serie = linhas[nome]
        if e_categorica(serie):
            textos = serie.map(rotulo_nivel)
            niveis = niveis_ordenados(serie)
            referencia = referencias.get(nome)
            if referencia not in niveis:
                referencia = nivel_referencia_padrao(serie)
            indices = []
            for nivel in niveis:
                if nivel == referencia:
                    continue
                indices.append(len(colunas))
                colunas.append(f"{nome}[{nivel}]")
                blocos.append((textos == nivel).to_numpy(dtype=float))
            contagens = {n: int((textos == n).sum()) for n in niveis}
            variaveis.append(VariavelModelo(nome, True, indices, niveis, referencia, contagens))
        else:
            variaveis.append(VariavelModelo(nome, False, [len(colunas)]))
            colunas.append(nome)
            blocos.append(serie.to_numpy(dtype=float))
    return DadosModelo(
        nome_y=nome_y,
        y=linhas[nome_y].to_numpy(dtype=float),
        x=np.column_stack(blocos),
        colunas=colunas,
        variaveis=variaveis,
        linhas=linhas,
        descartadas=descartadas,
    )


def variaveis_colineares(dados: DadosModelo) -> list[list[str]]:
    """Grupos de preditores em colinearidade perfeita (vazio se X tem posto completo).

    Cada vetor do núcleo de X (valores singulares ≈ 0) dá uma combinação linear exata; os
    preditores originais com peso não nulo nela formam um grupo. "(Intercepto)" aparece quando a
    combinação envolve uma constante (ex.: preditor constante, dummies que somam 1).
    """
    x = dados.x
    escala = np.where(np.abs(x).max(axis=0) > 0, np.abs(x).max(axis=0), 1.0)
    _, valores, vt = np.linalg.svd(x / escala, full_matrices=True)
    tolerancia = max(x.shape) * np.finfo(float).eps * (valores[0] if len(valores) else 1.0) * 1e3
    nulos = [vt[i] for i in range(x.shape[1]) if i >= len(valores) or valores[i] <= tolerancia]
    dono = {0: INTERCEPTO}
    for variavel in dados.variaveis:
        for indice in variavel.colunas:
            dono[indice] = variavel.nome
    grupos: list[list[str]] = []
    for vetor in nulos:
        pesos = np.abs(vetor) / np.abs(vetor).max()
        nomes: list[str] = []
        for indice in np.nonzero(pesos > 1e-6)[0]:
            if dono[int(indice)] not in nomes:
                nomes.append(dono[int(indice)])
        if nomes not in grupos:
            grupos.append(nomes)
    return grupos
