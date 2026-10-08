"""Mede o desempenho das partes do smartETL que dependem do tamanho dos dados (Fase 5).

    python scripts/medir_desempenho.py            # n = 100 000 linhas
    python scripts/medir_desempenho.py 20000      # outro n

Gera bases sintéticas (semente fixa) numa pasta temporária, mede leitura (CSV no formato
brasileiro e XLSX), detecção de tipos, prévia da tabela, cada teste do catálogo e a montagem
das figuras na UI. Imprime o tempo mediano de 3 repetições (1 nas etapas mais lentas).
"""

import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ui.graficos import desenhar_figura
from app.ui.tabela_dados import TabelaDados
from core import registry
from core.base import GrupoFiguras
from core.io import carregar_dados
from core.tipos import detectar_tipos


def medir(rotulo: str, funcao, repeticoes: int = 3):
    tempos, resultado = [], None
    for _ in range(repeticoes):
        inicio = time.perf_counter()
        resultado = funcao()
        tempos.append(time.perf_counter() - inicio)
    print(f"{rotulo:<52} {statistics.median(tempos):8.3f} s")
    return resultado


def base_sintetica(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    grupo = rng.choice(["A", "B", "C", "D"], n)
    x1, x2 = rng.normal(50, 10, n), rng.uniform(0, 100, n)
    y = 10 + 0.5 * x1 - 0.2 * x2 + np.select([grupo == "B", grupo == "C"], [3, -2], 0)
    y = y + rng.normal(0, 5, n)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "y": np.round(y, 3),
            "x1": np.round(x1, 3),
            "x2": np.round(x2, 3),
            "x3": np.round(x1 + rng.normal(0, 8, n), 3),
            "inteiro": rng.integers(1, 6, n),
            "grupo": grupo,
            "fator": rng.choice(["p", "q", "r"], n),
            "binaria": rng.choice(["Sim", "Não"], n, p=[0.4, 0.6]),
            "binaria2": rng.choice(["Sim", "Não"], n, p=[0.5, 0.5]),
        }
    )


PARAMS = {
    "teste_t_1am": {"coluna": "y", "mu0": 40, "alternativa": "μ ≠ μ₀", "alfa": 0.05},
    "teste_t_2am": {"coluna": "y", "grupo": "binaria", "variante": "Welch", "alfa": 0.05},
    "teste_t_pareado": {"coluna1": "x1", "coluna2": "x3", "alfa": 0.05},
    "teste_z_1prop": {"coluna": "binaria", "sucesso": "Sim", "p0": 0.4, "alfa": 0.05},
    "teste_z_2prop": {"coluna": "binaria", "sucesso": "Sim", "grupo": "binaria2", "alfa": 0.05},
    "qui_quadrado": {"modo": "Independência", "coluna1": "grupo", "coluna2": "fator", "alfa": 0.05},
    "fisher": {
        "coluna1": "binaria",
        "evento1": "Sim",
        "coluna2": "binaria2",
        "evento2": "Sim",
        "alfa": 0.05,
    },
    "mcnemar": {"coluna1": "binaria", "coluna2": "binaria2", "evento": "Sim", "alfa": 0.05},
    "teste_sinal": {"modo": "Uma amostra", "coluna1": "y", "m0": 40, "alfa": 0.05},
    "wilcoxon": {"modo": "Uma amostra", "coluna1": "y", "m0": 40, "alfa": 0.05},
    "mann_whitney": {"coluna": "y", "grupo": "binaria", "alfa": 0.05},
    "kruskal_wallis": {"coluna": "y", "grupo": "grupo", "dunn": True, "alfa": 0.05},
    "friedman": {"colunas": ["x1", "x2", "x3"], "comparacoes": True, "alfa": 0.05},
    "anova_1fator": {"coluna": "y", "grupo": "grupo", "tukey": True, "alfa": 0.05},
    "anova_2fator": {"coluna": "y", "fator_a": "grupo", "fator_b": "fator", "alfa": 0.05},
    "correlacao_pearson": {"x": "x1", "y": "y", "alfa": 0.05},
    "correlacao_spearman": {"x": "x1", "y": "y", "alfa": 0.05},
    "regres_linear": {"y": "y", "preditores": ["x1", "x2", "inteiro", "grupo"], "alfa": 0.05},
    "regres_logit": {
        "y": "binaria",
        "evento": "Sim",
        "preditores": ["x1", "x2", "grupo"],
        "alfa": 0.05,
    },
}


def completar(info, params: dict) -> dict:
    """Preenche os parâmetros não informados com os padrões do formulário."""
    teste = info.criar()
    for spec in teste.parametros():
        if spec.nome not in params and spec.padrao is not None:
            params[spec.nome] = spec.padrao
    return params


def main(n: int) -> None:
    print(f"smartETL — medição de desempenho (n = {n:,} linhas)".replace(",", "."))
    df = base_sintetica(n)
    with tempfile.TemporaryDirectory() as pasta:
        csv = Path(pasta) / "base_br.csv"
        df.to_csv(csv, sep=";", decimal=",", index=False, encoding="cp1252")
        xlsx = Path(pasta) / "base.xlsx"
        df.head(min(n, 50_000)).to_excel(xlsx, index=False)
        print(f"  CSV: {csv.stat().st_size / 1e6:.1f} MB; XLSX: {min(n, 50_000)} linhas\n")
        dados = medir("Leitura do CSV (formato brasileiro)", lambda: carregar_dados(csv))
        medir("Leitura do XLSX", lambda: carregar_dados(xlsx), repeticoes=1)
    medir("Detecção de tipos", lambda: detectar_tipos(dados.df))
    medir("Prévia da tabela (100 linhas)", lambda: TabelaDados().mostrar(dados.df))
    print()
    for info in registry.listar():
        if info.id not in PARAMS:
            continue
        teste = info.criar()
        params = completar(info, dict(PARAMS[info.id]))
        resultado = medir(
            f"{info.nome} (executar)",
            lambda teste=teste, params=params: teste.executar(dados.df, params),
            repeticoes=1,
        )
        figuras = []
        for item in resultado.figuras:
            figuras += list(item.opcoes.values()) if isinstance(item, GrupoFiguras) else [item]
        if figuras:
            medir(
                f"    figuras na UI ({len(figuras)})",
                lambda figuras=figuras: [desenhar_figura(f) for f in figuras],
                repeticoes=1,
            )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)
