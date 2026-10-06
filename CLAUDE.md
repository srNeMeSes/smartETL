# smartETL — Testes de Hipótese

## 1. Visão geral

Aplicativo desktop de **processamento e análise de dados** com foco em **testes de hipótese** (paramétricos e não paramétricos), ANOVA, regressão e diagnósticos de pressupostos. O usuário carrega um arquivo (CSV/XLSX), escolhe um teste na barra lateral, configura os parâmetros, executa e vê resultado, interpretação e gráficos.

- **Linguagem:** Python 3.10+ (o código atual já usa `list[dict] | None` em assinaturas; alvo: 3.11)
- **Interface:** Flet **`0.86.2`** (versão fixada em `requirements.txt`)
- **Estatística:** `scipy.stats`, `statsmodels`, `pandas`, `numpy`
- **Idioma da interface e das interpretações:** português do Brasil
- **Estado atual:** Fases 0, 1 e 2 concluídas. Arquitetura modular da seção 4 em funcionamento (`python main.py`), com leitura robusta de CSV/XLSX e detecção de tipos. Fase 3 em andamento (5/15): **grupos Médias e Proporções completos** (`teste_t_1am`, `teste_t_2am`, `teste_t_pareado`, `teste_z_1prop`, `teste_z_2prop`), cumprindo o checklist da seção 9; os demais aparecem como "ainda não disponível". Próximo: `qui_quadrado`.

## 2. Missão do Claude neste projeto

1. **Reestruturar** o projeto numa arquitetura modular (UI, controller e lógica estatística separados), preservando visual e comportamento atuais.
2. **Otimizar** o código existente (duplicação, acoplamento, imports, carregamento de dados).
3. **Implementar os testes um por um**, na ordem da seção 7, **validando cada um com testes automatizados antes de passar ao próximo**.
4. **Testar cada seção** do app (carregamento de dados, cada teste, cada grupo, controller, montagem da UI).

### Regras de trabalho

- Nunca implemente dois testes ao mesmo tempo. Um teste só está "pronto" quando cumpre o checklist da seção 9.
- Antes de refatorar, rode o app e registre o comportamento atual (seção 3). Depois, o visual e a navegação devem permanecer iguais.
- Mudanças pequenas e commits atômicos, um por etapa/teste.
- A versão do Flet é **0.86.2** e a API é a nova (`ft.run`, `ft.Alignment`, `ft.Padding.symmetric`, `FilePicker` como serviço em `page.services`, `pick_files` assíncrono). **Não use APIs antigas** vistas em tutoriais (`ft.app`, `ft.padding.all`, `ft.alignment.center`...). Em caso de dúvida, consulte a documentação dessa versão.
- Não invente valores de referência. Use exemplos de livro, resultados de R/scipy/statsmodels calculados de forma independente ou cálculo manual com numpy, e documente a fonte no próprio teste.
- Se algo for ambíguo (variante do teste, correção de continuidade, hipótese alternativa padrão), escolha o padrão de livro-texto, documente e deixe configurável na UI.
- Não apague o que não for substituído: cada arquivo antigo só sai quando o equivalente novo estiver funcionando.

## 3. Estado atual do código

A estrutura plana original (`smartetl_app.py`, `utils.py`, `conteiner_parametros.py`, `ConteinerTestes.py`, `load_table.py`) foi substituída na Fase 1 pela arquitetura da seção 4 e removida. O histórico está no git.

### Comportamento atual (preservar)

- Janela 1440×900 (mín. 1150×720), **centralizada** (`page.run_task(page.window.center)` — `Window.center` é assíncrono no Flet 0.86.2); fundo `tema.FUNDO`, sem padding na página.
- Sidebar de 260 px: logo "smartETL", subtítulo, botão **Arquivo** (fundo laranja suave), lista rolável de testes **agrupada por categoria** (cabeçalhos em maiúsculas, primeiro teste marcado por padrão), botão **Executar teste** (laranja, ícone de balança).
- Área principal: título "Processamento de dados", subtítulo "Testes de Hipótese | paramétricos e não paramétricos", tabela de prévia (altura 320, rolagem horizontal e vertical) e painel de 3 abas (**Parâmetros**, **Análise**, **Visualização**).
- Tabela vazia: 20 colunas `column1..20` (cinza claro) e 12 linhas em branco. Com arquivo: **só as colunas reais**, no máximo **100 linhas**, NaN exibido vazio.
- Seleção de arquivo: `FilePicker` em `page.services`, extensões `xlsx` e `csv`, um arquivo. Erros de leitura viram `SnackBar` vermelho em português.
- Estados vazios: sem arquivo → "Carregue um arquivo para começar."; teste sem implementação → "O <teste> ainda não está disponível nesta versão."; sem execução → "Configure os parâmetros e clique em Executar teste." "Processando..." só durante a execução.
- Formulário do `teste_t_1am` (gerado pelos `ParametroSpec`): dropdown "Variável" (só colunas numéricas), campo "Média Hipotética" (aceita vírgula decimal), dropdown "Nível de significância (α)" (0.01 / 0.05 / 0.10, padrão 0.05), e o card de comparação ao lado, com p-valores "—" e hipóteses "μ ≠ μ₀", "μ > μ₀", "μ < μ₀". A aba Análise tem uma **segunda instância** do card.

### Problemas da linha de base (Fase 0) e situação

| # | Problema | Situação |
|---|----------|----------|
| 1 | Abas presas em "Processando..."; teste inicial e arquivo carregado depois não atualizavam o painel | Resolvido (controller + estados vazios) |
| 2 | "Executar teste" sem `on_click` | Resolvido (ligado a `controller.executar`) |
| 3 | Despacho por `globals()` caía no formulário do `teste_t_1am` | Resolvido (`registry` + `TesteInfo.disponivel`) |
| 4 | Mesma instância do card em duas abas | Resolvido (duas instâncias) |
| 5 | Acesso por índice à árvore de controles | Resolvido (`PainelAbas.definir_*`) |
| 6 | Formulário sem referência aos campos, sem filtro de tipo e sem validação; estilos repetidos | Resolvido (`painel_parametros.py`, `campos.py`, `core/validacao.py`) |
| 7 | Cores duplicadas: 3 laranjas, **3 bordas**, 3 laranjas suaves e 4 cinzas de texto | Resolvido (`tema.py`; teste impede hex fora dele) |
| 8 | `import *`, imports e código morto | Resolvido (ruff com F403/F405, I, N, UP, B, RUF) |
| 9 | Leitura sem robustez; callback sem `try/except` | Resolvido na Fase 2 (`core/io.py`: encoding, separador, decimal/milhar, `ErroLeitura`) |
| 10 | Placeholders misturados aos dados; coluna real "column…" pintada como fantasma | Resolvido (`TabelaDados.mostrar_vazio` / `mostrar`) |
| 11 | Dependências incompletas; README com Python 3.9 | Resolvido (`requirements.txt` fixado, README reescrito) |
| 12 | Nomes `ConteinerTestes.py` / `conteiner_parametros.py` | Resolvido (`card_comparacao.py`, `painel_parametros.py`) |

Achados extras da Fase 0, também resolvidos: `CardComparacaoTestes.atualizar_*` quebrava fora da página (`Control.page` lança `RuntimeError` no Flet 0.86.2 — use `helpers.esta_na_pagina`); `ft.ElevatedButton` obsoleto (usar `ft.Button`); p-valores falsos "0.001" no card; hipótese fixa "μ¹ ≠ μ²"; janela não centralizada; `pd.errors.EmptyDataError` é `ValueError` (tratar antes de exibir mensagens de `ValueError`).

## 4. Arquitetura alvo

```
smartetl/
├── main.py                       # ponto de entrada: configura a página, ft.run(main)
├── CLAUDE.md
├── README.md
├── requirements.txt              # versões fixadas
├── pyproject.toml                # config do ruff (regras explícitas) e do pytest (pythonpath)
├── .gitignore
├── app/
│   ├── ui/
│   │   ├── tema.py               # UMA paleta (cores, raios, tipografia)
│   │   ├── helpers.py            # pad(), border_all(), border_only(), esta_na_pagina()
│   │   ├── sidebar.py            # logo, Arquivo, lista de testes agrupada, Executar
│   │   ├── tabela_dados.py       # prévia do dataset (estado vazio vs. dados)
│   │   ├── painel_abas.py        # Parâmetros / Análise / Visualização (API, sem índices)
│   │   ├── painel_parametros.py  # formulário gerado a partir de ParametroSpec
│   │   ├── tela_principal.py     # monta a tela e implementa a Visao do controller
│   │   ├── graficos.py           # desenha Figura com flet.canvas (nativo, minimalista)
│   │   └── componentes/
│   │       ├── campos.py         # dropdown/campo/checkbox com o estilo único
│   │       └── card_comparacao.py
│   ├── controller.py             # liga UI ↔ core via Protocol Visao (sem Flet, sem estatística)
│   └── state.py                  # df, teste selecionado, parâmetros, último resultado
├── core/
│   ├── base.py                   # contratos (ver abaixo) e exceções de domínio
│   ├── registry.py               # TesteInfo(id, nome, grupo, classe) dos 21 testes
│   ├── io.py                     # carregar_dados → DadosCarregados; ErroLeitura
│   ├── tipos.py                  # PerfilColuna / detectar_tipos (numérica, categórica, binária)
│   ├── interpretacao.py          # decidir (p ≤ α), interpretar, formatar_numero/p_valor em pt-BR
│   ├── figuras.py                # construtores de Figura (histograma, boxplot, barras)
│   └── testes/
│       ├── medias.py             # TesteT1Amostra (só formulário, por enquanto)
│       ├── proporcoes.py
│       ├── categoricos.py
│       ├── nao_parametricos.py
│       ├── anova.py
│       └── regressao.py
└── tests/
    ├── conftest.py               # datasets pequenos, determinísticos
    ├── ajudantes_ui.py           # percorrer a árvore de controles Flet nos testes
    ├── core/
    └── app/
```

Os testes rodam sem janela: a `Page` é um `MagicMock` e o controller é testado com uma `Visao` falsa. Nomes de arquivos de teste devem ser únicos entre `tests/core` e `tests/app` (não há `__init__.py`). O pytest só coleta `test_*` e `Test_*` (classes de domínio começam com `Teste`).

### Mapa de migração (concluído na Fase 1)

| Atual | Destino |
|-------|---------|
| `smartetl_app.py` (paleta, helpers) | `app/ui/tema.py`, `app/ui/helpers.py` |
| `smartetl_app.py` (sidebar, cabeçalho, tabela, abas) | `app/ui/sidebar.py`, `tabela_dados.py`, `painel_abas.py`; montagem em `app/ui/tela_principal.py` e `main.py` |
| `smartetl_app.py` (`selecionar_arquivo`, `carregar_base`, `mostrar_testes`) | `app/controller.py` + `app/state.py` |
| `utils.py` → `testes_hipotese` | gerada por `core/registry.py` (fonte única) |
| `utils.py` → `criar_sidebar_testes` | `app/ui/sidebar.py` (com cabeçalhos por grupo) |
| `conteiner_parametros.py` | `app/ui/painel_parametros.py` (genérico, por `ParametroSpec`) + cada teste em `core/testes/` |
| `ConteinerTestes.py` | `app/ui/componentes/card_comparacao.py` |
| `load_table.py` | `core/io.py` |

### Princípios

- **UI não calcula; `core/` não conhece Flet.** Toda lógica estatística é testável sem abrir janela.
- **Cada teste é uma classe** (`TesteBase`) registrada em `registry.py`. A UI é montada a partir dos metadados (`ParametroSpec`); acabam o `globals().get(...)` e os `if teste_id == ...`.
- Adicionar um teste novo = criar a classe + registrar.
- **Um único estilo de campo** (`componentes/campos.py`) extraído dos formulários atuais: borda 2 px cinza, raio 10, altura 58, foco laranja `#FF6A1A`, label em negrito.
- **Uma única paleta** em `tema.py`, partindo de `#FF6A1A` (laranja), `#FFF1E6` (laranja suave), `#F4F5F7` (fundo), `#FFFFFF` (cartões), `#E7E8EC` (borda), `#232529` (texto), `#8A8D93` / `#B4B6BC` (cinzas). O roxo do card (`#7C3AED`) permanece como cor do "segundo teste".

### Fluxo da aplicação

1. **Arquivo** → `controller.carregar_arquivo()` → `core/io.carregar_dados()` → `state.dados` (df + perfis de tipo + avisos) → atualiza prévia e o painel de parâmetros (se já houver teste selecionado); avisos da leitura vão na notificação.
2. **Seleção de teste** (inclusive o inicial) → `controller.selecionar_teste(id)` → painel de parâmetros renderiza o formulário a partir de `teste.parametros()`.
3. **Executar teste** → coleta valores dos campos → `teste.validar(df, params)` → `teste.executar(df, params)` (fora da thread da UI) → preenche **Análise** e **Visualização**.
4. Estados vazios claros em vez de "Processando...": sem arquivo ("Carregue um arquivo para começar"), sem execução ("Configure os parâmetros e clique em Executar teste"). "Processando..." só durante a execução real.

### Contrato de um teste

Implementado em `core/base.py` (é a fonte da verdade; abaixo, o resumo).

```python
@dataclass(frozen=True)
class ParametroSpec:
    nome: str
    rotulo: str                 # texto exibido na UI (pt-BR)
    tipo: Literal["coluna_numerica", "coluna_categorica", "coluna_binaria",
                  "multi_coluna", "numero", "alfa", "opcao", "booleano",
                  "nivel"]      # "nivel": um valor de outra coluna (ex.: o "sucesso")
    padrao: Any = None
    opcoes: list[str] | None = None
    obrigatorio: bool = True
    depende_de: str | None = None  # "nivel": parâmetro de coluna cujos valores são listados

@dataclass
class Figura:                   # especificação sem Flet; a UI desenha (app/ui/graficos.py)
    tipo: Literal["histograma", "boxplot", "barras"]  # novos tipos: core/figuras + ui/graficos
    titulo: str
    dados: dict[str, Any]       # histograma: bordas, contagens, rotulo_x, referencias
                                # boxplot: rotulo_y, grupos[{rotulo, n, q1, mediana, q3, bigodes, media, outliers}]
                                # barras: rotulo_y, maximo, percentual, categorias[{rotulo, valor}], referencias

@dataclass
class ResultadoTeste:
    teste_id: str
    estatisticas: dict[str, float]        # ex.: {"t": 2.31, "gl": 29}
    p_valor: float | None
    alfa: float
    decisao: str                           # "Rejeita H0" / "Não rejeita H0"
    interpretacao: str                     # texto em pt-BR
    tabelas: dict[str, pd.DataFrame]       # tabelas extras (ANOVA, coeficientes...)
    figuras: list[Figura]                  # para a aba Visualização
    avisos: list[str]                      # pressupostos duvidosos, n pequeno etc.
    comparacao: ComparacaoPValores | None  # alimenta o card (ver abaixo)

class TesteBase(ABC):
    id: str
    nome: str
    grupo: str
    @abstractmethod
    def parametros(self) -> list[ParametroSpec]: ...
    @abstractmethod
    def validar(self, df: pd.DataFrame, params: dict) -> list[str]: ...  # lista de erros
    @abstractmethod
    def executar(self, df: pd.DataFrame, params: dict) -> ResultadoTeste: ...
    def comparacao_inicial(self) -> ComparacaoPValores | None:  # card antes da execução
        return None

# Exceções: ErroValidacao(mensagens), ErroExecucao, TesteNaoImplementado(ErroExecucao)
```

O teste é registrado em `core/registry.py` (`TesteInfo(..., classe=MinhaClasse)`); `classe=None` significa "ainda não disponível" e a UI mostra esse estado. `id`, `nome` e `grupo` da classe devem coincidir com o `TesteInfo` (há teste automatizado). O formulário recebe os valores já convertidos por `PainelParametros.coletar_valores()` (números como `float`, α como `float`, colunas como `str`).

### Card de comparação (`CardComparacaoTestes`)

Mantenha o componente: ele mostra, para cada hipótese alternativa (≠, >, <), o p-valor do teste paramétrico (laranja, à esquerda) ao lado do p-valor do equivalente não paramétrico (roxo, à direita). A nota do rodapé diz que os valores são calculados após a execução. Antes da execução o card é criado com `CardComparacaoTestes.de_comparacao(teste.comparacao_inicial())` (p-valores "—"); depois, `TelaPrincipal.exibir_resultado` cria o card da Análise a partir de `ResultadoTeste.comparacao` e chama `card.aplicar(comparacao)` no card de Parâmetros (p-valores formatados com vírgula decimal, "< 0,001" abaixo de 0,001):

```python
@dataclass
class ComparacaoPValores:
    titulo_esquerda: str                               # "t Student"
    titulo_direita: str                                # "Wilcoxon"
    hipoteses: list[str]                               # H1 de cada linha: "μ ≠ μ₀", "μ > μ₀", "μ < μ₀"
    linhas: list[tuple[float | None, float | None]]    # (p_param, p_nao_param); None = não calculado
```

Nem todo teste tem essa estrutura (ex.: qui-quadrado, ANOVA, diagnósticos): o teste não sobrescreve `comparacao_inicial()` (retorna `None`), devolve `comparacao=None` no resultado, e a UI exibe só a tabela/estatísticas. O texto das hipóteses vem do teste (o card prefixa "Hₐ:").

## 5. Stack estatística

Pode-se delegar o cálculo a `scipy`/`statsmodels`, mas o *wrapper* deve mapear corretamente parâmetros (hipótese alternativa, correção de continuidade, `ddof`, exato vs. assintótico), e isso **precisa ser testado**.

Dependências (todas em `requirements.txt`, com versões fixadas): `flet==0.86.2`, `pandas`, `numpy`, `scipy`, `statsmodels`, `openpyxl`; opcional `matplotlib`. Dev: `pytest`, `ruff`.

## 6. Interface e identidade visual

Duas colunas, fundo claro, cartões brancos, texto cinza escuro, **laranja como destaque**, cantos arredondados, visual minimalista e profissional. Nenhum componente deve ter cor hardcoded fora de `tema.py`.

Melhorias já feitas na Fase 1 (sem alterar a identidade):
- Sidebar agrupada por categoria com cabeçalhos (os grupos da seção 7: Médias, Proporções, Categóricos, Não paramétricos, ANOVA, Regressão, Diagnóstico).
- Dropdowns de variável filtrados pelo tipo exigido pelo teste (`core/validacao.colunas_por_tipo` sobre os perfis de `core/tipos.py`, calculados uma vez na leitura).
- Mensagens de erro amigáveis em português via `page.show_dialog(ft.SnackBar(...))`, nunca traceback na tela.
- Tabela de prévia: estado vazio separado do estado com dados.

## 7. Catálogo de testes (ordem de implementação)

Implementar **nesta ordem**, um de cada vez.

| # | id | Teste | Grupo | Backend sugerido | Entrada principal |
|---|----|-------|-------|------------------|-------------------|
| 1 | `teste_t_1am` | Teste t (uma amostra) | Médias | `scipy.stats.ttest_1samp` | 1 coluna numérica + μ0 |
| 2 | `teste_t_2am` | Teste t (duas amostras) | Médias | `scipy.stats.ttest_ind` | numérica + grupo (2 níveis); Welch/pooled |
| 3 | `teste_t_pareado` | Teste t (pareado) | Médias | `scipy.stats.ttest_rel` | 2 colunas numéricas pareadas |
| 4 | `teste_z_1prop` | Teste Z (uma proporção) | Proporções | `statsmodels.stats.proportion.proportions_ztest` | coluna binária + p0 |
| 5 | `teste_z_2prop` | Teste Z (duas proporções) | Proporções | `proportions_ztest` | binária + grupo (2 níveis) |
| 6 | `qui_quadrado` | Qui-quadrado | Categóricos | `chi2_contingency` / `chisquare` | independência (2 cat.) ou aderência |
| 7 | `fisher` | Teste exato de Fisher | Categóricos | `scipy.stats.fisher_exact` | tabela 2×2 |
| 8 | `mcnemar` | McNemar | Categóricos | `statsmodels.stats.contingency_tables.mcnemar` | 2 binárias pareadas |
| 9 | `teste_sinal` | Teste do sinal | Não paramétricos | `scipy.stats.binomtest` | 1 coluna (mediana) ou 2 pareadas |
| 10 | `wilcoxon` | Wilcoxon | Não paramétricos | `scipy.stats.wilcoxon` | 1 ou 2 colunas pareadas |
| 11 | `mann_whitney` | Mann-Whitney U | Não paramétricos | `scipy.stats.mannwhitneyu` | numérica + grupo (2 níveis) |
| 12 | `kruskal_wallis` | Kruskal-Wallis | Não paramétricos | `scipy.stats.kruskal` | numérica + grupo (≥ 2 níveis) |
| 13 | `friedman` | Friedman | Não paramétricos | `scipy.stats.friedmanchisquare` | ≥ 3 colunas pareadas |
| 14 | `anova_1fator` | ANOVA (1 fator) | ANOVA | `scipy.stats.f_oneway` / `statsmodels` | numérica + fator; pós-teste opcional |
| 15 | `anova_2fator` | ANOVA (2 fatores) | ANOVA | `statsmodels.formula.api.ols` + `anova_lm` | numérica + 2 fatores (± interação) |
| 16 | `regres_linear` | Regressão Linear | Regressão | `statsmodels.api.OLS` | y + 1..n preditores |
| 17 | `regres_logit` | Regressão Logística | Regressão | `statsmodels.api.Logit` | y binário + preditores; odds ratio |
| 18 | `durbin_watson` | Durbin-Watson | Diagnóstico | `statsmodels.stats.stattools.durbin_watson` | resíduos da regressão linear |
| 19 | `breusch_pagan` | Breusch-Pagan | Diagnóstico | `statsmodels.stats.diagnostic.het_breuschpagan` | resíduos + preditores |
| 20 | `white` | White | Diagnóstico | `statsmodels.stats.diagnostic.het_white` | resíduos + preditores |
| 21 | `vif` | VIF | Diagnóstico | `statsmodels.stats.outliers_influence.variance_inflation_factor` | preditores |

### Lista oficial de ids (gerada por `core/registry.py` como `testes_hipotese`)

```python
testes_hipotese = [
    # Médias
    ("teste_t_1am", "Teste t (uma amostra)"),
    ("teste_t_2am", "Teste t (duas amostras)"),
    ("teste_t_pareado", "Teste t (pareado)"),

    # Proporções
    ("teste_z_1prop", "Teste Z (uma proporção)"),
    ("teste_z_2prop", "Teste Z (duas proporções)"),

    # Categóricos
    ("qui_quadrado", "Qui-quadrado"),
    ("fisher", "Teste exato de Fisher"),
    ("mcnemar", "McNemar"),

    # Não paramétricos
    ("teste_sinal", "Teste do sinal"),
    ("wilcoxon", "Wilcoxon"),
    ("mann_whitney", "Mann-Whitney U"),
    ("kruskal_wallis", "Kruskal-Wallis"),
    ("friedman", "Friedman"),

    # ANOVA
    ("anova_1fator", "ANOVA (1 fator)"),
    ("anova_2fator", "ANOVA (2 fatores)"),

    # Regressão / diagnóstico
    ("regres_linear", "Regressão Linear"),
    ("regres_logit", "Regressão Logística"),
    ("durbin_watson", "Durbin-Watson"),
    ("breusch_pagan", "Breusch-Pagan"),
    ("white", "White"),
    ("vif", "VIF"),
]
```

Os ids e rótulos **não mudam** (são a fonte da verdade). Só deixam de ser duplicados.

### Notas por grupo

- **Médias:** hipótese alternativa bilateral/maior/menor. No t de duas amostras, Welch (padrão) e variâncias iguais. Reportar IC da diferença e d de Cohen.
- **Proporções:** informar a codificação do "sucesso". Avisar quando `n·p` ou `n·(1−p)` < 5.
- **Categóricos:** qui-quadrado avisa quando há frequências esperadas < 5 e sugere Fisher; reportar V de Cramér. McNemar: exato vs. com correção.
- **Não paramétricos:** tratar empates e zeros (Wilcoxon, sinal) de forma explícita e documentada.
- **ANOVA:** tabela completa; pós-teste (Tukey) opcional quando significativa; documentar o tipo de soma de quadrados (II/III) em dados desbalanceados.
- **Regressão e diagnóstico:** Durbin-Watson, Breusch-Pagan, White e VIF **operam sobre um modelo linear ajustado**. O controller permite reutilizar o último modelo de `regres_linear` ou ajustar um novo a partir de y e preditores.

### Pressupostos
Cada teste declara seus pressupostos e a UI os mostra na aba Análise como avisos não bloqueantes (normalidade, homogeneidade de variâncias, independência, tamanho mínimo, frequências esperadas). Os testes de pressuposto já presentes no app são Durbin-Watson (independência dos resíduos), Breusch-Pagan e White (homocedasticidade) e VIF (multicolinearidade).

## 8. Plano de execução por fases

**Fase 0 — Linha de base** ✅ concluída
Rodar o app e conferir os 12 problemas da seção 3. Escrever testes de caracterização do que já funciona (`importar_dados`, lista de testes, montagem da UI sem exceção). Corrigir `requirements.txt` e instalar o ambiente do zero para provar que roda.

**Fase 1 — Reestruturação (sem lógica estatística)** ✅ concluída
Criar a estrutura da seção 4 seguindo o mapa de migração: `tema.py` com paleta única, `state.py`, `controller.py`, `core/base.py`, `registry.py`, `painel_abas.py` com API própria (fim dos índices), `campos.py` com o estilo único. Corrigir os itens 1–8 da seção 3. O app deve abrir e parecer idêntico, agora com estados vazios corretos e o botão Executar conectado ao controller.

Pendências da Fase 1 resolvidas na Fase 3: `Executar` roda via `page.run_thread` (com trava contra cliques repetidos) e `core/interpretacao.py` existe.

**Fase 2 — Dados** ✅ concluída
`core/io.py`: CSV com detecção de separador/encoding/decimal, XLSX via `openpyxl`, detecção de tipos (numérica, categórica, binária), erros tratados e exibidos ao usuário. Prévia limitada a 100 linhas. Testes: arquivo válido, vazio, com NaN, colunas mistas, encoding errado, extensão inválida.

Regras de leitura implementadas (detalhes na docstring de `core/io.py`):
- **Encoding:** BOM UTF-8/UTF-16 → UTF-8 estrito → cp1252 → latin-1.
- **Separador:** `;`, `,`, tab ou `|`, o que dá o mesmo número de campos (≥ 2) no cabeçalho e na maioria das linhas; nenhum → arquivo de uma coluna.
- **Decimal/milhar:** separador `,` implica decimal `.`; senão decimal `,` quando "1,5"/"1.234,5" predominam sobre "1.5" ("1.234" isolado é ambíguo e vale como ponto decimal). Milhar `.` só com decimal `,` e sem nenhum "1.5" na amostra.
- **Tipos** (`core/tipos.py`): numérica = dtype numérico não booleano; binária = 2 valores distintos; categórica = não numérica, ou numérica discreta (só inteiros, inclusive float com NaN) com até 10 níveis. Uma coluna pode ter mais de um papel.
- **Erros** (`ErroLeitura`, mensagem pronta em pt-BR): vazio, só cabeçalho, inexistente/bloqueado, binário, linha com colunas a mais (com número da linha), XLSX inválido, extensão não suportada. **Avisos:** coluna que mistura números e texto; XLSX com várias planilhas (lê a primeira). Colunas sem nome e vazias (separador sobrando) são descartadas.
- A prévia mostra até 100 linhas, com NaN vazio e decimais com vírgula. Ler arquivos grandes ainda acontece na thread da UI (otimizar na Fase 5, se medido como lento).

**Fase 3 — Testes de hipótese (um por um)**
Itens 1–15 da seção 7, cada um com o checklist da seção 9.

Andamento: **5/15** (Médias e Proporções completos)
- ✅ `teste_t_1am` (`core/testes/medias.py`, testes em `tests/core/test_medias.py`): card com Wilcoxon de x − μ₀; histograma com x̄ e μ₀.
- ✅ `teste_t_2am` (`tests/core/test_medias_2am.py`): coluna numérica + coluna de grupo com exatamente 2 níveis (`coluna_binaria`); grupo 1 = primeiro nível em ordem crescente; Welch (padrão) ou pooled; card com Mann-Whitney; boxplot por grupo.
- ✅ `teste_t_pareado` (`tests/core/test_medias_pareado.py`): duas colunas numéricas pareadas na mesma linha (d = medida 1 − medida 2); linhas incompletas descartadas com aviso; card com Wilcoxon das diferenças; histograma das diferenças.
- ✅ `teste_z_1prop` (`core/testes/proporcoes.py`, testes em `tests/core/test_proporcoes.py`): coluna binária + valor de "sucesso" (parâmetro `nivel` dependente da coluna); erro padrão com p₀ (teste de escore); IC de Wilson; h de Cohen; card com binomial exato; aviso se n·p₀ ou n·(1 − p₀) < 5; barras de proporções com p₀.
- ✅ `teste_z_2prop` (`tests/core/test_proporcoes_2p.py`): resposta binária + sucesso + grupo com 2 níveis; z com proporção combinada; IC de Wald (não combinado) para p₁ − p₂; h de Cohen e odds ratio com IC de Woolf (indefinida com célula zero); card com Fisher exato; tabela 2×2 na Análise; barras por grupo com a proporção combinada.
- Próximo: `qui_quadrado`.

Padrão estabelecido pelos testes já implementados (seguir nos próximos):
- `parametros()` inclui a hipótese alternativa como `opcao` com rótulos matemáticos (`μ ≠ μ₀`...) mapeados para o `alternative` do scipy; padrão bilateral.
- `validar()` devolve mensagens prontas; `executar()` chama `validar()` e lança `ErroValidacao` se houver erro.
- `ResultadoTeste`: `estatisticas` com chaves técnicas (`t`, `gl`, `p_valor`, `ic_inferior`...), `tabelas["Resumo"]` com colunas `Medida`/`Valor` já formatadas em pt-BR, `figuras` via `core/figuras.py`, `avisos` não bloqueantes e `comparacao` com as três alternativas na ordem ≠, >, <.
- Interpretação via `core/interpretacao.interpretar` (cita α, p, H₀, H₁ e a conclusão no contexto).
- Validações comuns em `core/validacao.py` (`erros_coluna`, `erro_opcao`, `erro_alfa`, `erro_numero`) para as mensagens ficarem iguais entre testes.
- Docstring da classe documenta as escolhas (ddof, zeros, correção, exato vs. assintótico, erro padrão).
- Avisos técnicos do scipy/statsmodels (em inglês) não chegam ao usuário: suprimir pontualmente e emitir aviso equivalente em pt-BR.
- **Card de comparação:** só quando há um equivalente natural (não paramétrico ou exato); testes sem essa estrutura (qui-quadrado, ANOVA, diagnósticos...) não sobrescrevem `comparacao_inicial()` e devolvem `comparacao=None` — decisão do autor.
- Valores padrão e números exibidos em pt-BR (vírgula decimal); os campos numéricos aceitam vírgula ou ponto.
- Referências nos testes: fórmula manual (numpy) + outra biblioteca (statsmodels) ou enumeração exata; fonte documentada no topo do arquivo de teste. Cálculos de referência reutilizáveis ficam em `tests/referencias.py`.

**Fase 4 — Regressão e diagnósticos**
Itens 16–21, com o fluxo "ajustar modelo → diagnosticar".

**Fase 5 — Otimização e acabamento**
Perfilar (`cProfile`/`time`) e otimizar só o que estiver medido como lento. Revisar mensagens, textos de interpretação, README.

## 9. Checklist de "pronto" para cada teste

1. `ParametroSpec` definidos e renderizados dinamicamente na aba Parâmetros (sem função de formulário específica na UI).
2. `validar()` cobre: coluna inexistente, tipo errado, NaN, n mínimo, número de grupos incorreto, variância zero, campo numérico inválido.
3. `executar()` retorna `ResultadoTeste` completo (estatística, p-valor, decisão, interpretação, avisos, `comparacao` quando fizer sentido).
4. Interpretação em português, gerada a partir do p-valor e do α escolhido, citando H0/H1.
5. Aba Visualização com ao menos um gráfico adequado (boxplot, histograma com a média de referência, barras de proporções, resíduos vs. ajustados...).
6. **Testes automatizados** (`pytest`):
   - valor de referência conhecido (estatística e p-valor com `pytest.approx(rel=1e-6)`);
   - cada hipótese alternativa e cada opção configurável;
   - casos de borda e entradas inválidas (mensagem de erro, não exceção não tratada);
   - consistência com cálculo independente (fórmula manual com numpy ou outra biblioteca).
7. Teste de integração no controller: carregar dataset de exemplo → selecionar teste → executar → `ResultadoTeste` sem erro.
8. `ruff` sem avisos e toda a suíte `pytest` passando.
9. Commit `feat(teste): implementa <nome do teste>`.

Só então passe ao próximo teste.

### Estratégia de testes por seção

| Seção | O que validar |
|-------|---------------|
| `core/io` | leitura de CSV (`,` e `;`, decimal `,`, encodings) e XLSX, detecção de tipos, NaN, arquivos inválidos |
| `core/validacao` | cada regra isoladamente |
| Cada teste em `core/testes` | checklist acima |
| `registry` | os 21 ids registrados, sem duplicatas, rótulos idênticos à lista oficial, ordem preservada |
| `controller` | fluxo carregar → selecionar → executar → resultado; arquivo carregado depois da seleção atualiza o painel; erros tratados |
| `app/ui` | montagem sem exceção; sidebar com todos os testes; as 3 abas existem; estados vazios corretos; `CardComparacaoTestes.atualizar_p_values` altera os textos esperados |

Datasets de teste pequenos e determinísticos (semente fixa) em `tests/conftest.py` ou `tests/data/`.

## 10. Otimização

- **Medir antes de otimizar.**
- Cálculos pesados fora da thread da UI, com indicador de progresso real (usar o mecanismo de tarefas/threads do Flet 0.86.2; confirmar na documentação). A UI nunca congela na execução.
- Prévia da tabela limitada a N linhas (hoje 100; manter). Evitar recriar o `DataTable` inteiro quando só os dados mudam.
- Operações vetorizadas com pandas/numpy; nada de loops sobre linhas fora da prévia.
- `import` tardio de `statsmodels` e `matplotlib` para abrir o app mais rápido.
- Guardar o dataset em `state.py`; nunca reler o arquivo a cada execução.
- Se usar matplotlib, backend `Agg`, converter a figura em imagem e fechar com `plt.close`.
- Eliminar duplicação: estilos de campo (`campos.py`), validações (`validacao.py`) e textos (`interpretacao.py`).

## 11. Convenções de código

- Python 3.10+, type hints em funções públicas, docstrings curtas em português.
- Sem `import *`. Sem `print` de debug (usar `logging`).
- Módulos em `snake_case`; identificadores coerentes com os ids da seção 7; textos de interface em pt-BR.
- Números exibidos com formatação brasileira (vírgula decimal) quando fizer sentido; internamente `float`.
- Formatação e lint com `ruff`. Funções curtas, uma responsabilidade por função.
- Callbacks do Flet só chamam o `controller`; sem regra de negócio neles.
- Exceções de domínio (`ErroValidacao`, `ErroExecucao`) convertidas em mensagens amigáveis.
- Referências a controles por atributo/objeto (ex.: `painel_abas.definir_parametros(...)`), nunca por índice na árvore.

## 12. Comandos

```bash
# ambiente
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

# rodar o app
python main.py

# testes e qualidade
pytest -q
pytest tests/core/test_registry.py -q  # apenas um arquivo
ruff check . && ruff format .
```

No Windows (PowerShell 5.1), passe mensagens de commit com `git commit -F arquivo.txt`: aspas duplas dentro de `-m` são quebradas pelo PowerShell ao chamar executáveis nativos.

## 13. Decisões em aberto (confirmar com o autor antes de implementar)

- Qual equivalente não paramétrico aparece no card para cada teste e quais testes não terão card. **Decidido:** t de uma amostra ↔ Wilcoxon; t de duas amostras ↔ Mann-Whitney (Welch como padrão, opção pooled; entrada só no formato coluna numérica + grupo de 2 níveis). t pareado ↔ Wilcoxon das diferenças (entrada: duas colunas pareadas na mesma linha; linhas incompletas descartadas com aviso). Z de uma proporção ↔ binomial exato (entrada: coluna binária + valor de sucesso escolhido no formulário; erro padrão com p₀; IC de Wilson). Z de duas proporções ↔ Fisher exato (proporção combinada no z, IC de Wald não combinado; efeitos: diferença, h de Cohen e odds ratio). **Regra geral do autor:** card só quando a comparação for possível e útil; caso contrário, sem card. Ainda em aberto (sugestão): ANOVA 1 fator ↔ Kruskal-Wallis.
- ~~O card continua nas abas Parâmetros e Análise ou fica só em Análise?~~ Decidido na Fase 1: **duas instâncias** (Parâmetros e Análise), para preservar o visual. Pode ser revisto depois.
- ~~Gráficos nativos do Flet ou imagens do matplotlib?~~ Decidido na Fase 3: **nativos do Flet, simples e minimalistas**, desenhados com `flet.canvas` (no Flet 0.86.2 `BarChart`/`LineChart` saíram do pacote principal para a extensão `flet-charts`; o canvas é do núcleo e não exige dependência nova). matplotlib não é usado.
- Pós-testes (Tukey, Dunn) e pressupostos extras (Shapiro-Wilk, Levene) como funcionalidade adicional.
- Formatos de arquivo além de CSV/XLSX e exportação de resultados (PDF/HTML/CSV).
