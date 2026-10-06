# smartETL — Testes de Hipótese

## 1. Visão geral

Aplicativo desktop de **processamento e análise de dados** com foco em **testes de hipótese** (paramétricos e não paramétricos), ANOVA, regressão e diagnósticos de pressupostos. O usuário carrega um arquivo (CSV/XLSX), escolhe um teste na barra lateral, configura os parâmetros, executa e vê resultado, interpretação e gráficos.

- **Linguagem:** Python 3.10+ (o código atual já usa `list[dict] | None` em assinaturas; alvo: 3.11)
- **Interface:** Flet **`0.86.2`** (versão fixada em `requirements.txt`)
- **Estatística:** `scipy.stats`, `statsmodels`, `pandas`, `numpy`
- **Idioma da interface e das interpretações:** português do Brasil
- **Estado atual:** interface montada e funcional até o carregamento de arquivo; **só existe o formulário de parâmetros do `teste_t_1am`, e nenhum cálculo estatístico está implementado**.

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

### Arquivos (estrutura plana na raiz)

| Arquivo | O que faz hoje |
|---------|----------------|
| `smartetl_app.py` | Entrada (`ft.run(main)`), paleta de cores, helpers de layout, sidebar, cabeçalho, `DataTable` de prévia, abas e callbacks (`selecionar_arquivo`, `carregar_base`, `mostrar_testes`). Tudo dentro de `main(page)`. |
| `utils.py` | Lista `testes_hipotese` + `criar_sidebar_testes` (lista de testes com ícone de rádio, seleção visual). |
| `conteiner_parametros.py` | `teste_selecionado(nome, colunas)` (despacho por `globals()`) e `teste_t_1am(colunas)` (formulário + card). Retorna as 3 abas: `(parametro, analise, visual)`. |
| `ConteinerTestes.py` | `CardComparacaoTestes` (`ft.Container`): card que compara p-valores de dois testes (padrão: "t Student" × "Wilcoxon") para as hipóteses alternativas ≠, >, <. Tem `atualizar_p_values` e `atualizar_p_value_linha`. |
| `load_table.py` | `importar_dados(path, extensao)` com `read_excel`/`read_csv`. |
| `requirements.txt` | Apenas `flet==0.86.2`. |
| `README.md` | **Desatualizado** (descreve o protótipo antigo com "Gerar relatório", Histórico, Logs, gráfico de barras). |

### Comportamento atual (preservar)

- Janela 1440×900 (mín. 1150×720), centralizada; fundo `BG`, sem padding na página.
- Sidebar de 260 px: logo "smartETL", subtítulo, botão **Arquivo** (fundo laranja suave), lista rolável de testes (primeiro marcado por padrão), botão **Executar teste** (laranja, ícone de balança).
- Área principal: título "Processamento de dados", subtítulo "Testes de Hipótese | paramétricos e não paramétricos", tabela de prévia (altura 320, rolagem horizontal e vertical) e painel de 3 abas (**Parâmetros**, **Análise**, **Visualização**).
- Tabela vazia: 20 colunas `column1..20` e 12 linhas em branco. Com arquivo: no máximo **100 linhas**, completando até 20 colunas com `columnN` vazias.
- Seleção de arquivo: `FilePicker`, extensões `xlsx` e `csv`, um arquivo.
- Formulário do `teste_t_1am`: dropdown "Variável", campo "Média Hipotética", dropdown "Nível de significância (α)" (0.01 / 0.05 / 0.10, padrão 0.05), e o card de comparação ao lado.

### Problemas conhecidos (corrigir nas Fases 0 e 1)

1. **Abas ficam em "Processando..." (é o que aparece no print).** `mostrar_testes` retorna sem fazer nada quando não há arquivo carregado, e o primeiro teste aparece marcado sem disparar `on_selecionar`. Além disso, carregar o arquivo *depois* de escolher o teste não atualiza o painel.
2. **Botão "Executar teste" sem `on_click`.**
3. **Despacho frágil:** `teste_selecionado` usa `globals().get(nome)` e, se a função não existe, **cai no formulário do `teste_t_1am`**. Ou seja, hoje qualquer teste mostra o formulário do t de uma amostra. Há `print` de debug.
4. **Mesma instância de `CardComparacaoTestes` colocada em dois lugares** (dentro de `parametro` e de `analise`). Um controle Flet deve ter um único pai; use instâncias separadas ou mantenha o card só na aba Análise.
5. **Acesso por índice à árvore de controles** em `mostrar_testes` (`tabs.content.controls[1].controls[0].content = ...`). Qualquer mudança no layout quebra silenciosamente.
6. **Formulário sem referência aos campos:** não há como ler os valores digitados/selecionados. O dropdown lista todas as colunas (inclusive não numéricas). "Média Hipotética" não tem validação. Os estilos de `Dropdown`/`TextField` estão repetidos linha a linha.
7. **Três laranjas diferentes:** `#FF6A1A` (app), `#FF8A3D` (`utils.py`) e `#F97316` (card). `COR_BORDA` também está duplicada com valores distintos.
8. **Imports e código morto:** `from utils import *` e `from load_table import *`; `import pandas as pd` não usado em `smartetl_app.py`; em `load_table.py` há `import flet as flet` sem uso e `pandas` importado duas vezes; constantes e helpers sem uso (`ALIGN_*`, `GREEN`, `section_title`, `card_container`, `radius`, `page.fonts = {}`).
9. **Leitura de dados sem robustez:** `read_csv` sem separador/encoding/decimal (CSV brasileiro costuma ter `;`, vírgula decimal e `latin-1`); `selecionar_arquivo` sem `try/except` (erro some no callback, sem mensagem ao usuário); `print` de debug.
10. **Dados reais misturados com placeholders** em `carregar_base` (colunas `columnN` e células `"   "` para completar 20 colunas). Separar "estado vazio" de "dados".
11. **Dependências incompletas:** `requirements.txt` não lista `pandas`, `openpyxl` (necessário para `read_excel`), `numpy`, `scipy`, `statsmodels`. O README diz Python 3.9, mas o código exige 3.10+.
12. **Nomes:** `ConteinerTestes.py` (CamelCase e "Conteiner") e `conteiner_parametros.py`. Padronizar para `snake_case` e `container`.

## 4. Arquitetura alvo

```
smartetl/
├── main.py                       # ponto de entrada: ft.run(main)
├── CLAUDE.md
├── README.md                     # reescrito (ver seção 12)
├── requirements.txt / pyproject.toml
├── app/
│   ├── ui/
│   │   ├── tema.py               # UMA paleta (cores, raios, tipografia)
│   │   ├── helpers.py            # pad(), border_all(), border_only()...
│   │   ├── sidebar.py            # logo, Arquivo, lista de testes, Executar
│   │   ├── tabela_dados.py       # prévia do dataset (estado vazio vs. dados)
│   │   ├── painel_abas.py        # Parâmetros / Análise / Visualização (API, sem índices)
│   │   ├── painel_parametros.py  # formulário gerado a partir de ParametroSpec
│   │   └── componentes/
│   │       ├── campos.py         # dropdown/campo com o estilo único atual
│   │       └── card_comparacao.py# (era ConteinerTestes.py)
│   ├── controller.py             # liga UI ↔ core (sem lógica estatística)
│   └── state.py                  # df, teste selecionado, parâmetros, último resultado
├── core/
│   ├── base.py                   # TesteBase, ParametroSpec, ResultadoTeste
│   ├── registry.py               # id → classe; gera a lista da sidebar
│   ├── io.py                     # (era load_table.py) leitura CSV/XLSX robusta
│   ├── validacao.py
│   ├── interpretacao.py          # textos em pt-BR a partir de p-valor e α
│   └── testes/
│       ├── medias.py
│       ├── proporcoes.py
│       ├── categoricos.py
│       ├── nao_parametricos.py
│       ├── anova.py
│       └── regressao.py
└── tests/
    ├── conftest.py               # datasets pequenos, determinísticos
    ├── core/
    └── app/
```

### Mapa de migração (arquivo atual → destino)

| Atual | Destino |
|-------|---------|
| `smartetl_app.py` (paleta, helpers) | `app/ui/tema.py`, `app/ui/helpers.py` |
| `smartetl_app.py` (sidebar, cabeçalho, tabela, abas) | `app/ui/sidebar.py`, `tabela_dados.py`, `painel_abas.py`; montagem em `main.py` |
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

1. **Arquivo** → `controller.carregar_arquivo()` → `core/io.py` → `state.df` → atualiza prévia e o painel de parâmetros (se já houver teste selecionado).
2. **Seleção de teste** (inclusive o inicial) → `controller.selecionar_teste(id)` → painel de parâmetros renderiza o formulário a partir de `teste.parametros()`.
3. **Executar teste** → coleta valores dos campos → `teste.validar(df, params)` → `teste.executar(df, params)` (fora da thread da UI) → preenche **Análise** e **Visualização**.
4. Estados vazios claros em vez de "Processando...": sem arquivo ("Carregue um arquivo para começar"), sem execução ("Configure os parâmetros e clique em Executar teste"). "Processando..." só durante a execução real.

### Contrato de um teste

```python
@dataclass
class ParametroSpec:
    nome: str
    rotulo: str                 # texto exibido na UI (pt-BR)
    tipo: Literal["coluna_numerica", "coluna_categorica", "multi_coluna",
                  "numero", "alfa", "opcao", "booleano"]
    padrao: Any = None
    opcoes: list[str] | None = None
    obrigatorio: bool = True

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
```

### Card de comparação (`CardComparacaoTestes`)

Mantenha o componente: ele mostra, para cada hipótese alternativa (≠, >, <), o p-valor do teste paramétrico (laranja, à esquerda) ao lado do p-valor do equivalente não paramétrico (roxo, à direita). A nota do rodapé diz que os valores são calculados após a execução. Ele deve ser alimentado com `atualizar_p_values(...)` ao final de `executar`, via `ResultadoTeste.comparacao`:

```python
@dataclass
class ComparacaoPValores:
    titulo_esquerda: str                  # "t Student"
    titulo_direita: str                   # "Wilcoxon"
    linhas: list[tuple[float, float]]     # (p_param, p_nao_param) para ≠, >, <
```

Nem todo teste tem essa estrutura (ex.: qui-quadrado, ANOVA, diagnósticos). O teste declara `comparacao = None` e a UI exibe só a tabela/estatísticas. Quando for usada, o texto da hipótese de cada linha deve refletir o teste (hoje é fixo em "μ¹ ≠ μ²"; para o t de uma amostra deve ser "μ ≠ μ0" etc.).

## 5. Stack estatística

Pode-se delegar o cálculo a `scipy`/`statsmodels`, mas o *wrapper* deve mapear corretamente parâmetros (hipótese alternativa, correção de continuidade, `ddof`, exato vs. assintótico), e isso **precisa ser testado**.

Dependências (todas em `requirements.txt`, com versões fixadas): `flet==0.86.2`, `pandas`, `numpy`, `scipy`, `statsmodels`, `openpyxl`; opcional `matplotlib`. Dev: `pytest`, `ruff`.

## 6. Interface e identidade visual

Duas colunas, fundo claro, cartões brancos, texto cinza escuro, **laranja como destaque**, cantos arredondados, visual minimalista e profissional. Nenhum componente deve ter cor hardcoded fora de `tema.py`.

Melhorias previstas (sem alterar a identidade):
- Agrupar a sidebar por categoria com cabeçalhos (Médias, Proporções, Categóricos, Não paramétricos, ANOVA, Regressão e diagnóstico).
- Dropdowns de variável filtrados pelo tipo exigido pelo teste (numérica, categórica, binária).
- Mensagens de erro amigáveis em português, nunca traceback na tela (usar `SnackBar`/banner do Flet 0.86.2).
- Tabela de prévia: estado vazio separado do estado com dados; sem colunas/células fantasmas misturadas aos dados reais.

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

### Lista oficial de ids (já existente em `utils.py`; passa a ser gerada pelo `registry.py`)

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

**Fase 0 — Linha de base**
Rodar o app e conferir os 12 problemas da seção 3. Escrever testes de caracterização do que já funciona (`importar_dados`, lista de testes, montagem da UI sem exceção). Corrigir `requirements.txt` e instalar o ambiente do zero para provar que roda.

**Fase 1 — Reestruturação (sem lógica estatística)**
Criar a estrutura da seção 4 seguindo o mapa de migração: `tema.py` com paleta única, `state.py`, `controller.py`, `core/base.py`, `registry.py`, `painel_abas.py` com API própria (fim dos índices), `campos.py` com o estilo único. Corrigir os itens 1–8 da seção 3. O app deve abrir e parecer idêntico, agora com estados vazios corretos e o botão Executar conectado ao controller.

**Fase 2 — Dados**
`core/io.py`: CSV com detecção de separador/encoding/decimal, XLSX via `openpyxl`, detecção de tipos (numérica, categórica, binária), erros tratados e exibidos ao usuário. Prévia limitada a 100 linhas. Testes: arquivo válido, vazio, com NaN, colunas mistas, encoding errado, extensão inválida.

**Fase 3 — Testes de hipótese (um por um)**
Itens 1–15 da seção 7, cada um com o checklist da seção 9. Começar pelo `teste_t_1am`, que já tem formulário e card prontos para ligar ao cálculo.

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

# rodar o app (hoje)
python smartetl_app.py
# rodar o app (após a Fase 1)
python main.py

# testes e qualidade
pytest -q
pytest tests/core/test_medias.py -q    # apenas um grupo
ruff check . && ruff format .
```

Reescrever o `README.md` na Fase 1: remover as descrições do protótipo antigo (Histórico, Logs, "Gerar relatório", gráfico de barras, dados fictícios), atualizar requisitos (Python 3.10+) e estrutura.

## 13. Decisões em aberto (confirmar com o autor antes de implementar)

- Qual equivalente não paramétrico aparece no card para cada teste (sugestão: t 1 amostra e t pareado ↔ Wilcoxon; t 2 amostras ↔ Mann-Whitney; ANOVA 1 fator ↔ Kruskal-Wallis) e quais testes não terão card.
- O card continua nas abas Parâmetros e Análise ou fica só em Análise? (hoje a mesma instância é usada nas duas)
- Gráficos nativos do Flet ou imagens do matplotlib na aba Visualização.
- Pós-testes (Tukey, Dunn) e pressupostos extras (Shapiro-Wilk, Levene) como funcionalidade adicional.
- Formatos de arquivo além de CSV/XLSX e exportação de resultados (PDF/HTML/CSV).
