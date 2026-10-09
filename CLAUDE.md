# smartETL — Testes de Hipótese

## 1. Visão geral

Aplicativo desktop de **processamento e análise de dados** com foco em **testes de hipótese** (paramétricos e não paramétricos), ANOVA, regressão e diagnósticos de pressupostos. O usuário carrega um arquivo (CSV/XLSX), escolhe um teste na barra lateral, configura os parâmetros, executa e vê resultado, interpretação e gráficos.

- **Linguagem:** Python 3.10+ (o código atual já usa `list[dict] | None` em assinaturas; alvo: 3.11)
- **Interface:** Flet **`0.86.2`** (versão fixada em `requirements.txt`)
- **Estatística:** `scipy.stats`, `statsmodels`, `pandas`, `numpy`
- **Idioma da interface e das interpretações:** português do Brasil
- **Estado atual:** Fases 0 a 5 concluídas. Arquitetura modular da seção 4 em funcionamento (`python main.py`), com leitura robusta de CSV/XLSX e detecção de tipos. Fase 3 concluída (15/15): **grupos Médias, Proporções, Categóricos, Não paramétricos e ANOVA completos** (`teste_t_1am`, `teste_t_2am`, `teste_t_pareado`, `teste_z_1prop`, `teste_z_2prop`, `qui_quadrado`, `fisher`, `mcnemar`, `teste_sinal`, `wilcoxon`, `mann_whitney`, `kruskal_wallis`, `friedman`) `anova_1fator` e `anova_2fator`, cumprindo o checklist da seção 9. Fase 4 concluída (2/2): **`regres_linear` e `regres_logit`** (valores conferidos contra o R, aba Simulação). **Os 19 testes do catálogo estão implementados** (inclusive as correlações, acrescentadas depois). Fase 5 (otimização e acabamento) concluída. Seção 13: Shapiro-Wilk como aviso nos testes t e nas ANOVAs e **exportação da análise em PDF** (`core/relatorio.py`) implementados.

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

- **Splash** (`app/ui/splash.py`): `main.py` só importa Flet e a splash, mostra a tela de abertura e retorna; o carregamento pesado roda em `page.run_thread` (`carregar_aplicativo`) e troca a splash pela tela principal com esmaecimento (`AnimatedSwitcher`), após no mínimo 1,2 s. No Flet 0.86.2, o que um `main` síncrono adiciona só chega à janela quando ele retorna.
- **Decisão colorida** em todos os testes: "Rejeita H₀" em verde suave `#3F9B6B` (`tema.DECISAO_REJEITA`) e "Não rejeita H₀" em vermelho terracota suave `#C8705F` (`tema.DECISAO_NAO_REJEITA`) — invertidas em 2026-10-08 a pedido do autor (rejeitar = o teste encontrou o efeito), via `tela_principal.cor_da_decisao`.
- **Simulação com inteiros:** preditor com todos os valores inteiros (`CampoSimulacao.inteiro`) começa na média arredondada, mostra números sem casas e o slider anda de 1 em 1 (até 500 posições).
- **Barra de tarefas (Windows):** a janela é do cliente Flet (`flet.exe`), cuja descrição é "Flet description"; `app/identidade_windows.py` (pywin32) grava nela ID próprio, nome "smartETL", ícone e o comando de reabrir (smartETL.exe ou `pythonw main.py`), para o menu e o "Fixar na barra de tarefas" ficarem certos. **O ID é derivado do comando de reabrir** (`smartETL.Desktop.<sha1>` / `smartETL.Codigo.<sha1>`): o Windows guarda o 1º comando de cada ID num atalho em `User Pinned/ImplicitAppShortcuts` (que sobra depois de desafixar) e ignora mudanças; em 2026-10-08 um teste manual gravou "python.exe teste" num ID fixo e o ícone fixado passou a abrir só um console. Nunca gravar essas propriedades com outro comando num ID já usado. Linha da aba ativa em laranja (`TabBar.indicator_color`).
- Janela 1440×900 (mín. 1150×720), **centralizada** (`page.run_task(page.window.center)` — `Window.center` é assíncrono no Flet 0.86.2); fundo `tema.FUNDO`, sem padding na página.
- Sidebar de 260 px: logo "smartETL", subtítulo, botão **Arquivo** (fundo laranja suave), lista rolável de testes **agrupada por categoria** (cabeçalhos em maiúsculas, primeiro teste marcado por padrão), botão **Executar teste** (laranja, ícone de balança).
- Área principal: título "Processamento de dados", subtítulo "Testes de Hipótese | paramétricos e não paramétricos", tabela de prévia (altura 320, rolagem horizontal e vertical) e painel de 3 abas (**Parâmetros**, **Análise**, **Visualização**).
- Tabela vazia: 20 colunas `column1..20` (cinza claro) e 12 linhas em branco. Com arquivo: **só as colunas reais**, no máximo **100 linhas**, NaN exibido vazio.
- Seleção de arquivo: `FilePicker` em `page.services`, extensões `xlsx` e `csv`, um arquivo. Erros de leitura viram `SnackBar` vermelho em português.
- Estados vazios: sem arquivo → "Carregue um arquivo para começar."; teste sem implementação → "O <teste> ainda não está disponível nesta versão."; sem execução → "Configure os parâmetros e clique em Executar teste." "Processando..." só durante a execução.
- Formulário do `teste_t_1am` (gerado pelos `ParametroSpec`): dropdown "Variável" (só colunas numéricas), campo "Média Hipotética" (aceita vírgula decimal), dropdown "Nível de significância (α)" (0,01 / 0,05 / 0,10, padrão 0,05), e o card de comparação ao lado, com p-valores "—" e hipóteses "μ ≠ μ₀", "μ > μ₀", "μ < μ₀". A aba Análise tem uma **segunda instância** do card.

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
├── bases/                       # bases para testar no app (formato brasileiro) + README com o uso de cada uma
├── app/
│   ├── ui/
│   │   ├── tema.py               # UMA paleta (cores, raios, tipografia)
│   │   ├── helpers.py            # pad(), border_all(), border_only(), esta_na_pagina()
│   │   ├── sidebar.py            # logo, Arquivo, lista de testes agrupada, Executar
│   │   ├── tabela_dados.py       # prévia do dataset (estado vazio vs. dados)
│   │   ├── painel_abas.py        # Parâmetros / Análise / Visualização (+ Simulação)
│   │   ├── painel_parametros.py  # formulário gerado a partir de ParametroSpec
│   │   ├── painel_simulacao.py   # aba Simulação (equação, campos + sliders, previsão)
│   │   ├── tela_principal.py     # monta a tela e implementa a Visao do controller
│   │   ├── graficos.py           # desenha Figura com flet.canvas (nativo, minimalista)
│   │   └── componentes/
│   │       ├── campos.py         # dropdown/campo/checkbox com o estilo único
│   │       └── card_comparacao.py
│   ├── controller.py             # liga UI ↔ core via Protocol Visao (sem Flet, sem estatística)
│   └── state.py                  # df, teste selecionado, parâmetros, último resultado
├── core/
│   ├── base.py                   # contratos (ver abaixo) e exceções de domínio
│   ├── registry.py               # TesteInfo(id, nome, grupo, classe) dos 22 testes
│   ├── io.py                     # carregar_dados → DadosCarregados; ErroLeitura
│   ├── tipos.py                  # PerfilColuna / detectar_tipos (numérica, categórica, binária)
│   ├── interpretacao.py          # decidir (p ≤ α), interpretar, formatar_numero/p_valor em pt-BR
│   ├── figuras.py                # construtores de Figura (histograma, boxplot, barras,
│   │                             #   dispersão, Q-Q, cascata)
│   ├── relatorio.py              # PDF da análise (fpdf2 + gráficos matplotlib/Agg)
│   ├── diagnosticos.py           # GQ e HMC (lmtest), GVIF (car), Lilliefors (nortest),
│   │                             #   faixas internas de DW e VIF
│   └── testes/
│       ├── medias.py             # TesteT1Amostra (só formulário, por enquanto)
│       ├── proporcoes.py
│       ├── categoricos.py
│       ├── nao_parametricos.py
│       ├── anova.py
│       ├── correlacao.py
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
                  "nivel",      # "nivel": um valor de outra coluna (ex.: o "sucesso")
                  "preditores", # caixas com colunas numéricas e categóricas
                  "niveis_referencia",  # uma lista de níveis por categórica marcada em depende_de
                  "ordenacao"]  # opcoes[0] ("Valores ajustados") + colunas numéricas
    padrao: Any = None
    opcoes: list[str] | None = None
    obrigatorio: bool = True
    depende_de: str | None = None  # "nivel"/"niveis_referencia": parâmetro de origem
    ajuda: str | None = None       # texto explicativo exibido abaixo do campo

@dataclass
class Figura:                   # especificação sem Flet; a UI desenha (app/ui/graficos.py)
    tipo: Literal["histograma", "boxplot", "barras", "barras_agrupadas",
                  "dispersao", "cascata"]  # core/figuras + ui/graficos
    titulo: str
    dados: dict[str, Any]       # histograma: bordas, contagens, rotulo_x, referencias
                                # boxplot: rotulo_y, grupos[{rotulo, n, q1, mediana, q3, bigodes, media, outliers}]
                                # barras: rotulo_y, maximo, percentual, categorias[{rotulo, valor}], referencias
                                # barras_agrupadas: rotulo_y, maximo, percentual, series[...], grupos[{rotulo, valores}]
                                # dispersao: rotulo_x/y, x, y, n_total, limites, linhas[...] (Q-Q também)
                                # cascata: rotulo_y, inicio, etapas[{rotulo, valor, de, ate}], final
                                # (cores das séries em tema.GRAFICO_SERIES)
# GrupoFiguras(rotulo, opcoes: dict[str, Figura], padrao): a UI mostra uma por vez, com lista.
# Secao(titulo, nivel, destaque, textos, tabelas, notas, avisos): bloco da Análise; tabelas
#   podem ter DataFrame.attrs["dicas"] (tooltip por coluna) e ["destaques"] (linhas em negrito).

@dataclass
class ResultadoTeste:
    teste_id: str
    estatisticas: dict[str, float]        # ex.: {"t": 2.31, "gl": 29}
    p_valor: float | None
    alfa: float
    decisao: str                           # "Rejeita H0" / "Não rejeita H0"
    interpretacao: str                     # texto em pt-BR
    tabelas: dict[str, pd.DataFrame]       # tabelas extras (ANOVA, coeficientes...)
    figuras: list[Figura | GrupoFiguras]   # para a aba Visualização
    avisos: list[str]                      # pressupostos duvidosos, n pequeno etc.
    comparacao: ComparacaoPValores | None  # alimenta o card (ver abaixo)
    secoes: list[Secao]                    # se houver, a Análise segue esta ordem
    simulacao: Any                         # Simulador (aba Simulação) ou None

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

Nem todo teste tem essa estrutura (ex.: qui-quadrado, Kruskal-Wallis, diagnósticos): o teste não sobrescreve `comparacao_inicial()` (retorna `None`), devolve `comparacao=None` no resultado, e a UI exibe só a tabela/estatísticas. O texto das hipóteses vem do teste (o card prefixa "Hₐ:").

## 5. Stack estatística

Pode-se delegar o cálculo a `scipy`/`statsmodels`, mas o *wrapper* deve mapear corretamente parâmetros (hipótese alternativa, correção de continuidade, `ddof`, exato vs. assintótico), e isso **precisa ser testado**.

Dependências (todas em `requirements.txt`, com versões fixadas): `flet==0.86.2`, `pandas`, `numpy`, `scipy`, `statsmodels`, `openpyxl`, `python-calamine` (leitor rápido de XLSX; o openpyxl é a reserva), `fpdf2` e `matplotlib` (só no relatório em PDF, importados quando ele é gerado). Dev: `pytest`, `ruff`, `pypdf` (lê o PDF nos testes).

## 6. Interface e identidade visual

Duas colunas, fundo claro, cartões brancos, texto cinza escuro, **laranja como destaque**, cantos arredondados, visual minimalista e profissional. Nenhum componente deve ter cor hardcoded fora de `tema.py`.

Melhorias já feitas na Fase 1 (sem alterar a identidade):
- Sidebar agrupada por categoria com cabeçalhos (os grupos da seção 7: Médias, Proporções, Categóricos, Não paramétricos, ANOVA, Correlação, Regressão).
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
| 16 | `regres_linear` | Regressão Linear | Regressão | `statsmodels.api.OLS` | y + 1..n preditores (numéricos e categóricos); pressupostos (BP, Goldfeld-Quandt, Harrison-McCabe, DW, Breusch-Godfrey, VIF/GVIF, normalidade) dentro do teste — especificação completa em `docs/regressao_linear.md` |
| 17 | `regres_logit` | Regressão Logística | Regressão | `statsmodels.api.Logit` | y binário + preditores; odds ratio |
| 18 | `correlacao_pearson` | Correlação de Pearson | Relação | `scipy.stats.pearsonr` | 2 colunas numéricas; card com a Spearman |
| 19 | `correlacao_spearman` | Correlação de Spearman | Relação | `scipy.stats.spearmanr` | 2 colunas numéricas |
| 20 | `information_value` | Information Value (IV) | Relação | WoE/IV (Siddiqi), implementação própria | y binário + evento + colunas X |
| 21 | `correlacao_parcial` | Correlação parcial | Relação | resíduos (= `ppcor::pcor.test`) | X, Y numéricas + 1..k controles; Pearson ou Spearman |
| 22 | `correlacao_kendall` | Correlação de Kendall | Relação | `scipy.stats.kendalltau` | 2 colunas numéricas; card com a Spearman |

Os itens 18 e 19 foram acrescentados depois da Fase 5, a pedido do autor (2026-10-07); na barra lateral o grupo fica entre ANOVA e Regressão. Em 2026-10-08 o grupo **Correlação passou a se chamar Relação** e recebeu os itens 20 a 22: `information_value` (primeiro do grupo), `correlacao_kendall` (depois da Spearman) e `correlacao_parcial` (último do grupo).

Os antigos itens 18–21 (`durbin_watson`, `breusch_pagan`, `white`, `vif`, grupo Diagnóstico) foram **removidos da lista** por decisão do autor (2026-10-06): DW, BP e VIF passaram para dentro da regressão linear.

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

    # Relação
    ("information_value", "Information Value (IV)"),
    ("correlacao_pearson", "Correlação de Pearson"),
    ("correlacao_spearman", "Correlação de Spearman"),
    ("correlacao_kendall", "Correlação de Kendall"),
    ("correlacao_parcial", "Correlação parcial"),

    # Regressão
    ("regres_linear", "Regressão Linear"),
    ("regres_logit", "Regressão Logística"),
]
```

Os ids e rótulos **não mudam** (são a fonte da verdade). Só deixam de ser duplicados.

### Notas por grupo

- **Médias:** hipótese alternativa bilateral/maior/menor. No t de duas amostras, Welch (padrão) e variâncias iguais. Reportar IC da diferença e d de Cohen.
- **Proporções:** informar a codificação do "sucesso". Avisar quando `n·p` ou `n·(1−p)` < 5.
- **Categóricos:** qui-quadrado avisa quando há frequências esperadas < 5 e sugere Fisher; reportar V de Cramér. McNemar: exato vs. com correção.
- **Não paramétricos:** tratar empates e zeros (Wilcoxon, sinal) de forma explícita e documentada.
- **ANOVA:** tabela completa; pós-teste (Tukey) opcional quando significativa; documentar o tipo de soma de quadrados (II/III) em dados desbalanceados.
- **Regressão:** a regressão linear segue `docs/regressao_linear.md` (especificação do autor + decisões): pressupostos antes de tudo na Análise, tabela do modelo, coeficientes agrupados por categórica com a referência explícita, resíduos e Q-Q na Visualização e uma aba **Simulação** própria (só para testes que a oferecem).

### Pressupostos
Cada teste declara seus pressupostos e a UI os mostra na aba Análise como avisos não bloqueantes (normalidade, homogeneidade de variâncias, independência, tamanho mínimo, frequências esperadas). Na regressão linear, os pressupostos são uma seção própria da Análise (heterocedasticidade, autocorrelação, colinearidade e normalidade), com faixas de classificação internas que nunca são exibidas.

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
- **Tipos** (`core/tipos.py`): numérica = dtype numérico não booleano; binária = 2 valores distintos; categórica = não numérica (exceto identificadores: texto com todos os valores distintos e mais de 10 valores, como nomes ou códigos), ou numérica discreta (só inteiros, inclusive float com NaN) com até 10 níveis. Uma coluna pode ter mais de um papel.
- **Erros** (`ErroLeitura`, mensagem pronta em pt-BR): vazio, só cabeçalho, inexistente/bloqueado, binário, linha com colunas a mais (com número da linha), XLSX inválido, extensão não suportada. **Avisos:** coluna que mistura números e texto; XLSX com várias planilhas (lê a primeira). Colunas sem nome e vazias (separador sobrando) são descartadas.
- A prévia mostra até 100 linhas, com NaN vazio e decimais com vírgula. A leitura roda fora da thread da UI (`Controller.abrir_arquivo` usa `asyncio.to_thread`).

**Fase 3 — Testes de hipótese (um por um)** ✅ concluída
Itens 1–15 da seção 7, cada um com o checklist da seção 9.

Andamento: **15/15** (Médias, Proporções, Categóricos, Não paramétricos e ANOVA completos)
- ✅ `teste_t_1am` (`core/testes/medias.py`, testes em `tests/core/test_medias.py`): card com Wilcoxon de x − μ₀; histograma com x̄ e μ₀.
- ✅ `teste_t_2am` (`tests/core/test_medias_2am.py`): coluna numérica + coluna de grupo com exatamente 2 níveis (`coluna_binaria`); grupo 1 = primeiro nível em ordem crescente; Welch (padrão) ou pooled; card com Mann-Whitney; boxplot por grupo.
- ✅ `teste_t_pareado` (`tests/core/test_medias_pareado.py`): duas colunas numéricas pareadas na mesma linha (d = medida 1 − medida 2); linhas incompletas descartadas com aviso; card com Wilcoxon das diferenças; histograma das diferenças.
- ✅ `teste_z_1prop` (`core/testes/proporcoes.py`, testes em `tests/core/test_proporcoes.py`): coluna binária + valor de "sucesso" (parâmetro `nivel` dependente da coluna); erro padrão com p₀ (teste de escore); IC de Wilson; h de Cohen; card com binomial exato; aviso se n·p₀ ou n·(1 − p₀) < 5; barras de proporções com p₀.
- ✅ `teste_z_2prop` (`tests/core/test_proporcoes_2p.py`): resposta binária + sucesso + grupo com 2 níveis; z com proporção combinada; IC de Wald (não combinado) para p₁ − p₂; h de Cohen e odds ratio com IC de Woolf (indefinida com célula zero); card com Fisher exato; tabela 2×2 na Análise; barras por grupo com a proporção combinada.
- ✅ `qui_quadrado` (`core/testes/categoricos.py`, testes em `tests/core/test_categoricos.py`): campo "Tipo de teste" — Independência (duas categóricas, crosstab, `chi2_contingency`) ou Aderência (uma categórica, proporções iguais, `chisquare`); Yates opcional (desligado; só 2×2); V de Cramér / w de Cohen; **sem card**; tabelas observada (com totais) e esperada; aviso de Cochran (sugere Fisher em 2×2); barras agrupadas (independência) ou barras com 1/k (aderência).
- ✅ `fisher` (`core/testes/categoricos.py`, testes em `tests/core/test_fisher.py`): duas colunas com 2 valores + "evento" de cada uma (parâmetros `nivel`); tabela [[a, b], [c, d]] com linhas (evento₁, outro₁) e colunas (evento₂, outro₂); hipóteses OR ≠ 1 / OR > 1 / OR < 1; p exato; odds ratio amostral (ad/bc) e condicional (EMV) com IC exato condicional (iguais ao `fisher.test` do R); aviso de célula zero; **sem card**; tabela 2×2 com totais; barras agrupadas.
- ✅ `mcnemar` (`core/testes/categoricos.py`, testes em `tests/core/test_mcnemar.py`): duas colunas binárias pareadas com os mesmos 2 valores + "evento"; tabela de pares; **card exato (binomial nos discordantes) × qui-quadrado com Edwards**, sem campo de escolha (raiz com sinal nas unilaterais); decisão automática: exato se b + c < 25, senão assintótico (o Resumo informa); diferença de proporções marginais com IC de Wald pareado; odds ratio pareada b/c com IC exato (Clopper-Pearson); barras antes/depois.
- ✅ `teste_sinal` (`core/testes/nao_parametricos.py`, testes em `tests/core/test_sinal.py`): "Tipo de teste" Uma amostra (coluna + M₀) ou Pareado (mediana das diferenças contra M₀, padrão 0); empates com M₀ descartados (n reduzido, aviso); p exato binomial; mediana com IC exato por estatísticas de ordem e a confiança obtida; **sem card**; histograma com mediana e M₀. Bases: `bases/atendimento_br.csv` (uma amostra) e `bases/pressao_br.csv` (pareado).
- ✅ `wilcoxon` (`core/testes/nao_parametricos.py`, testes em `tests/core/test_wilcoxon.py`): mesmo formulário/validação do sinal (Uma amostra ou Pareado); diferenças nulas descartadas (aviso); empates com posto médio; p exato se n ≤ 50 sem empates, senão normal com correção de empates (o Resumo informa); W⁺/W⁻; pseudomediana de Hodges-Lehmann com IC (exato como o `wilcox.test` do R, ou aproximado); r = z/√n; aviso de simetria; **sem card**; histograma com pseudomediana e M₀. Bases: `bases/dieta_br.csv` (pareado, exato) e `bases/atendimento_br.csv` (uma amostra, aproximado).
- ✅ `mann_whitney` (`core/testes/nao_parametricos.py`, testes em `tests/core/test_mann_whitney.py`): numérica + grupo com 2 níveis (grupo 1 = primeiro em ordem crescente); hipóteses G₁ ≠/>/< G₂; exato se min(n₁, n₂) ≤ 8 sem empates, senão normal com correções de empates e continuidade (o Resumo informa); R₁/R₂, U₁/U₂, deslocamento de Hodges-Lehmann com IC (exato como o `wilcox.test` do R ou aproximado), probabilidade de superioridade U₁/(n₁n₂) e r = z/√N; **sem card**; boxplot. Base: `bases/turmas_br.csv`.
- ✅ `kruskal_wallis` (`core/testes/nao_parametricos.py`, testes em `tests/core/test_kruskal.py`): numérica + grupo (`coluna_categorica`) com 2 a 20 níveis e ≥ 2 observações por grupo; H corrigido para empates, p qui-quadrado (k − 1 gl); aviso se algum grupo tem n < 5; ε² = H/(N − 1); tabela por grupo (n, mediana, posto médio); **pós-teste de Dunn opcional** (desligado; só exibido com H₀ rejeitada) com p ajustado por Holm; **sem card**; boxplot. Base: `bases/entregas_br.csv`.
- ✅ `friedman` (`core/testes/nao_parametricos.py`, testes em `tests/core/test_friedman.py`): 3+ colunas numéricas na mesma linha (caixas de seleção, `multi_coluna`); cada linha é um bloco; estatística com correção de empates, p qui-quadrado (k − 1 gl); aviso com menos de 10 blocos; W de Kendall; tabela por coluna (mediana, posto médio); **comparações múltiplas opcionais** (Wilcoxon pareado entre pares + Holm; só com H₀ rejeitada); **sem card**; boxplot por coluna. Base: `bases/provas_br.csv`.
- ✅ `anova_1fator` (`core/testes/anova.py`, testes em `tests/core/test_anova.py`): numérica + fator com 2 a 20 níveis (≥ 2 obs. por grupo); campo "Variante" — Clássica (padrão, `f_oneway`) ou Welch (`f_oneway(equal_var=False)`, exige variância > 0 em cada grupo); tabela ANOVA (SQ, gl, QM, F, p) e η²/ω² sempre da decomposição clássica; tabela por grupo (n, média, desvio, IC da média por t); Levene centrado na mediana (Brown-Forsythe) só como aviso quando p < 0,05; **Tukey HSD opcional** (`scipy.stats.tukey_hsd`, Tukey-Kramer; desligado; só com H₀ rejeitada; aviso se usado com Welch); **card ANOVA × Kruskal-Wallis com uma única linha** ("algum μᵢ ≠ μⱼ"); boxplot. Base: `bases/fertilizantes_br.csv`. As tabelas da Análise passaram a rolar na horizontal (`tela_principal._tabela`), para não cortar texto ao lado do card.
- ✅ `anova_2fator` (`core/testes/anova.py`, testes em `tests/core/test_anova2.py`): numérica + Fator A + Fator B (categóricas, 2 a 20 níveis); toda combinação com ≥ 1 observação (≥ 2 com a interação); "Incluir interação A × B" (ligado) e "Soma de quadrados" Tipo II (padrão) ou III, via `ols` + `anova_lm` com codificação por soma (import tardio do statsmodels); aviso de desenho desbalanceado; tabela por fonte com η² parcial; médias por combinação; decisão principal = interação (sem ela, os dois efeitos principais com **correção de Bonferroni**, p × 2: rejeita se algum p ajustado ≤ α; decisão do autor, 2026-10-08); interpretação comenta cada efeito e pede cautela com interação significativa; Levene (mediana) entre combinações só como aviso; sem pós-teste; **sem card**; barras agrupadas das médias (A nos grupos, B nas séries). Base: `bases/canteiros_br.csv`.
- Próximo: Fase 4 (ver abaixo).
- Com todos os testes implementados, o estado "teste indisponível" é testado com a fixture `teste_indisponivel` (`tests/conftest.py`), que troca a Regressão Logística por uma entrada sem classe só durante o teste.
- Sugestão automática de "sucesso"/"evento" (`core/tipos.nivel_sucesso_padrao`): valor típico ("1", "Sim", "Aprovado", "Doente", "Positivo"...); senão, entre "X" e "Não X"/"Sem X", sugere "X"; senão, o último nível. O usuário pode sempre trocar.

Padrão estabelecido pelos testes já implementados (seguir nos próximos):
- `parametros()` inclui a hipótese alternativa como `opcao` com rótulos matemáticos (`μ ≠ μ₀`...) mapeados para o `alternative` do scipy; padrão bilateral.
- `validar()` devolve mensagens prontas; `executar()` chama `validar()` e lança `ErroValidacao` se houver erro.
- `ResultadoTeste`: `estatisticas` com chaves técnicas (`t`, `gl`, `p_valor`, `ic_inferior`...), `tabelas["Resumo"]` com colunas `Medida`/`Valor` já formatadas em pt-BR, `figuras` via `core/figuras.py`, `avisos` não bloqueantes e `comparacao` com as três alternativas na ordem ≠, >, <.
- Interpretação via `core/interpretacao.interpretar` (cita α, p, H₀, H₁ e a conclusão no contexto).
- Validações comuns em `core/validacao.py` (`erros_coluna`, `erro_opcao`, `erro_alfa`, `erro_numero`) para as mensagens ficarem iguais entre testes.
- Docstring da classe documenta as escolhas (ddof, zeros, correção, exato vs. assintótico, erro padrão).
- Avisos técnicos do scipy/statsmodels (em inglês) não chegam ao usuário: suprimir pontualmente e emitir aviso equivalente em pt-BR.
- **Card de comparação:** só quando há um equivalente natural (não paramétrico ou exato); testes sem essa estrutura (qui-quadrado, não paramétricos de postos, diagnósticos...) não sobrescrevem `comparacao_inicial()` e devolvem `comparacao=None` — decisão do autor.
- Valores padrão e números exibidos em pt-BR (vírgula decimal); os campos numéricos aceitam vírgula ou ponto. Sem vírgula, o ponto é decimal ("0.05", "1.5"), mas um texto que só pode ser milhar ou decimal ("1.000", "2.500") é **rejeitado** com mensagem (`core.validacao.NumeroAmbiguo`; decisão do autor, 2026-10-08). Por isso os campos inteiros da Simulação aparecem sem ponto de milhar ("1200").
- Referências nos testes: fórmula manual (numpy) + outra biblioteca (statsmodels) ou enumeração exata; fonte documentada no topo do arquivo de teste. Cálculos de referência reutilizáveis ficam em `tests/referencias.py`.
- **Bases para teste manual no app ficam em `bases/`** (pedido do autor), nunca só em pasta temporária. Cada base nova é salva lá (preferir o formato brasileiro: cp1252, `;`, vírgula decimal, sufixo `_br`) e ganha uma linha em `bases/README.md` com o teste, como preencher o formulário e o resultado esperado.

**Fase 4 — Regressão** ✅ concluída
Itens 16 e 17. Andamento: **2/2**.
- ✅ `regres_linear` (`core/testes/regressao.py`, `core/diagnosticos.py`, `app/ui/painel_simulacao.py`; testes em `tests/core/test_diagnosticos.py`, `tests/core/test_regressao.py`, `tests/app/test_ui_regressao.py`): segue `docs/regressao_linear.md` (especificação do autor + escolhas de implementação). Referências do R em `tests/referencias_r/` (script, dados e JSON; o pytest não depende do R). Preditores numéricos codificados podem entrar como categóricas (campo "Tratar como categóricas", `multi_coluna` com `depende_de="preditores"`; também na logística). Análise por `Secao` (Pressupostos → Modelo → Coeficientes), resíduos com lista "Eixo X" (`GrupoFiguras`) e Q-Q, aba Simulação (previsão no core, pedida pelo `controller.simular`). Base: `bases/salarios_br.csv`.
- ✅ `regres_logit` (`core/testes/regressao_logistica.py`, herda da linear; testes em `tests/core/test_regressao_logit.py` e `tests/app/test_ui_regressao_logit.py`): segue `docs/regressao_logistica.md` (7 pontos aprovados pelo autor + escolhas de implementação). y com 2 valores + "Evento"; Logit (statsmodels); IC de Wald; VIF/GVIF, Box-Tidwell, Hosmer-Lemeshow, EPV e separação; teste da razão de verossimilhança, pseudo-R² de McFadden e Nagelkerke, AIC; classificação pelo limiar (matriz de confusão, acurácia, sensibilidade, especificidade, AUC); coeficientes com odds ratio; ROC, probabilidades por classe e resíduos de deviance; Simulação com probabilidade prevista, IC e classe. Referências do R em `tests/referencias_r/gerar_referencias_logit.R` (`glm.control(epsilon = 1e-14)`: com o padrão, o IRLS do R para antes do máximo). Base: `bases/credito_br.csv`.
- O painel da Simulação é genérico: textos e rodapé da equação vêm do simulador (`titulo_resultado`, `rodape_equacao`, `textos_previsao`).
- Flet 0.86.2: com as abas já na tela, `Tabs.selected_index` não move a aba visível; `PainelAbas.ir_para` usa `Tabs.move_to` (via `page.run_task`). Mudar o número de abas exige um `page.update()` antes de selecionar.

**Fase 5 — Otimização e acabamento** ✅ concluída
Perfilar (`cProfile`/`time`) e otimizar só o que estiver medido como lento. Revisar mensagens, textos de interpretação, README.

Medição: `python scripts/medir_desempenho.py [n]` (bases sintéticas com semente fixa; leitura, tipos, prévia, cada teste e figuras). Resultado com n = 100 000 (antes → depois da otimização):

| Etapa | Antes | Depois | O que mudou |
|-------|-------|--------|-------------|
| Wilcoxon / Mann-Whitney | **falha de memória** (9 GB) | 0,8 s / 0,5 s | Hodges-Lehmann por contagem (bissecção + `searchsorted`) acima de 2 milhões de pares |
| Regressão linear e logística (validação) | **falha de memória** (74 GB) | — | `svd(full_matrices=False)` na detecção de colinearidade |
| Teste exato de Fisher | 20,9 s | 0,3 s | odds ratio condicional e IC vetorizados (`core/exatos.py`) |
| Regressão logística | 25,2 s | 0,7 s | curva ROC por somas acumuladas (era O(n²)) |
| Friedman | 9,3 s | 0,2 s | postos por linha com `rankdata(axis=1)` |
| Teste do sinal | 3,5 s | 0,02 s | cdf binomial vetorizada no IC da mediana |
| Leitura de arquivo | congelava a janela | em `asyncio.to_thread`, com aviso "Lendo…" | — |
| Leitura de XLSX (50 mil linhas) | 5,1 s | 0,6 s | `python-calamine` (decisão do autor); openpyxl de reserva |

Medidos e mantidos: regressão linear 3 s (custo do Harrison-McCabe com 1000 simulações, como o `hmctest`); abertura do app ~1,5 s (pandas/scipy; coberta pela splash; statsmodels no topo de `proporcoes.py` custa 23 ms); demais testes < 1 s. Os caminhos rápidos são conferidos contra os de referência em `tests/core/test_desempenho.py`.

Acabamento (revisão de todos os textos exibidos — validações, avisos, notas, interpretações, rótulos de tabelas e gráficos dos 17 testes):
- Mann-Whitney: H₁ textual como o H₀ (antes "G₁ ≠ G₂" no meio da frase).
- Card de comparação: "p-valor" (antes "p-value"); depois da execução o rodapé diz "p-valores da última execução do teste de hipótese.".
- α exibido com vírgula ("0,01 / 0,05 / 0,10"), como todos os números da interface.
- Z de duas proporções: "Odds ratio (razão de chances)", alinhado aos demais testes.
- Mensagens de validação e avisos conferidos por varredura (AST): frases completas, maiúscula inicial e ponto final.
- README reescrito para o usuário (como usar, como ler o resultado, catálogo, desenvolvimento).
- Regressões: avisos dos dados e gráficos de resíduos em funções comuns (`avisos_dos_dados`, `residuos_por_eixo`).

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
| `registry` | os 22 ids registrados, sem duplicatas, rótulos idênticos à lista oficial, ordem preservada |
| `controller` | fluxo carregar → selecionar → executar → resultado; arquivo carregado depois da seleção atualiza o painel; erros tratados |
| `app/ui` | montagem sem exceção; sidebar com todos os testes; as 3 abas existem (4 com Simulação, só na regressão linear); estados vazios corretos; `CardComparacaoTestes.atualizar_p_values` altera os textos esperados |

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

Executável Windows (só quando o autor pedir; o 1º foi gerado em 2026-10-08):
`powershell -ExecutionPolicy Bypass -File scripts\gerar_executavel.ps1` → `flet pack` em modo
pasta (`dist\smartETL\smartETL.exe` + `dist\smartETL-windows.zip`), ícone `assets/icon2.ico`
no `.exe` e na janela (`page.window.icon`, `main.recurso` resolve o caminho dentro do pacote).
No executável o log vai para `%LOCALAPPDATA%\smartETL\smartetl.log`. O cliente Flet vai no
pacote (`flet_desktop/app/flet-windows.zip`) e é extraído no 1º uso, sem internet.

No Windows (PowerShell 5.1), passe mensagens de commit com `git commit -F arquivo.txt`: aspas duplas dentro de `-m` são quebradas pelo PowerShell ao chamar executáveis nativos.

## 13. Decisões em aberto (confirmar com o autor antes de implementar)

- Qual equivalente não paramétrico aparece no card para cada teste e quais testes não terão card. **Decidido:** t de uma amostra ↔ Wilcoxon; t de duas amostras ↔ Mann-Whitney (Welch como padrão, opção pooled; entrada só no formato coluna numérica + grupo de 2 níveis). t pareado ↔ Wilcoxon das diferenças (entrada: duas colunas pareadas na mesma linha; linhas incompletas descartadas com aviso). Z de uma proporção ↔ binomial exato (entrada: coluna binária + valor de sucesso escolhido no formulário; erro padrão com p₀; IC de Wilson). Z de duas proporções ↔ Fisher exato (proporção combinada no z, IC de Wald não combinado; efeitos: diferença, h de Cohen e odds ratio). Qui-quadrado: **sem card** (modos independência e aderência com proporções iguais; Yates opcional e desligado; V de Cramér / w de Cohen). **Fisher (implementado):** entrada = duas colunas categóricas com exatamente 2 valores cada (tabela 2×2), e o formulário pede qual valor de cada coluna é o "evento" (parâmetros `nivel` dependentes das colunas, como o "sucesso" do teste Z); hipóteses bilateral, "odds ratio > 1" e "odds ratio < 1"; resultados: p exato (`scipy.stats.fisher_exact`), odds ratio amostral com IC exato condicional (`scipy.stats.contingency.odds_ratio(kind="conditional")`, o mesmo do `fisher.test` do R), tabela 2×2 observada e barras agrupadas; **sem card**. **McNemar (implementado):** entrada = duas binárias pareadas na mesma linha + evento; sem campo exato/assintótico — o card mostra os dois lado a lado (exato binomial × qui-quadrado com correção de Edwards); a decisão usa o exato quando b + c < 25 e o assintótico nos demais (regra automática escolhida pelo autor). **Teste do sinal (implementado):** campo "Tipo de teste", como no qui-quadrado — Uma amostra (coluna numérica + mediana hipotética M₀) ou Pareado (duas colunas numéricas na mesma linha; mediana das diferenças); empates com M₀ (diferença zero) descartados, reduzindo o n, com aviso; hipóteses M ≠ M₀, M > M₀, M < M₀; resultados: sinais + e −, p exato (binomial com p = 1/2), mediana amostral com IC exato por estatísticas de ordem, histograma com a mediana e M₀; **sem card**. **Wilcoxon (implementado):** mesmo "Tipo de teste" do sinal; zeros descartados (`zero_method="wilcox"`), postos médios nos empates, `method` exato/aproximado automático (informado no Resumo); W⁺, p nas três alternativas, pseudomediana de Hodges-Lehmann com IC, r = z/√n; **sem card**. **Mann-Whitney (implementado):** numérica + grupo com 2 níveis; método automático (informado); U₁/U₂ e somas de postos; deslocamento de Hodges-Lehmann com IC; probabilidade de superioridade e r; boxplot; **sem card**. **Kruskal-Wallis (implementado):** numérica + grupo com ≥ 2 níveis; H com empates e p qui-quadrado; ε²; tabela por grupo; Dunn opcional (desligado) com Holm, só quando H₀ é rejeitada; **sem card**. **Friedman (implementado):** 3+ colunas pareadas (caixas de seleção); Q com empates e p qui-quadrado; W de Kendall; tabela por coluna; comparações opcionais Wilcoxon pareado + Holm, só com H₀ rejeitada; **sem card**. **ANOVA 1 fator (implementado):** numérica + fator (≥ 2 níveis); variante clássica (padrão) ou Welch, com aviso do Levene quando as variâncias diferem; tabela ANOVA completa, η² e ω², tabela por grupo e boxplot; Tukey HSD opcional (desligado), só com H₀ rejeitada; **card ANOVA × Kruskal-Wallis, só a linha bilateral**. **ANOVA 2 fatores (implementado):** Tipo II padrão (opção III), interação ligada por padrão, η² parcial, médias por combinação em barras agrupadas, Levene como aviso, sem Tukey e **sem card**. **Regra geral do autor:** card só quando a comparação for possível e útil; caso contrário, sem card.
- ~~O card continua nas abas Parâmetros e Análise ou fica só em Análise?~~ Decidido na Fase 1: **duas instâncias** (Parâmetros e Análise), para preservar o visual. Pode ser revisto depois.
- ~~Gráficos nativos do Flet ou imagens do matplotlib?~~ Decidido na Fase 3: **nativos do Flet, simples e minimalistas**, desenhados com `flet.canvas` (no Flet 0.86.2 `BarChart`/`LineChart` saíram do pacote principal para a extensão `flet-charts`; o canvas é do núcleo e não exige dependência nova). Na interface o matplotlib não é usado; ele só desenha os gráficos do PDF (`core/relatorio.py`, a partir das mesmas `Figura`).
- Pós-testes (Tukey, Dunn) e pressupostos extras (Shapiro-Wilk, Levene) como funcionalidade adicional. **Decidido em parte:** Dunn com Holm no Kruskal-Wallis, opcional e desligado por padrão. Tukey HSD na ANOVA de 1 fator, opcional e desligado; Levene (centrado na mediana) só como aviso não bloqueante quando p < 0,05. **Shapiro-Wilk (decidido e implementado, 2026-10-08):** nos três testes t (sobre a coluna, cada grupo ou as diferenças) e nas duas ANOVAs (sobre os resíduos), com α fixo de 0,05 como o Levene; linha "p-valor do Shapiro-Wilk (...)" no Resumo ("—" com n < 3 ou valores constantes; Lilliefors acima de 5000) e aviso não bloqueante que sugere o equivalente (Wilcoxon, Mann-Whitney, Kruskal-Wallis; na ANOVA de 2 fatores, uma transformação) e, com n ≥ 30, lembra que a falta de normalidade afeta pouco o teste. Funções em `core/diagnosticos.py` (`normalidade_amostra`, `aviso_normalidade`); referências do R em `tests/referencias_r/gerar_referencias_normalidade.R`.
- **Regressão logística (decidido e implementado, 2026-10-07):** `docs/regressao_logistica.md`.
- **Correlações (decidido e implementado, 2026-10-07):** grupo novo Correlação (entre ANOVA e Regressão), dois testes. Entrada: X e Y numéricas, linhas completas (aviso), n ≥ 3, variação nas duas. H₀: ρ = 0 com ≠ / > / <. Pearson: r, r², t (n − 2 gl), IC pela z de Fisher (unilateral quando H₁ é), força pelas faixas de Cohen (|r| < 0,1 desprezível, < 0,3 fraca, < 0,5 moderada, senão forte), aviso se |r − ρₛ| > 0,2, dispersão com a reta de mínimos quadrados e **card Pearson × Spearman**. Spearman: ρₛ com postos médios, p pela aproximação t (= `cor.test(exact = FALSE)`), IC pela z de Fisher com a variância de Bonett-Wright, aviso de empates, dispersão de valores ou postos (lista "Escala"), **sem card**. Referências do R em `tests/referencias_r/gerar_referencias_correlacao.R`; base `bases/estudo_br.csv`.
- **Correlação de Kendall (decidido e implementado, 2026-10-08):** `core/testes/correlacao_kendall.py` (herda a base das correlações: X e Y numéricas, linhas completas, n ≥ 3). τ-b pelo `kendalltau`; p exato com n < 50 sem empates, senão normal com correção de empates e sem continuidade (= `cor.test(method = "kendall")`; o Resumo informa); pares concordantes/discordantes e z = S/√var(S) com a variância de empates do R, em O(n log n); IC pela z de Fisher com var 0,437/(n − 4) (Fieller-Hartley-Pearson; sem IC com n ≤ 4); força pelas faixas de Cohen; **card Kendall × Spearman**; aviso de empates; dispersão. H₀ cita τ. Referências: `tests/referencias_r/gerar_referencias_kendall.R`; base `bases/estudo_br.csv`.
- **Correlação parcial (decidido e implementado, 2026-10-08):** `core/testes/correlacao_parcial.py`. X, Y e 1..k controles numéricos (`multi_coluna`), linhas completas (aviso), n ≥ k + 4; controles constantes/colineares ou X/Y explicadas totalmente pelos controles bloqueiam. Método Pearson (padrão) ou Spearman (postos médios), r entre os resíduos das regressões sobre [1, Z] (= `ppcor::pcor.test`); t com n − 2 − k gl, H₁ ≠/>/<; IC pela z de Fisher com var 1/(n − 3 − k); **card parcial × simples** (mesmo método, sem controles); aviso se |parcial − simples| > 0,2; dispersão dos resíduos com a reta. Referências: `tests/referencias_r/gerar_referencias_parcial.R` (ppcor 1.1 instalado na biblioteca do projeto); base `bases/estudo_br.csv` (controle `faltas`).
- **Information Value (decidido e implementado, 2026-10-08):** `core/testes/information_value.py`, não é teste de hipótese: `p_valor=None`, `decisao=""` (a Análise e o PDF omitem a linha da decisão; sem α no formulário). Entrada: y com 2 valores + evento, colunas X (`preditores`), faixas 5/10/20 (padrão 10). Quantitativas com mais valores distintos que o nº de faixas viram faixas por quantis (`qcut`, faixas repetidas juntadas), rotuladas pelo menor e maior valor observados ("18 – 30"); poucas distintas e categóricas: um valor por categoria; X ausente = "(ausente)"; y ausente descartado. WoE = ln(%não eventos/%eventos), IV = Σ(%NE − %E)·WoE; categoria com contagem zero: +0,5 nas duas contagens (aviso). Tabela geral Variável | IV | Poder preditivo | Categorias (maior IV primeiro, faixas de Siddiqi; > 0,5 "verificar vazamento" com aviso) e uma tabela por variável (categorias pelo maior IV + Total em negrito). Gráficos: IV por variável (linhas 0,1 e 0,3) e WoE por categoria (lista "Variável"). Referência: exemplo calculado à mão + `crosstab` independente (`tests/core/test_information_value.py`).
- **Regressão linear (decidido, 2026-10-06):** especificação e decisões em `docs/regressao_linear.md` (referências geradas com o R instalado na máquina; entradas do grupo Diagnóstico removidas; teste F na seção do modelo; aba Simulação só para a regressão).
- **Exportação em PDF (decidido e implementado, 2026-10-08):** botão "Exportar PDF" na linha das abas, à direita (`PainelAbas.definir_acao`; visível em qualquer aba, só depois de uma execução) → diálogo de salvar (`FilePicker.save_file`, nome `<teste>_<data-hora>.pdf`) → `Controller.exportar_pdf` gera fora da thread da UI (`asyncio.to_thread`) e notifica. Conteúdo (`core/relatorio.py`, fpdf2 + DejaVu Sans do matplotlib): cabeçalho com **só o nome do teste** em todas as páginas (pedido do autor), arquivo, data/hora e parâmetros; decisão colorida, interpretação, avisos e tabelas (ou as seções, nas regressões), com as dicas das colunas em notas; o card como tabela; todas as figuras (cada opção de um `GrupoFiguras`, desenhadas com matplotlib/Agg); sem a Simulação. Rodapé: **"by smartETL"** e "Página X de Y". As cores do PDF repetem as de `tema.py` (há teste).
- Formatos de arquivo além de CSV/XLSX e exportação em outros formatos (HTML/CSV).
