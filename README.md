# smartETL — Testes de Hipótese

Aplicativo desktop (Flet) para carregar uma base de dados (CSV ou XLSX), escolher um
teste de hipótese, configurar os parâmetros e ver o resultado, a interpretação e os gráficos.
O catálogo prevê 21 testes paramétricos e não paramétricos, além de ANOVA, regressão e
diagnósticos de pressupostos.

## Estado atual

- Leitura de CSV (inclusive o formato brasileiro: separador `;`, vírgula decimal, latin-1/cp1252) e XLSX,
  com mensagens de erro e avisos em português.
- Interface completa: carregamento de arquivo, prévia de até 100 linhas, lista de testes
  agrupada por categoria e abas **Parâmetros**, **Análise** e **Visualização**.
- **Teste t (uma amostra)** completo: hipótese alternativa configurável, estatísticas, IC,
  d de Cohen, interpretação em português, comparação com o Wilcoxon e histograma.
- **Teste t (duas amostras)** completo: Welch ou variâncias iguais, IC da diferença, d de Cohen,
  comparação com o Mann-Whitney e boxplot por grupo.
- **Teste t (pareado)** completo: duas medidas na mesma linha (ex.: antes/depois), IC da diferença,
  d de Cohen (d_z), comparação com o Wilcoxon e histograma das diferenças.
- **Teste Z (uma proporção)** completo: escolha do valor de sucesso, IC de Wilson, h de Cohen,
  comparação com o binomial exato e gráfico de barras das proporções.
- **Teste Z (duas proporções)** completo: IC da diferença, h de Cohen, razão de chances (odds ratio),
  tabela 2×2, comparação com o Fisher exato e barras por grupo.
- **Qui-quadrado** completo: independência (tabela de contingência, V de Cramér, frequências
  observadas e esperadas, barras agrupadas) ou aderência com proporções iguais (w de Cohen).
- **Teste exato de Fisher** completo: tabela 2×2 com escolha do evento de cada variável, p exato,
  odds ratio amostral e condicional com IC exato (como o `fisher.test` do R) e barras agrupadas.
- Os demais testes aparecem na lista como "ainda não disponível nesta versão".

## Requisitos

- Python **3.10 ou superior** (testado com 3.13)
- Dependências fixadas em `requirements.txt` (Flet 0.86.2, pandas, openpyxl, numpy,
  scipy, statsmodels; pytest e ruff para desenvolvimento)

## Instalação e execução

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python main.py
```

## Testes e qualidade

```bash
pytest -q
ruff check . && ruff format --check .
```

## Estrutura

```
main.py                    # ponto de entrada (ft.run)
app/
  state.py                 # estado: dataset, teste selecionado, último resultado
  controller.py            # liga a interface ao core (sem lógica estatística)
  ui/
    tema.py                # paleta única, raios e tipografia
    helpers.py             # pad(), border_all(), border_only()...
    sidebar.py             # logo, Arquivo, lista de testes, Executar
    tabela_dados.py        # prévia do dataset
    painel_abas.py         # abas Parâmetros / Análise / Visualização
    painel_parametros.py   # formulário gerado a partir dos parâmetros do teste
    tela_principal.py      # montagem da tela
    graficos.py            # gráficos nativos (flet.canvas)
    componentes/           # campos com estilo único e card de comparação
core/                      # lógica de domínio, sem Flet
  base.py                  # contratos: ParametroSpec, ResultadoTeste, TesteBase...
  registry.py              # catálogo dos 21 testes
  io.py                    # leitura de CSV/XLSX (encoding, separador e decimal automáticos)
  tipos.py                 # detecção de colunas numéricas, categóricas e binárias
  validacao.py             # regras de validação reutilizáveis
  interpretacao.py         # decisão e textos de interpretação em pt-BR
  figuras.py               # dados dos gráficos (sem Flet)
  testes/                  # implementações por grupo
tests/                     # pytest (core/ e app/)
```

## Licença

Ver [LICENSE](LICENSE).
