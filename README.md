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
- O **Teste t (uma amostra)** já tem formulário e card de comparação (t Student × Wilcoxon).
  O cálculo estatístico entra na próxima fase.
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
    componentes/           # campos com estilo único e card de comparação
core/                      # lógica de domínio, sem Flet
  base.py                  # contratos: ParametroSpec, ResultadoTeste, TesteBase...
  registry.py              # catálogo dos 21 testes
  io.py                    # leitura de CSV/XLSX (encoding, separador e decimal automáticos)
  tipos.py                 # detecção de colunas numéricas, categóricas e binárias
  validacao.py             # regras de validação reutilizáveis
  testes/                  # implementações por grupo
tests/                     # pytest (core/ e app/)
```

## Licença

Ver [LICENSE](LICENSE).
