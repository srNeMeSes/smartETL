# smartETL — Testes de Hipótese

Aplicativo desktop para análise estatística de bases de dados. Você carrega um arquivo CSV ou
XLSX, escolhe um teste na barra lateral, preenche os parâmetros e recebe o resultado: decisão,
interpretação em português, tabelas e gráficos. Os 19 testes do catálogo — de testes t e
qui-quadrado a ANOVA, correlações e regressões — têm os valores conferidos contra o R e o scipy/statsmodels.

## Como usar

1. **Abra o app** (`python main.py`). Uma tela de abertura aparece enquanto os módulos
   estatísticos carregam.
2. **Clique em Arquivo** e escolha um `.csv` ou `.xlsx`. O formato é detectado sozinho:
   separador (`;`, `,`, tab ou `|`), vírgula ou ponto decimal, milhar e codificação (UTF-8,
   Windows-1252, Latin-1). A prévia mostra as primeiras 100 linhas.
3. **Escolha o teste** na barra lateral. A aba **Parâmetros** mostra só os campos daquele teste,
   e as listas de colunas já vêm filtradas pelo tipo certo (numérica, categórica, com 2 valores…).
4. **Clique em Executar teste.** O resultado abre na aba **Análise**; os gráficos ficam em
   **Visualização**. Nas regressões há ainda a aba **Simulação**.
5. **Exporte a análise** (opcional) com o botão **Exportar PDF**, à direita das abas: o
   relatório traz os parâmetros, a decisão, a interpretação, os avisos, todas as tabelas e todos
   os gráficos.

Para experimentar sem dados próprios, use as bases da pasta [`bases/`](bases/): o
[`bases/README.md`](bases/README.md) diz, para cada uma, que teste usar, como preencher o
formulário e qual resultado esperar.

## Como ler o resultado

- **Decisão:** "Rejeita H₀" (vermelho suave) ou "Não rejeita H₀" (verde suave), sempre pela regra
  p ≤ α, com o α escolhido no formulário (0,01, 0,05 ou 0,10).
- **Interpretação:** uma frase que cita α, o p-valor, H₀, H₁ e a conclusão no contexto dos seus
  dados ("Há evidência estatística de que a média de 'nota' é diferente de 7").
- **Avisos** (⚠): pressupostos duvidosos ou dados que pedem cautela — amostra pequena, empates,
  frequências esperadas baixas, linhas removidas por valor ausente, separação na regressão
  logística… Eles não bloqueiam o resultado.
- **Card de comparação:** quando existe um equivalente natural, o p-valor do teste paramétrico
  aparece ao lado do não paramétrico (ex.: t de Student × Wilcoxon), para cada hipótese
  alternativa.
- **Dicas** (ⓘ ao lado do nome da coluna de uma tabela): passe o mouse para ver a fórmula ou o
  significado da medida.
- **Números** em formato brasileiro (vírgula decimal); p-valores abaixo de 0,001 aparecem como
  "< 0,001".

## Testes disponíveis

| Grupo | Testes |
|-------|--------|
| Médias | Teste t (uma amostra), Teste t (duas amostras — Welch ou variâncias iguais), Teste t (pareado) |
| Proporções | Teste Z (uma proporção), Teste Z (duas proporções) |
| Categóricos | Qui-quadrado (independência ou aderência), Teste exato de Fisher, McNemar |
| Não paramétricos | Teste do sinal, Wilcoxon, Mann-Whitney U, Kruskal-Wallis (com Dunn), Friedman |
| ANOVA | ANOVA (1 fator — clássica ou Welch, com Tukey HSD), ANOVA (2 fatores — Tipo II ou III) |
| Correlação | Correlação de Pearson (com o card Pearson × Spearman), Correlação de Spearman |
| Regressão | Regressão Linear, Regressão Logística |

Destaques:

- **Tamanhos de efeito e intervalos de confiança** em todos os testes (d de Cohen, h de Cohen,
  V de Cramér, odds ratio, ε², W de Kendall, η², ω²…).
- **Regressão Linear:** preditores numéricos e categóricos (com o nível de referência escolhido).
  A Análise começa pelos pressupostos — heterocedasticidade (Breusch-Pagan, Goldfeld-Quandt,
  Harrison-McCabe), autocorrelação (Durbin-Watson, Breusch-Godfrey), colinearidade (VIF/GVIF) e
  normalidade dos resíduos —, depois o modelo e os coeficientes. Visualização com resíduos e Q-Q.
- **Regressão Logística:** y com 2 valores e o evento escolhido; Box-Tidwell, Hosmer-Lemeshow,
  eventos por variável e detecção de separação; teste da razão de verossimilhança, pseudo-R²,
  matriz de confusão, AUC, odds ratios e curva ROC.
- **Simulação** (regressões): a equação do modelo no topo, um campo com slider para cada preditor
  numérico e uma lista para cada categórico. A previsão — com intervalo de confiança e, na
  linear, intervalo de predição; na logística, a probabilidade e a classe prevista — atualiza em
  tempo real, e um gráfico mostra a contribuição de cada termo. Valores fora da faixa observada
  são sinalizados como extrapolação.

## Instalação

Requer Python **3.10 ou superior** (testado com 3.13).

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python main.py
```

As dependências estão fixadas em `requirements.txt`: Flet 0.86.2 (interface), pandas, numpy,
scipy, statsmodels, openpyxl e python-calamine (leitura de XLSX), fpdf2 e matplotlib (relatório
em PDF); pytest, ruff e pypdf para desenvolvimento.

## Desenvolvimento

```bash
pytest -q                               # ~1100 testes, sem abrir janela
ruff check . && ruff format --check .   # lint e formatação
python scripts/medir_desempenho.py      # tempos de leitura e de cada teste com n = 100 000
```

- **Valores de referência:** cada teste é conferido contra cálculo independente (fórmula manual,
  enumeração exata ou outra biblioteca), com a fonte documentada no topo do arquivo de teste. As
  regressões são conferidas contra o R (`lm`, `glm`, `lmtest`, `car`, `nortest`,
  `ResourceSelection`, `pROC`): os scripts e os resultados ficam em
  [`tests/referencias_r/`](tests/referencias_r/) — o R só é necessário para regerá-los.
- **Decisões de projeto e de cada teste:** [`CLAUDE.md`](CLAUDE.md);
  especificações das regressões em [`docs/`](docs/).
- **Desempenho:** com 100 000 linhas, a leitura leva menos de 1 s e todos os testes rodam em até
  ~3 s; leitura e execução acontecem fora da thread da interface.

### Estrutura

```
main.py                    # ponto de entrada: splash e carregamento em segundo plano
app/                       # interface (Flet) e controller
  controller.py            # liga a interface ao core (sem lógica estatística)
  state.py                 # dataset carregado, teste selecionado, último resultado
  ui/
    splash.py              # tela de abertura
    sidebar.py             # logo, Arquivo, lista de testes, Executar
    tabela_dados.py        # prévia do dataset
    painel_abas.py         # abas Parâmetros / Análise / Visualização (+ Simulação)
    painel_parametros.py   # formulário gerado a partir dos parâmetros do teste
    painel_simulacao.py    # aba Simulação das regressões
    tela_principal.py      # montagem da tela
    graficos.py            # gráficos nativos (flet.canvas)
    tema.py, helpers.py    # paleta única e utilitários de layout
    componentes/           # campos com estilo único e card de comparação
core/                      # lógica estatística, sem Flet
  base.py                  # contratos: ParametroSpec, ResultadoTeste, TesteBase...
  registry.py              # catálogo dos 19 testes
  io.py, tipos.py          # leitura de CSV/XLSX e detecção de tipos de coluna
  validacao.py             # regras de validação reutilizáveis
  interpretacao.py         # decisão e textos de interpretação em pt-BR
  figuras.py               # dados dos gráficos
  diagnosticos.py          # pressupostos (Shapiro-Wilk, GQ, HMC, GVIF, Hosmer-Lemeshow...)
  relatorio.py             # exportação da análise em PDF
  exatos.py                # odds ratio condicional (Fisher)
  testes/                  # um módulo por grupo de testes
tests/                     # pytest (core/ e app/) e referências do R
bases/                     # bases de exemplo, com instruções de uso
docs/                      # especificações das regressões
scripts/                   # medição de desempenho
```

## Licença

Ver [LICENSE](LICENSE).
