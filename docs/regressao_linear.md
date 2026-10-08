# Regressão Linear — especificação do autor (fonte da verdade)

Especificação entregue pelo autor em 2026-10-06 para o teste `regres_linear` (Fase 4), seguida das
decisões complementares tomadas na mesma data. Qualquer mudança deve ser confirmada com o autor.

## 1. Entradas

- **Variável dependente (y):** obrigatoriamente numérica. Bloquear a seleção de colunas não
  numéricas e explicar o motivo.
- **Preditores (X):** um ou mais; aceitar numéricos e categóricos.
- **Categóricas:** converter em dummies (k − 1 colunas para k níveis).
  - O usuário escolhe o nível de referência de cada categórica por lista suspensa.
  - Padrão inicial: o nível mais frequente.
  - Avisar quando um nível tiver poucas observações (ex.: < 5).
  - **Numéricas codificadas** (ex.: escolaridade 1 a 5): o campo "Tratar como categóricas"
    (só lista os preditores numéricos marcados) faz a coluna entrar como dummies, com nível de
    referência como as demais; até 50 níveis. Sem a marca, numéricas entram como número.
    Vale também na regressão logística (acrescentado em 2026-10-08, revisão geral).
- **Validações:** remover linhas com valores ausentes nas variáveis do modelo e informar quantas
  foram removidas; exigir n > p + 1; detectar colinearidade perfeita e informar quais variáveis a
  causam.

## 2. Aba "Análise" (nesta ordem)

### 2.1 Pressupostos (exibidos antes de tudo)

Cada teste mostra estatística, p-valor (quando houver) e uma interpretação em linguagem simples.

- **Heterocedasticidade:**
  - Breusch-Pagan
  - Goldfeld-Quandt (usuário escolhe a variável de ordenação; padrão: valores ajustados)
  - Harrison-McCabe (não existe no statsmodels: implementar, com p-valor por simulação; validar
    contra `lmtest::hmctest` do R)
- **Autocorrelação:**
  - Estatística de Durbin-Watson, classificada pelas faixas de referência abaixo.
  - Teste de Breusch-Godfrey (statsmodels) com estatística e p-valor.
  - Nota visível: "regra prática; os limites formais dependem de n e do número de preditores".
  - Aviso: "só é interpretável se as linhas tiverem ordem significativa (tempo, sequência de
    coleta)".
- **Colinearidade:**
  - VIF e tolerância (1/VIF) de cada preditor, classificados pelas faixas de referência abaixo.
  - Categóricas com 3+ níveis: usar GVIF (Fox & Monette) e GVIF^(1/(2·gl)), uma linha por
    variável original; classificar usando (GVIF^(1/(2·gl)))² nas mesmas faixas do VIF.
- **Normalidade dos resíduos:** Shapiro-Wilk (n ≤ 5000) ou Kolmogorov-Smirnov com correção de
  Lilliefors (n > 5000). Nota: com n grande, o teste rejeita desvios irrelevantes; consultar o Q-Q.

#### Faixas de referência (USO INTERNO — NÃO EXIBIR NA INTERFACE)

Constantes únicas no código, nunca exibidas inteiras ao usuário. Intervalos fechados à esquerda e
sem sobreposição.

Durbin-Watson:

| DW            | Interpretação                    |
|---------------|----------------------------------|
| 0 a < 1,0     | Forte autocorrelação positiva    |
| 1,0 a < 1,5   | Autocorrelação positiva          |
| 1,5 a 2,5     | Sem autocorrelação relevante     |
| > 2,5 a 3,0   | Autocorrelação negativa          |
| > 3,0 a 4     | Forte autocorrelação negativa    |

VIF:

| VIF         | Tolerância   | Interpretação               |
|-------------|--------------|-----------------------------|
| 1 a < 2     | > 0,50       | Sem colinearidade relevante |
| 2 a < 5     | 0,20 a 0,50  | Baixa/moderada              |
| 5 a < 10    | 0,10 a 0,20  | Problemática                |
| 10 a < 20   | 0,05 a 0,10  | Grave                       |
| ≥ 20        | ≤ 0,05       | Muito grave                 |

#### O que a interface DEVE exibir

Somente os valores calculados para o modelo, com a interpretação da faixa em que cada valor caiu.
Não exibir as faixas.

- Durbin-Watson: tabela de UMA linha (DW | Interpretação).
- VIF: UMA linha POR PREDITOR do modelo (Preditor | VIF | Tolerância | Interpretação).

### 2.2 Tabela do modelo

R | R² | R² ajustado | RMSE | F | gl1 | gl2 | p-valor | n
(RMSE = erro padrão dos resíduos, √(SQE / (n − p − 1)); fórmula visível em tooltip.)

### 2.3 Tabela de coeficientes

Preditor | Estimativa | EP | LI | LS | t | p-valor. Categóricas agrupadas por variável, com a
referência explícita: linha de cabeçalho "**Variável** (ref.: nível)", uma linha por nível
recuado e a linha do nível de referência com "—". Nível de confiança do IC configurável (padrão
95%).

## 3. Aba "Visualização"

- Dispersão dos resíduos com lista suspensa acima para escolher o eixo X: padrão "Valores
  ajustados"; demais opções: cada preditor do modelo. Para preditores categóricos, boxplot dos
  resíduos por nível. Linha horizontal em y = 0.
- Gráfico Q-Q dos resíduos.

## 4. Aba "Simulação"

- **Topo:** equação do modelo montada dinamicamente, destacando o termo cujo valor o usuário está
  alterando.
- **Campos:** numéricas com campo + slider, iniciando na média, limites na faixa observada; fora
  da faixa, aviso de extrapolação (permitido, mas sinalizado). Categóricas: lista suspensa com os
  níveis (referência incluída); o app converte internamente para dummies.
- **Resultado (tempo real):** valor previsto de y; IC da média prevista e intervalo de predição;
  contribuição de cada termo (barras ou cascata), do intercepto até o valor previsto.

## 5. Testes automatizados

- Coeficientes, EP, t, p, R², F e todos os pressupostos contra referências do R (`lm`, `lmtest`,
  `car::vif`) em pelo menos dois datasets públicos.
- Fronteiras de cada faixa (VIF 4,99 → "Baixa/moderada"; 5,00 → "Problemática"; DW 1,49 →
  "Autocorrelação positiva"; 1,50 → "Sem autocorrelação relevante").
- A interface exibe apenas a linha do DW calculado e as linhas dos preditores do modelo no VIF,
  nunca as tabelas de faixas.
- Casos-limite: uma única preditora; apenas categóricas; categórica com nível raro; colinearidade
  perfeita; troca do nível de referência (coeficientes mudam de forma consistente, R² não muda).

## 6. Restrições

- statsmodels/scipy onde houver implementação; o restante (Harrison-McCabe, GVIF) em funções
  próprias, documentadas e testadas.
- Todo texto da interface em português; interpretações acessíveis, sem esconder os valores.

## Decisões complementares (2026-10-06)

- **Referências do R:** R 4.6.1 instalado na máquina de desenvolvimento (lmtest 0.9.40, car 3.1.5,
  em `%LOCALAPPDATA%\R\smartetl-lib`). Os valores são gerados uma vez por um script R guardado em
  `tests/referencias_r/` e gravados no repositório; o app e o pytest não dependem do R.
- **Entradas do grupo Diagnóstico removidas:** `durbin_watson`, `breusch_pagan`, `white` e `vif`
  saíram da lista oficial (agora 17 testes); DW, BP e VIF ficam dentro da regressão. White não
  entra na especificação acima.
- **Teste F global:** decisão e interpretação ficam na seção do modelo. Ordem da Análise:
  Pressupostos → Modelo (decisão, interpretação, tabela) → Coeficientes.
- **Aba Simulação:** só aparece quando o teste oferece simulação (hoje, só a regressão linear); os
  demais testes continuam com 3 abas.
- **Padrões do R, documentados:** Breusch-Pagan studentizado (Koenker, padrão do `bptest`);
  Breusch-Godfrey de ordem 1 (`bgtest`); Goldfeld-Quandt com divisão na metade, sem omitir
  observações centrais, alternativa "variância maior na segunda parte" (`gqtest`);
  Harrison-McCabe com divisão na metade e 1000 simulações (`hmctest`), semente fixa. Estatísticas
  determinísticas conferidas com `rel=1e-6`; o p simulado do Harrison-McCabe, dentro do erro de
  Monte Carlo.
- **Nível de confiança:** um campo só, usado no IC dos coeficientes e nos intervalos da Simulação
  (padrão 95%).
- **Nome da aba:** continua "Visualização".

## Escolhas de implementação (2026-10-07)

Pontos que a especificação não fixava; documentados aqui e nas docstrings.

- **Tipo de cada preditor:** coluna numérica entra como número; coluna não numérica (texto,
  booleana) vira dummies. Identificadores de texto (todos os valores distintos e mais de 10) não
  aparecem na lista de preditores. Uma coluna 0/1 numérica entra como número (o coeficiente é o
  mesmo da dummy).
- **Ordem das dummies:** níveis em ordem crescente, sem a referência; na tabela de coeficientes, a
  referência vem por último, com "—". Empate no nível mais frequente: o primeiro em ordem crescente.
- **Ordenação do Goldfeld-Quandt e do Harrison-McCabe:** um único campo ("Ordenação dos dados"),
  usado pelos dois; pode ser qualquer coluna numérica (não precisa ser preditor).
- **Goldfeld-Quandt com metade degenerada** (uma dummy constante numa das metades): o cálculo
  segue o `gqtest` (gl com k colunas) e a Análise mostra um aviso de resultado pouco confiável.
- **Classificação do DW e do VIF:** aplicada ao valor exibido (2 casas), para a interpretação
  nunca contradizer o número na tela. Tolerância = 1/(valor classificado).
- **Coluna "VIF / GVIF":** só quando há categórica com 3+ níveis; nesse caso a tabela ganha gl e
  GVIF^(1/(2·gl)). Sem elas, as colunas da especificação (Preditor | VIF | Tolerância |
  Interpretação). Com um único preditor, a seção diz que não há colinearidade a avaliar.
- **Testes de pressuposto:** usam o mesmo α do formulário para a interpretação.
- **Números:** coeficientes, EP e IC com pelo menos 4 algarismos significativos e 2 casas
  (3.135,82; 35,41; 0,001013); t com 3 casas; p no formato do app ("< 0,001").
- **Gráficos:** dispersões com mais de 2000 pontos desenham uma amostra regular (a legenda
  informa); Q-Q com os quantis do `qqnorm` e a reta pelos quartis (`qqline`).
- **Simulação:** o campo numérico aceita qualquer valor (o slider fica na borda da faixa e aparece
  o aviso de extrapolação); categóricas com o nível de referência como valor inicial; a cascata vai
  do intercepto ao previsto, um degrau por preditor original.

