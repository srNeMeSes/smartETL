# Regressão Logística — decisões do autor (fonte da verdade)

Proposta aprovada pelo autor em 2026-10-07 ("concordo com os 7 pontos"). Reaproveita a estrutura da
regressão linear (`docs/regressao_linear.md`): formulário, Análise em seções, coeficientes
agrupados, lista "Eixo X" e aba Simulação.

1. **Entradas:** y com exatamente 2 valores + campo "Evento" (o valor que conta como sucesso,
   sugestão automática como nos testes Z/Fisher/McNemar); preditores como na linear (numéricos e
   categóricos, referência escolhida, y fora da lista). Validações da linear (linhas incompletas
   removidas com aviso, colinearidade perfeita, n mínimo) + cada classe de y com ao menos 1
   observação, aviso de poucos eventos por variável (EPV < 10, Peduzzi) e de separação completa
   ou quase completa (coeficientes e p-valores não confiáveis).
2. **Ajuste:** máxima verossimilhança (`statsmodels.api.Logit`, = `glm(family = binomial)` do R);
   IC de Wald (= `confint.default`) no nível de confiança do formulário.
3. **Análise:**
   - Pressupostos e diagnósticos: colinearidade (VIF/GVIF, mesmas faixas internas da linear);
     linearidade do logit (Box-Tidwell: termo x·ln x, só para preditores numéricos positivos);
     qualidade do ajuste (Hosmer-Lemeshow, 10 grupos, como `ResourceSelection::hoslem.test`);
     tamanho amostral e separação (avisos).
   - Modelo: decisão pelo teste da razão de verossimilhança (modelo × nulo), com a cor
     vermelho/verde suave; pseudo-R² de McFadden e Nagelkerke, AIC, log-verossimilhança, n,
     eventos.
   - Classificação: limiar configurável (padrão 0,5); matriz de confusão, acurácia,
     sensibilidade, especificidade e AUC.
   - Coeficientes: Preditor | Estimativa (log-odds) | EP | z | p-valor | Odds ratio | LI | LS
     (do OR), categóricas agrupadas com a referência explícita.
4. **Visualização:** curva ROC com a AUC; boxplot das probabilidades previstas por classe
   observada, com a linha do limiar; resíduos de deviance com a lista "Eixo X".
5. **Simulação:** equação em log-odds (`logit(P) = b₀ + …`, termo alterado em destaque) e
   `P = 1 / (1 + e^(−logit))`; campos como na linear; probabilidade prevista com IC (escala do
   logit, convertido); classe prevista pelo limiar; sem intervalo de predição (nota explica);
   cascata das contribuições em log-odds.
6. **Sem card.**
7. **Referências:** R 4.6.1 (`glm`, `car::vif`, `ResourceSelection::hoslem.test` 0.3.6,
   `pROC::auc` 1.19.1), script e dados em `tests/referencias_r/`.

## Escolhas de implementação

- Box-Tidwell: cada preditor numérico positivo é testado separadamente, acrescentando x·ln(x) ao
  modelo completo; p do z de Wald desse termo (o `car::boxTidwell` é só para `lm`; a referência
  é o `glm` do R com o termo acrescentado). Valores ≤ 0 → "não aplicável".
- Hosmer-Lemeshow: grupos pelos quantis (tipo 7) das probabilidades previstas, `cut` com
  intervalos fechados à direita e o primeiro fechado dos dois lados; gl = (grupos não vazios) − 2.
- IC da probabilidade na Simulação: η̂ ± z·EP(η̂) na escala do logit (z normal, como no Wald dos
  coeficientes), convertido por 1/(1 + e^(−η)).
- Separação: completa quando o preditor linear separa as classes sem empate na fronteira; quase
  completa quando separa com empate. Ambas geram aviso (o ajuste é mantido).
- EPV = (número da classe menos frequente) / (número de coeficientes sem o intercepto).
