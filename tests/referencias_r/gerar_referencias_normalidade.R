# Gera os valores de referência do Shapiro-Wilk nos testes de médias e nas ANOVAs
# (tests/core/test_normalidade.py).
#
# Rodado uma vez com R 4.6.1 (stats::shapiro.test). O pytest não depende do R: os resultados
# ficam em referencias_normalidade.json e os dados em normalidade_*.csv nesta pasta.
#
#   Rscript tests/referencias_r/gerar_referencias_normalidade.R
#
# Bases do próprio R: mtcars (mpg; mpg por am), sleep (extra pareado por ID), PlantGrowth
# (weight por group) e ToothGrowth (len por supp × dose). Nas ANOVAs, o teste é aplicado aos
# resíduos do modelo (aov), com e sem interação.

args <- commandArgs(trailingOnly = FALSE)
script <- sub("--file=", "", args[grep("--file=", args)])
pasta <- if (length(script)) dirname(normalizePath(script)) else "tests/referencias_r"

sw <- function(v) {
  r <- shapiro.test(v)
  sprintf("{\"W\": %.17g, \"p\": %.17g, \"n\": %d}", unname(r$statistic), r$p.value, length(v))
}

mt <- data.frame(mpg = mtcars$mpg, am = ifelse(mtcars$am == 1, "manual", "automatico"))
write.csv(mt, file.path(pasta, "normalidade_mtcars.csv"), row.names = FALSE)

antes <- sleep$extra[sleep$group == 1]
depois <- sleep$extra[sleep$group == 2]
write.csv(data.frame(antes = antes, depois = depois),
          file.path(pasta, "normalidade_sleep.csv"), row.names = FALSE)

write.csv(data.frame(peso = PlantGrowth$weight, grupo = as.character(PlantGrowth$group)),
          file.path(pasta, "normalidade_plantgrowth.csv"), row.names = FALSE)

tg <- data.frame(len = ToothGrowth$len, supp = as.character(ToothGrowth$supp),
                 dose = paste0("d", ToothGrowth$dose))
write.csv(tg, file.path(pasta, "normalidade_toothgrowth.csv"), row.names = FALSE)

itens <- c(
  t_1am = sw(mt$mpg),
  t_2am_automatico = sw(mt$mpg[mt$am == "automatico"]),
  t_2am_manual = sw(mt$mpg[mt$am == "manual"]),
  t_pareado = sw(antes - depois),
  anova_1fator = sw(residuals(aov(peso ~ grupo,
    data = data.frame(peso = PlantGrowth$weight, grupo = PlantGrowth$group)))),
  anova_2fator_interacao = sw(residuals(aov(len ~ supp * dose, data = tg))),
  anova_2fator_aditivo = sw(residuals(aov(len ~ supp + dose, data = tg)))
)
corpo <- paste0("  \"", names(itens), "\": ", itens, collapse = ",\n")
writeLines(paste0("{\n", corpo, "\n}"), file.path(pasta, "referencias_normalidade.json"))
