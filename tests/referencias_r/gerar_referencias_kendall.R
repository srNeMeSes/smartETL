# Gera os valores de referência da correlação de Kendall (tests/core/test_correlacao_kendall.py).
#
# Rodado uma vez com R 4.6.1 (stats::cor.test, method = "kendall"). O pytest não depende do R: os
# resultados ficam em referencias_kendall.json e os dados em kendall_*.csv nesta pasta.
#
#   Rscript tests/referencias_r/gerar_referencias_kendall.R
#
# Bases: "distintos" (n = 20, sem empates, semente 7: p exato, como o R faz com n < 50 sem
# empates), mtcars (wt × mpg, com empates: aproximação normal) e cars (speed × dist, com
# empates). Com empates, o R usa τ-b e z = S/√var(S) com a correção de empates e sem
# correção de continuidade (o padrão de cor.test).

args <- commandArgs(trailingOnly = FALSE)
script <- sub("--file=", "", args[grep("--file=", args)])
pasta <- if (length(script)) dirname(normalizePath(script)) else "tests/referencias_r"
num <- function(v) sprintf("%.17g", v)

set.seed(7)
x <- round(runif(20, 0, 100), 2)
y <- round(0.5 * x + rnorm(20, 0, 15), 2)
dados <- list(
  distintos = data.frame(x = x, y = y),
  mtcars = data.frame(x = mtcars$wt, y = mtcars$mpg),
  cars = data.frame(x = cars$speed, y = cars$dist)
)

itens <- character(0)
for (nome in names(dados)) {
  d <- dados[[nome]]
  write.csv(d, file.path(pasta, paste0("kendall_", nome, ".csv")), row.names = FALSE)
  partes <- character(0)
  for (alt in c("two.sided", "greater", "less")) {
    r <- suppressWarnings(cor.test(d$x, d$y, method = "kendall", alternative = alt))
    partes <- c(partes, sprintf(
      "\"%s\": {\"tau\": %s, \"estatistica\": %s, \"p\": %s, \"nome\": \"%s\"}",
      alt, num(r$estimate), num(r$statistic), num(r$p.value), names(r$statistic)
    ))
  }
  itens <- c(itens, sprintf("  \"%s\": {\"n\": %d, %s}", nome, nrow(d), paste(partes, collapse = ", ")))
}
writeLines(paste0("{\n", paste(itens, collapse = ",\n"), "\n}"),
           file.path(pasta, "referencias_kendall.json"))
