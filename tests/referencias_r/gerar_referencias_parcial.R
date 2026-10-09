# Gera os valores de referência da correlação parcial (tests/core/test_correlacao_parcial.py).
#
# Rodado uma vez com R 4.6.1 e ppcor 1.1 (pcor.test, Pearson e Spearman). O pytest não depende
# do R: os resultados ficam em referencias_parcial.json e os dados em parcial_*.csv nesta pasta.
#
#   Rscript tests/referencias_r/gerar_referencias_parcial.R
#
# Bases: mtcars (x = wt, y = mpg; controles hp e hp + disp) e iris (x = Sepal.Length,
# y = Petal.Length, controle Petal.Width; com empates). O pcor.test devolve o p bilateral do t
# com n − 2 − k gl; os unilaterais são conferidos pela mesma estatística t no pytest.

args <- commandArgs(trailingOnly = FALSE)
script <- sub("--file=", "", args[grep("--file=", args)])
pasta <- if (length(script)) dirname(normalizePath(script)) else "tests/referencias_r"
lib <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "smartetl-lib")
.libPaths(c(lib, .libPaths()))
library(ppcor)

num <- function(v) sprintf("%.17g", v)

dados <- list(
  mtcars = data.frame(x = mtcars$wt, y = mtcars$mpg, z1 = mtcars$hp, z2 = mtcars$disp),
  iris = data.frame(x = iris$Sepal.Length, y = iris$Petal.Length, z1 = iris$Petal.Width)
)
casos <- list(
  mtcars_1 = list(base = "mtcars", controles = "z1"),
  mtcars_2 = list(base = "mtcars", controles = c("z1", "z2")),
  iris_1 = list(base = "iris", controles = "z1")
)

for (nome in names(dados)) {
  write.csv(dados[[nome]], file.path(pasta, paste0("parcial_", nome, ".csv")), row.names = FALSE)
}

itens <- character(0)
for (caso in names(casos)) {
  d <- dados[[casos[[caso]]$base]]
  z <- d[, casos[[caso]]$controles, drop = FALSE]
  partes <- character(0)
  for (metodo in c("pearson", "spearman")) {
    r <- pcor.test(d$x, d$y, z, method = metodo)
    simples <- cor(d$x, d$y, method = metodo)
    partes <- c(partes, sprintf(
      "\"%s\": {\"r\": %s, \"t\": %s, \"p\": %s, \"n\": %d, \"k\": %d, \"simples\": %s}",
      metodo, num(r$estimate), num(r$statistic), num(r$p.value), r$n, r$gp, num(simples)
    ))
  }
  itens <- c(itens, sprintf("  \"%s\": {%s}", caso, paste(partes, collapse = ", ")))
}
writeLines(paste0("{\n", paste(itens, collapse = ",\n"), "\n}"),
           file.path(pasta, "referencias_parcial.json"))
