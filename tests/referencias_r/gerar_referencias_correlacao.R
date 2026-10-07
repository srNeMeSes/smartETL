# Gera os valores de referência das correlações (tests/core/test_correlacao.py).
#
# Rodado uma vez com R 4.6.1 (stats::cor.test). O pytest não depende do R: os resultados ficam
# em referencias_correlacao.json e os dados em correlacao_*.csv nesta pasta.
#
#   Rscript tests/referencias_r/gerar_referencias_correlacao.R
#
# Pearson: cor.test(method = "pearson") — t com n − 2 gl e IC pela z de Fisher (unilateral
# quando H₁ é unilateral). Spearman: cor.test(method = "spearman", exact = FALSE) — p pela
# aproximação t (a mesma do scipy.stats.spearmanr); postos médios nos empates.

args <- commandArgs(trailingOnly = FALSE)
script <- sub("--file=", "", args[grep("--file=", args)])
pasta <- if (length(script)) dirname(normalizePath(script)) else "tests/referencias_r"

num <- function(v) ifelse(is.na(v), "null", sprintf("%.17g", v))
json <- function(x) {
  if (is.list(x)) {
    itens <- vapply(names(x), function(k) paste0("\"", k, "\": ", json(x[[k]])), "")
    return(paste0("{", paste(itens, collapse = ", "), "}"))
  }
  if (is.character(x)) return(paste0("\"", x, "\""))
  if (length(x) == 1) return(num(x))
  paste0("[", paste(num(x), collapse = ", "), "]")
}

set.seed(42)
x_out <- c(rnorm(30, 50, 10), 120)
y_out <- c(0.4 * x_out[1:30] + rnorm(30, 0, 6), 5)  # um outlier que derruba a Pearson
dados <- list(
  mtcars = data.frame(x = mtcars$wt, y = mtcars$mpg),
  cars = data.frame(x = cars$speed, y = cars$dist),
  outlier = data.frame(x = round(x_out, 3), y = round(y_out, 3))
)

refs <- list()
for (nome in names(dados)) {
  d <- dados[[nome]]
  write.csv(d, file.path(pasta, paste0("correlacao_", nome, ".csv")), row.names = FALSE)
  r <- list(n = nrow(d))
  for (alt in c("two.sided", "greater", "less")) {
    p <- cor.test(d$x, d$y, method = "pearson", alternative = alt, conf.level = 0.95)
    s <- suppressWarnings(
      cor.test(d$x, d$y, method = "spearman", alternative = alt, exact = FALSE)
    )
    r[[paste0("pearson_", alt)]] <- list(
      r = unname(p$estimate), t = unname(p$statistic), gl = unname(p$parameter),
      p = p$p.value, ic_li = p$conf.int[1], ic_ls = p$conf.int[2]
    )
    r[[paste0("spearman_", alt)]] <- list(rho = unname(s$estimate), p = s$p.value)
  }
  p90 <- cor.test(d$x, d$y, method = "pearson", conf.level = 0.90)
  r$pearson_ic90 <- list(ic_li = p90$conf.int[1], ic_ls = p90$conf.int[2])
  refs[[nome]] <- r
}
refs$versoes <- list(R = paste(R.version$major, R.version$minor, sep = "."))
writeLines(json(refs), file.path(pasta, "referencias_correlacao.json"), useBytes = TRUE)
cat("referencias_correlacao.json gravado em", pasta, "\n")
