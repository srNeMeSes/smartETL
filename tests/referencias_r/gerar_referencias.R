# Gera os valores de referência da regressão linear (tests/core/test_regressao.py).
#
# Rodado uma vez na máquina de desenvolvimento com R 4.6.1, lmtest 0.9.40, car 3.1.5 e
# nortest 1.0-4. O app e o pytest NÃO dependem do R: os resultados ficam gravados em
# referencias.json e os dados exatos usados, nos CSV desta pasta.
#
#   Rscript tests/referencias_r/gerar_referencias.R
#
# Datasets públicos: mtcars, cars e longley (pacote datasets do R) e Prestige (carData).
# grande.csv (n = 6000, gerado com numpy, semente 2026) exercita o Lilliefors (n > 5000).
# Níveis de referência = o nível mais frequente (padrão do app).

lib <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "smartetl-lib")
.libPaths(c(lib, .libPaths()))
suppressMessages({
  library(lmtest)
  library(car)
  library(nortest)
})

args <- commandArgs(trailingOnly = FALSE)
script <- sub("--file=", "", args[grep("--file=", args)])
pasta <- if (length(script)) dirname(normalizePath(script)) else "tests/referencias_r"

# ------------------------------------------------------------------ dados
mt <- data.frame(
  mpg = mtcars$mpg, wt = mtcars$wt, hp = mtcars$hp,
  cyl = paste(mtcars$cyl, "cilindros"),
  am = ifelse(mtcars$am == 1, "Manual", "Automático")
)
write.csv(mt, file.path(pasta, "mtcars.csv"), row.names = FALSE, fileEncoding = "UTF-8")

pr <- data.frame(
  prestige = Prestige$prestige, education = Prestige$education,
  income = Prestige$income, type = as.character(Prestige$type)
)
write.csv(pr, file.path(pasta, "prestige.csv"), row.names = FALSE, na = "",
          fileEncoding = "UTF-8")

lo <- longley
names(lo) <- gsub(".", "_", names(lo), fixed = TRUE)
write.csv(lo, file.path(pasta, "longley.csv"), row.names = FALSE, fileEncoding = "UTF-8")

write.csv(cars, file.path(pasta, "cars.csv"), row.names = FALSE, fileEncoding = "UTF-8")

gr <- read.csv(file.path(pasta, "grande.csv"), fileEncoding = "UTF-8")

mais_frequente <- function(x) names(sort(table(x), decreasing = TRUE))[1]
como_fator <- function(df) {
  for (nome in names(df)) {
    if (is.character(df[[nome]])) {
      f <- factor(df[[nome]])
      df[[nome]] <- relevel(f, ref = mais_frequente(df[[nome]]))
    }
  }
  df
}

# ------------------------------------------------------------------ JSON mínimo
num <- function(v) ifelse(is.na(v), "null", sprintf("%.17g", v))
json <- function(x) {
  if (is.list(x)) {
    itens <- vapply(names(x), function(k) paste0("\"", k, "\": ", json(x[[k]])), "")
    return(paste0("{", paste(itens, collapse = ", "), "}"))
  }
  if (is.character(x)) {
    textos <- paste0("\"", x, "\"")
    return(if (length(x) == 1) textos else paste0("[", paste(textos, collapse = ", "), "]"))
  }
  if (length(x) == 1) return(num(x))
  paste0("[", paste(num(x), collapse = ", "), "]")
}

# ------------------------------------------------------------------ referências
referencia <- function(formula, dados, nova, ordenar_por = NULL) {
  dados <- como_fator(dados)
  m <- lm(formula, data = dados)
  s <- summary(m)
  cf <- coef(s)
  fs <- s$fstatistic
  ic95 <- confint(m, level = 0.95)
  ic90 <- confint(m, level = 0.90)
  n <- nobs(m)
  r <- list(
    n = n,
    termos = rownames(cf),
    estimativa = unname(cf[, 1]), ep = unname(cf[, 2]),
    t = unname(cf[, 3]), p = unname(cf[, 4]),
    ic95_li = unname(ic95[, 1]), ic95_ls = unname(ic95[, 2]),
    ic90_li = unname(ic90[, 1]), ic90_ls = unname(ic90[, 2]),
    r2 = s$r.squared, r2_ajustado = s$adj.r.squared, rmse = s$sigma,
    f = unname(fs[1]), gl1 = unname(fs[2]), gl2 = unname(fs[3]),
    p_f = unname(pf(fs[1], fs[2], fs[3], lower.tail = FALSE))
  )
  bp <- bptest(m)
  r$bp <- list(estatistica = unname(bp$statistic), gl = unname(bp$parameter),
               p = bp$p.value)
  ajustados <- fitted(m)
  gq <- gqtest(m, order.by = ajustados)
  r$gq_ajustados <- list(estatistica = unname(gq$statistic), gl1 = unname(gq$parameter[1]),
                         gl2 = unname(gq$parameter[2]), p = gq$p.value)
  hmc <- hmctest(m, order.by = ajustados, simulate.p = FALSE)
  set.seed(1)
  hmc_p <- hmctest(m, order.by = ajustados, nsim = 100000)$p.value
  r$hmc_ajustados <- list(estatistica = unname(hmc$statistic), p_100000 = hmc_p)
  if (!is.null(ordenar_por)) {
    z <- model.frame(m)[[ordenar_por]]
    gq2 <- gqtest(m, order.by = z)
    hmc2 <- hmctest(m, order.by = z, simulate.p = FALSE)
    r$ordenar_por <- ordenar_por
    r$gq_variavel <- list(estatistica = unname(gq2$statistic), p = gq2$p.value)
    r$hmc_variavel <- list(estatistica = unname(hmc2$statistic))
  }
  r$dw <- unname(dwtest(m)$statistic)
  bg <- bgtest(m, order = 1)
  r$bg <- list(estatistica = unname(bg$statistic), gl = unname(bg$parameter), p = bg$p.value)
  res <- residuals(m)
  if (n <= 5000) {
    sw <- shapiro.test(res)
    r$normalidade <- list(teste = "shapiro", estatistica = unname(sw$statistic), p = sw$p.value)
  } else {
    lf <- lillie.test(res)
    r$normalidade <- list(teste = "lilliefors", estatistica = unname(lf$statistic),
                          p = lf$p.value)
  }
  v <- if (length(attr(terms(m), "term.labels")) >= 2) vif(m) else NULL
  if (is.null(v)) {
    r$vif <- "nenhum"
  } else if (is.matrix(v)) {
    r$vif <- list(variaveis = rownames(v), gvif = unname(v[, 1]), gl = unname(v[, 2]),
                  gvif_ajustado = unname(v[, 3]))
  } else {
    r$vif <- list(variaveis = names(v), gvif = unname(v), gl = rep(1, length(v)),
                  gvif_ajustado = unname(sqrt(v)))
  }
  nova <- como_fator_novo(nova, dados)
  pc <- predict(m, newdata = nova, interval = "confidence", level = 0.95)
  pp <- predict(m, newdata = nova, interval = "prediction", level = 0.95)
  r$previsao <- list(valor = unname(pc[1, 1]), ic_li = unname(pc[1, 2]),
                     ic_ls = unname(pc[1, 3]), ip_li = unname(pp[1, 2]),
                     ip_ls = unname(pp[1, 3]))
  r
}

como_fator_novo <- function(nova, dados) {
  for (nome in names(nova)) {
    if (is.factor(dados[[nome]])) nova[[nome]] <- factor(nova[[nome]], levels = levels(dados[[nome]]))
  }
  nova
}

refs <- list(
  mtcars = referencia(mpg ~ wt + hp + cyl + am, mt,
                      data.frame(wt = 3, hp = 150, cyl = "6 cilindros", am = "Manual"),
                      ordenar_por = "wt"),
  prestige = referencia(prestige ~ education + income + type, pr,
                        data.frame(education = 11, income = 7000, type = "wc"),
                        ordenar_por = "income"),
  longley = referencia(Employed ~ GNP_deflator + GNP + Unemployed + Armed_Forces +
                         Population + Year, lo,
                       data.frame(GNP_deflator = 100, GNP = 400, Unemployed = 300,
                                  Armed_Forces = 250, Population = 117, Year = 1955),
                       ordenar_por = "Year"),
  cars = referencia(dist ~ speed, cars, data.frame(speed = 21), ordenar_por = "speed"),
  grande = referencia(y ~ x1 + x2 + g, gr, data.frame(x1 = 10, x2 = 2.5, g = "B"))
)

# Troca do nível de referência (Prestige, type = "prof"): mesmos R², outros coeficientes.
pr2 <- como_fator(pr)
pr2$type <- relevel(pr2$type, ref = "prof")
m2 <- lm(prestige ~ education + income + type, data = pr2)
refs$prestige_ref_prof <- list(termos = names(coef(m2)), estimativa = unname(coef(m2)),
                               r2 = summary(m2)$r.squared)

versoes <- list(R = paste(R.version$major, R.version$minor, sep = "."),
                lmtest = as.character(packageVersion("lmtest")),
                car = as.character(packageVersion("car")),
                nortest = as.character(packageVersion("nortest")))
refs$versoes <- versoes

writeLines(json(refs), file.path(pasta, "referencias.json"), useBytes = TRUE)
cat("referencias.json gravado em", pasta, "\n")
