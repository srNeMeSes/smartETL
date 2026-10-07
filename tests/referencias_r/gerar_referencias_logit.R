# Gera os valores de referência da regressão logística (tests/core/test_regressao_logit.py).
#
# Rodado uma vez com R 4.6.1, car 3.1.5, ResourceSelection 0.3.6 e pROC 1.19.1. O app e o pytest
# NÃO dependem do R: os resultados ficam em referencias_logit.json e os dados nos CSV desta pasta.
#
#   Rscript tests/referencias_r/gerar_referencias_logit.R
#
# Datasets públicos: Mroz (carData) e birthwt (MASS). Níveis de referência = o mais frequente
# (padrão do app); o evento é "yes" (Mroz) e 1 (birthwt).

lib <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "smartetl-lib")
.libPaths(c(lib, .libPaths()))
suppressMessages({
  library(car)
  library(ResourceSelection)
  library(pROC)
  library(MASS)
})

args <- commandArgs(trailingOnly = FALSE)
script <- sub("--file=", "", args[grep("--file=", args)])
pasta <- if (length(script)) dirname(normalizePath(script)) else "tests/referencias_r"

# ------------------------------------------------------------------ dados
mroz <- data.frame(
  lfp = as.character(Mroz$lfp), k5 = Mroz$k5, k618 = Mroz$k618, age = Mroz$age,
  wc = as.character(Mroz$wc), hc = as.character(Mroz$hc), lwg = Mroz$lwg, inc = Mroz$inc
)
write.csv(mroz, file.path(pasta, "mroz.csv"), row.names = FALSE, fileEncoding = "UTF-8")

bw <- data.frame(
  low = birthwt$low, age = birthwt$age, lwt = birthwt$lwt,
  race = c("branca", "negra", "outra")[birthwt$race],
  smoke = ifelse(birthwt$smoke == 1, "sim", "não"), ht = birthwt$ht, ui = birthwt$ui
)
write.csv(bw, file.path(pasta, "birthwt.csv"), row.names = FALSE, fileEncoding = "UTF-8")

mais_frequente <- function(x) names(sort(table(x), decreasing = TRUE))[1]
como_fator <- function(df, y) {
  for (nome in setdiff(names(df), y)) {
    if (is.character(df[[nome]])) {
      df[[nome]] <- relevel(factor(df[[nome]]), ref = mais_frequente(df[[nome]]))
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
# Convergência apertada: com o padrão (epsilon = 1e-8) o IRLS para antes do máximo e a covariância
# usa os pesos da penúltima iteração (EP diferentes em ~1e-5); o app converge até o máximo.
controle <- glm.control(epsilon = 1e-14, maxit = 100)

referencia <- function(dados, y, evento, preditores, nova) {
  dados <- como_fator(dados, y)
  dados$.y <- as.numeric(as.character(dados[[y]]) == evento)
  formula <- reformulate(preditores, response = ".y")
  m <- glm(formula, data = dados, family = binomial, control = controle)
  s <- summary(m)
  cf <- coef(s)
  n <- nobs(m)
  ll <- as.numeric(logLik(m))
  m0 <- glm(.y ~ 1, data = model.frame(m), family = binomial, control = controle)
  ll0 <- as.numeric(logLik(m0))
  lr <- 2 * (ll - ll0)
  gl <- length(coef(m)) - 1
  cox_snell <- 1 - exp(2 * (ll0 - ll) / n)
  p <- fitted(m)
  yy <- model.frame(m)$.y
  hl <- suppressWarnings(hoslem.test(yy, p, g = 10))
  r <- list(
    n = n, eventos = sum(yy),
    termos = rownames(cf),
    estimativa = unname(cf[, 1]), ep = unname(cf[, 2]), z = unname(cf[, 3]), p = unname(cf[, 4]),
    ic95_li = unname(confint.default(m, level = 0.95)[, 1]),
    ic95_ls = unname(confint.default(m, level = 0.95)[, 2]),
    ic90_li = unname(confint.default(m, level = 0.90)[, 1]),
    ic90_ls = unname(confint.default(m, level = 0.90)[, 2]),
    loglik = ll, loglik_nulo = ll0, lr = lr, gl = gl,
    p_lr = pchisq(lr, gl, lower.tail = FALSE),
    aic = AIC(m), bic = BIC(m),
    mcfadden = 1 - ll / ll0, cox_snell = cox_snell,
    nagelkerke = cox_snell / (1 - exp(2 * ll0 / n)),
    hl = list(estatistica = unname(hl$statistic), gl = unname(hl$parameter), p = hl$p.value),
    auc = as.numeric(auc(roc(yy, p, quiet = TRUE, direction = "<"))),
    residuos_deviance_5 = unname(residuals(m, type = "deviance")[1:5]),
    ajustados_5 = unname(p[1:5])
  )
  previsto <- as.numeric(p >= 0.5)
  r$confusao_05 <- list(
    vp = sum(previsto == 1 & yy == 1), fn = sum(previsto == 0 & yy == 1),
    fp = sum(previsto == 1 & yy == 0), vn = sum(previsto == 0 & yy == 0)
  )
  v <- vif(m)
  if (is.matrix(v)) {
    r$vif <- list(variaveis = rownames(v), gvif = unname(v[, 1]), gl = unname(v[, 2]),
                  gvif_ajustado = unname(v[, 3]))
  } else {
    r$vif <- list(variaveis = names(v), gvif = unname(v), gl = rep(1, length(v)),
                  gvif_ajustado = unname(sqrt(v)))
  }
  # Box-Tidwell: x·ln(x) acrescentado ao modelo completo, um preditor numérico positivo por vez.
  bt_var <- c(); bt_z <- c(); bt_p <- c()
  for (x in preditores) {
    valores <- model.frame(m)[[x]]
    if (is.numeric(valores) && all(valores > 0)) {
      dados$.xlnx <- dados[[x]] * log(dados[[x]])
      m_bt <- glm(update(formula, . ~ . + .xlnx), data = dados, family = binomial,
                  control = controle)
      linha <- coef(summary(m_bt))[".xlnx", ]
      bt_var <- c(bt_var, x); bt_z <- c(bt_z, linha[3]); bt_p <- c(bt_p, linha[4])
    }
  }
  r$box_tidwell <- list(variaveis = bt_var, z = unname(bt_z), p = unname(bt_p))
  for (nome in names(nova)) {
    if (is.factor(dados[[nome]])) nova[[nome]] <- factor(nova[[nome]], levels = levels(dados[[nome]]))
  }
  pr <- predict(m, newdata = nova, type = "link", se.fit = TRUE)
  zc <- qnorm(0.975)
  r$previsao <- list(
    logit = unname(pr$fit), ep = unname(pr$se.fit), prob = plogis(unname(pr$fit)),
    ic_li = plogis(unname(pr$fit - zc * pr$se.fit)), ic_ls = plogis(unname(pr$fit + zc * pr$se.fit))
  )
  r
}

refs <- list(
  mroz = referencia(mroz, "lfp", "yes", c("k5", "k618", "age", "wc", "hc", "lwg", "inc"),
                    data.frame(k5 = 1, k618 = 2, age = 40, wc = "yes", hc = "no", lwg = 1.2,
                               inc = 20)),
  birthwt = referencia(bw, "low", "1", c("age", "lwt", "race", "smoke", "ht", "ui"),
                       data.frame(age = 25, lwt = 120, race = "negra", smoke = "sim", ht = 0,
                                  ui = 1))
)
refs$versoes <- list(
  R = paste(R.version$major, R.version$minor, sep = "."),
  car = as.character(packageVersion("car")),
  ResourceSelection = as.character(packageVersion("ResourceSelection")),
  pROC = as.character(packageVersion("pROC"))
)
writeLines(json(refs), file.path(pasta, "referencias_logit.json"), useBytes = TRUE)
cat("referencias_logit.json gravado em", pasta, "\n")
