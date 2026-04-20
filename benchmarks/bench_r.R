# TBATS benchmark driver for forecast::tbats.
#
# Usage: Rscript bench_r.R <series.csv> <out.json> <mode> <use_trend> <use_damping> <periods> <k_vector> [h]
#   mode        : "auto" (forecast::tbats full search) or "fixed" (fitSpecificTBATS, pinned k)
#   use_trend   : "TRUE" / "FALSE"
#   use_damping : "TRUE" / "FALSE"
#   periods     : comma-separated, e.g. "24,168"
#   k_vector    : comma-separated, e.g. "3,5"
#   h           : optional integer forecast horizon; if given, emits h-step point forecasts
#
# Output: JSON with wall_time_s, ssr, aic, n, chosen_harmonics (auto only), forecast (if h).

suppressPackageStartupMessages({
  library(forecast)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7) stop("need >=7 args: series.csv out.json mode use_trend use_damping periods k_vector [h]")

series_path  <- args[1]
out_path     <- args[2]
mode         <- args[3]
use_trend    <- as.logical(args[4])
use_damping  <- as.logical(args[5])
periods      <- as.numeric(strsplit(args[6], ",")[[1]])
k_vector     <- as.integer(strsplit(args[7], ",")[[1]])
h            <- if (length(args) >= 8) as.integer(args[8]) else 0L

y <- scan(series_path, quiet = TRUE)
n <- length(y)

use_box_cox <- if (length(args) >= 9) as.logical(args[9]) else FALSE

fit_auto <- function() {
  t0 <- proc.time()
  model <- tbats(
    y,
    seasonal.periods   = periods,
    use.trend          = use_trend,
    use.damped.trend   = use_damping,
    use.box.cox        = use_box_cox,
    use.arma.errors    = FALSE,
    use.parallel       = FALSE
  )
  wall <- as.numeric((proc.time() - t0)["elapsed"])
  list(model = model, wall = wall, chosen_k = model$k.vector)
}

fit_fixed <- function() {
  # Direct call to internal fixed-structure fitter; structure is fully pinned.
  t0 <- proc.time()
  model <- forecast:::fitSpecificTBATS(
    y,
    use.box.cox      = use_box_cox,
    use.beta         = use_trend,
    use.damping      = use_damping,
    seasonal.periods = periods,
    k.vector         = k_vector
  )
  wall <- as.numeric((proc.time() - t0)["elapsed"])
  list(model = model, wall = wall, chosen_k = k_vector)
}

res <- switch(mode,
  auto  = fit_auto(),
  fixed = fit_fixed(),
  stop(paste("unknown mode:", mode))
)

model <- res$model
resid <- as.numeric(residuals(model))
ssr   <- sum(resid ^ 2)
aic   <- tryCatch(as.numeric(model$AIC), error = function(e) NA_real_)
if (is.null(aic) || is.na(aic)) {
  # fitSpecificTBATS return: check likelihood field
  aic <- tryCatch(as.numeric(model$aic), error = function(e) NA_real_)
}

fcast <- NULL
if (h > 0) {
  fcast <- tryCatch(
    as.numeric(forecast(model, h = h)$mean),
    error = function(e) NULL
  )
}

# Extract fitted params for cross-backend comparison
params <- list(
  alpha   = tryCatch(as.numeric(model$alpha), error = function(e) NA_real_),
  beta    = tryCatch(as.numeric(model$beta), error = function(e) NA_real_),
  phi     = tryCatch(as.numeric(model$damping.parameter), error = function(e) NA_real_),
  gamma1  = tryCatch(as.numeric(model$gamma.one.values), error = function(e) NULL),
  gamma2  = tryCatch(as.numeric(model$gamma.two.values), error = function(e) NULL),
  x_final = tryCatch(as.numeric(model$x[, ncol(model$x)]), error = function(e) NULL),
  x_nought = tryCatch(as.numeric(model$x[, 1]), error = function(e) NULL)
)

out <- list(
  mode           = mode,
  wall_time_s    = res$wall,
  ssr            = ssr,
  aic            = aic,
  n              = n,
  neg_log_lik    = n * log(ssr / n),
  chosen_k       = res$chosen_k,
  forecast       = fcast,
  params         = params,
  forecast_ver   = as.character(packageVersion("forecast")),
  r_ver          = paste0(R.version$major, ".", R.version$minor)
)

# Minimal JSON writer so we don't depend on jsonlite
to_json <- function(x) {
  if (is.null(x) || (length(x) == 1 && is.na(x))) return("null")
  if (is.list(x) && !is.null(names(x))) {
    kv <- mapply(function(k, v) paste0('"', k, '":', to_json(v)), names(x), x, SIMPLIFY = TRUE)
    return(paste0("{", paste(kv, collapse = ","), "}"))
  }
  if (is.logical(x) && length(x) == 1) return(if (x) "true" else "false")
  if (is.numeric(x) && length(x) == 1) return(format(x, digits = 15, scientific = FALSE))
  if (is.character(x) && length(x) == 1) return(paste0('"', gsub('"', '\\\\"', x), '"'))
  if (length(x) > 1 || is.list(x)) {
    return(paste0("[", paste(sapply(x, to_json), collapse = ","), "]"))
  }
  stop("don't know how to serialize")
}

writeLines(to_json(out), out_path)
