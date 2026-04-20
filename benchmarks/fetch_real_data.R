# Dump forecast::taylor to CSV for cross-backend benchmarking.
# Also dumps a hint file with the canonical seasonal periods.

suppressPackageStartupMessages(library(forecast))

outdir <- if (length(commandArgs(trailingOnly = TRUE)) >= 1)
          commandArgs(trailingOnly = TRUE)[1] else "data"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# taylor: half-hourly electricity demand, England/Wales, 2000-06-05 to 2000-08-27
# 12 weeks * 7 days * 48 half-hours = 4032 observations
# Two seasonal periods: daily (48) and weekly (336)
y <- as.numeric(taylor)
writeLines(formatC(y, digits = 12, format = "g"),
           file.path(outdir, "taylor.csv"))

meta <- list(
  name       = "taylor",
  n          = length(y),
  periods    = c(48, 336),
  source     = "forecast::taylor",
  description = "Half-hourly electricity demand, England/Wales 2000-06-05 to 2000-08-27"
)
jsn <- paste0("{",
  paste0('"', names(meta), '":', c(
    paste0('"', meta$name, '"'),
    meta$n,
    paste0("[", paste(meta$periods, collapse = ","), "]"),
    paste0('"', meta$source, '"'),
    paste0('"', meta$description, '"')
  ), collapse = ","),
  "}")
writeLines(jsn, file.path(outdir, "taylor.json"))

cat("wrote", length(y), "points to", file.path(outdir, "taylor.csv"), "\n")
