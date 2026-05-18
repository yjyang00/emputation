# scripts/setup.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(torch)
  library(mice)
  library(missForest)
  library(energy)
  library(kernlab)
  library(readxl)
})

source("src/R/specs/get_ccmv_spec.R")
source("src/R/emputation/helper_pattern.R")
source("src/R/emputation/dftomat.R")
source("src/R/emputation/ImpEngfit.R")
source("src/R/emputation/ImpEng.R")
source("src/R/emputation/predict.R")
source("src/R/emputation/ccmv.R")
source("src/R/emputation/mar.R")
source("src/R/emputation/mcar.R")
source("src/R/emputation/memory.R")
source("src/R/emputation/em_gaussian.R")
source("scripts/config.R")


# helpers
ensure_dir = function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
  invisible(path)
}

# root folders
RUN_ROOT     = "run"
RESULTS_ROOT = "results"
ensure_dir(RUN_ROOT)
ensure_dir(RESULTS_ROOT)

# scenario base folder under run/
run_scenario_dir = function(dataset, dgp, train) {
  file.path(RUN_ROOT, dataset, paste0("dgp_", dgp), paste0("train_", train))
}

iter_base_rda = function(dataset, dgp, train, iter) {
  file.path(run_scenario_dir(dataset, dgp, train), sprintf("iter_%03d.rda", iter))
}

iter_emputation_rda = function(dataset, dgp, train, iter) {
  file.path(run_scenario_dir(dataset, dgp, train), sprintf("iter_%03d_emputation.rda", iter))
}

iter_gain_csv = function(dataset, dgp, train, iter) {
  file.path(run_scenario_dir(dataset, dgp, train), sprintf("iter_%03d_gain.csv", iter))
}

iter_missforest_rda = function(dataset, dgp, train, iter) {
  file.path(run_scenario_dir(dataset, dgp, train), sprintf("iter_%03d_missForest.rda", iter))
}

results_scenario_dir = function(dataset, dgp, train) {
  file.path(RESULTS_ROOT, dataset, paste0("dgp_", dgp), paste0("train_", train))
}

eval_shard_rda = function(dataset, dgp, train, task_id) {
  file.path(results_scenario_dir(dataset, dgp, train), sprintf("res_%03d.rda", task_id))
}

eval_agg_rda = function(dataset, dgp, train) {
  file.path(results_scenario_dir(dataset, dgp, train), "eval.rda")
}

# data loader
load_dataset = function(dataset) {
  dataset = tolower(dataset)
  if (is.null(cfg$data_files[[dataset]])) {
    stop("No data file configured for dataset: ", dataset)
  }
  f = cfg$data_files[[dataset]]
  if (!file.exists(f)) stop("Data file not found: ", f)
  
  if (grepl("\\.csv$", f, ignore.case = TRUE)) {
    sep = if (grepl("wine", f)) ";" else ","
    dat = read.csv(f, sep = sep, stringsAsFactors = FALSE)
    return(as.data.frame(dat))
  }
  
  if (grepl("\\.xlsx$", f, ignore.case = TRUE)) {
    sheet = cfg$xlsx_sheet[[dataset]]
    dat = readxl::read_xlsx(f, sheet = sheet)
    return(as.data.frame(dat))
  }
  
  stop("Unsupported data extension: ", f)
}


# Scenario lookup
get_scenario = function(scenario_id) {
  # 1-based index
  if (scenario_id < 1 || scenario_id > length(cfg$scenarios)) {
    stop("scenario_id out of range: ", scenario_id)
  }
  cfg$scenarios[[scenario_id]]
}

# Emputation
get_emputation_params = function(dataset) {
  base = cfg$emputation_default
  overrides = cfg$emputation_override[[dataset]]
  if (!is.null(overrides)) {
    for (nm in names(overrides)) base[[nm]] = overrides[[nm]]
  }
  return(base)
}