# run_example.R: to illustrate the pipeline with the CONCRETE dataset.
#
# This script runs the emputation pipeline, covering all six
# scenarios for the concrete dataset (3 DGPs * 2 train mechanisms). 
#
# Methods evaluated: Emputation, MICE, EM, Mean, MissForest
# GAIN is excluded here as it requires a Python environment.
# Further details and code for GAIN are available in the README
#
# Runtime note:
#   Simulations are run for n_iter = 100 iterations per scenario on cluster.
#   All simulation raw results are in results folder.
#   Running emputation locally may require more time than on cluster depending on the machine
#   We recommend n_iter = 10 or lower for a quick pipeline check. 
#   To reproduce results under n_iter = 100 on cluster, follow readme instructions.

# Prerequisites:
#   - R packages: torch, tidyverse, mice, missForest, energy, kernlab, readxl, MASS
#
# Concrete scenarios (IDs 1-6 in scripts/config.R):
#   1: dgp=mcar, train=mcar   
#   2: dgp=ccmv, train=ccmv  
#   3: dgp=mcar, train=ccmv  
#   4: dgp=ccmv, train=mcar
#   5: dgp=mar,  train=mcar
#   6: dgp=mar,  train=ccmv
# -----------------------------------------------------------------------


n_iter    <- 100    # can set to 10 or lower for quicker check
overwrite <- 1     # 1 = rerun even if output exists; 0 = skip existing

concrete_scenario_ids <- 1:6

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
rscript <- file.path(R.home("bin"), "Rscript")
run_script <- function(script, args) {
  status <- system2(rscript, args = c(script, as.character(args)),
                    stdout = "", stderr = "")
  if (status != 0) {
    stop("Script failed (exit status ", status, "): ", script,
         "\n  args: ", paste(args, collapse = " "))
  }
  invisible(status)
}
ts <- function() format(Sys.time(), "%H:%M:%S")

# -----------------------------------------------------------------------
# Sanity check: must be run from project root
# -----------------------------------------------------------------------
if (!file.exists("scripts/setup.R")) {
  stop("scripts/setup.R not found.\n")
}

# -----------------------------------------------------------------------
cat("=======================================================\n")
cat("  Emputation pipeline — CONCRETE dataset\n")
cat("  Scenarios  :", paste(concrete_scenario_ids, collapse = ", "), "\n")
cat("  Iterations :", n_iter, "\n")
cat("  Started    :", ts(), "\n")
cat("=======================================================\n\n")

# -----------------------------------------------------------------------
# Steps 1-2: Generate missing data + run Emputation
# -----------------------------------------------------------------------
cat("--- Steps 1-2: Missing data generation + Emputation ---\n\n")

for (scenario_id in concrete_scenario_ids) {
  cat("[", ts(), "] Scenario", scenario_id, "— starting\n")
  
  for (iter in seq_len(n_iter)) {
    cat("  iter", sprintf("%3d", iter), "/", n_iter, "\n")
    
    # Step 1: generate missing data for this (scenario, iter)
    tryCatch(
      run_script("scripts/01_make_missing.R", c(
        "--scenario_id", scenario_id,
        "--iter",        iter,
        "--overwrite",   overwrite
      )),
      error = function(e) cat("  WARNING [01]:", conditionMessage(e), "\n")
    )
    
    # Step 2: fit Emputation and produce multiple imputations
    tryCatch(
      run_script("scripts/02_run_emputation.R", c(
        "--scenario_id", scenario_id,
        "--iter",        iter,
        "--overwrite",   overwrite
      )),
      error = function(e) cat("  WARNING [02]:", conditionMessage(e), "\n")
    )
  }
  
  cat("[", ts(), "] Scenario", scenario_id, "— done\n\n")
}

# -----------------------------------------------------------------------
# Step 3: Evaluate all methods per (scenario, iter)
#
# Methods run: Emputation, MICE, EM, Mean, MissForest
# GAIN: skipped automatically when no gain CSV is present in run/
# -----------------------------------------------------------------------
cat("--- Step 3: Evaluation ---\n\n")

for (scenario_id in concrete_scenario_ids) {
  cat("[", ts(), "] Evaluating scenario", scenario_id, "\n")
  
  for (iter in seq_len(n_iter)) {
    cat("  iter", sprintf("%3d", iter), "/", n_iter, "\n")
    
    tryCatch(
      run_script("scripts/05_eval.R", c(
        "--scenario_id", scenario_id,
        "--iter",        iter,
        "--overwrite",   overwrite
      )),
      error = function(e) cat("  WARNING [05]:", conditionMessage(e), "\n")
    )
  }
  
  cat("[", ts(), "] Scenario", scenario_id, "— done\n\n")
}

# -----------------------------------------------------------------------
# Step 4: Aggregate per-iter shards into one eval.rda per scenario
# -----------------------------------------------------------------------
cat("--- Step 4: Aggregation ---\n\n")

for (scenario_id in concrete_scenario_ids) {
  cat("[", ts(), "] Aggregating scenario", scenario_id, "\n")
  
  tryCatch(
    run_script("scripts/06_aggregate_eval.R", c(
      "--scenario_id", scenario_id,
      "--overwrite",   overwrite
    )),
    error = function(e) cat("  WARNING [06]:", conditionMessage(e), "\n")
  )
}

# -----------------------------------------------------------------------
# Summary: load path helpers and print output locations
# -----------------------------------------------------------------------
source("scripts/setup.R")

cat("\n=======================================================\n")
cat("  Pipeline complete!\n")
cat("  Finished:", ts(), "\n\n")
cat("  Output files (eval.rda):\n")

all_ok <- TRUE
for (scenario_id in concrete_scenario_ids) {
  sc <- get_scenario(scenario_id)
  f  <- eval_agg_rda(sc$dataset, sc$dgp, sc$train)
  status_str <- if (file.exists(f)) "  OK  " else " MISSING "
  if (!file.exists(f)) all_ok <- FALSE
  cat(sprintf("    [%s] %s\n", status_str, f))
}

cat("\n")
if (all_ok) {
  cat("  All output files present.\n")
} else {
  cat("  WARNING: some output files are missing — check warnings above.\n")
}
cat("\n")
cat("  To inspect results interactively:\n")
cat("    load('results/concrete/dgp_mcar/train_mcar/eval.rda')\n")
cat("    View(summary_df)\n")
cat("=======================================================\n")