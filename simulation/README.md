# Simulation Code — README

This folder contains code to reproduce the simulation results in the paper.
The simulation evaluates imputation methods under three missing data mechanisms
(MCAR, MAR, CCMV) across three datasets (concrete, ccpp, wine), at two missingness levels (20% and 40%).

---

## Directory Structure

- `data/` — raw datasets: `concrete_data.csv`, `winequality-white.csv`, `Folds_pp.xlsx`
- `src/R/emputation/` — Emputation source code
- `src/R/specs/` — CCMV pattern specifications
- `scripts/` — pipeline scripts plus `config.R` and `setup.R`
- `slurm/` — SLURM batch scripts for cluster execution
- `results/` — simulation results for 20% and 40% missingness
- `run_example.R` — local tutorial script (see details below)

---

## Provided Results

The `results/` folder includes raw simulation results reported in the paper
These can be used directly with the visualization code for reproducing the figures.


## Scenarios

The pipeline covers 18 scenarios: 3 datasets * 3 DGPs * 2 training mechanisms.
Scenarios are defined in `scripts/config.R` as `cfg$scenarios` and indexed 1-18.

| ID | Dataset  | DGP  | Train | ID | Dataset | DGP  | Train |
|----|----------|------|-------|----|---------|------|-------|
| 1  | concrete | mcar | mcar  | 10 | ccpp    | ccmv | mcar  |
| 2  | concrete | ccmv | ccmv  | 11 | ccpp    | mar  | mcar  |
| 3  | concrete | mcar | ccmv  | 12 | ccpp    | mar  | ccmv  |
| 4  | concrete | ccmv | mcar  | 13 | wine    | mcar | mcar  |
| 5  | concrete | mar  | mcar  | 14 | wine    | ccmv | ccmv  |
| 6  | concrete | mar  | ccmv  | 15 | wine    | mcar | ccmv  |
| 7  | ccpp     | mcar | mcar  | 16 | wine    | ccmv | mcar  |
| 8  | ccpp     | ccmv | ccmv  | 17 | wine    | mar  | mcar  |
| 9  | ccpp     | mcar | ccmv  | 18 | wine    | mar  | ccmv  |

The `train` column indicates which missing data mechanism Emputation assumes
during model fitting. Benchmark methods (MICE, MissForest, EM, GAIN, Mean)
do not use mechanism information and are trained and evaluated once per
(dataset, DGP) combination regardless of the `train` column.
GAIN results are reported in the paper but the source code is not redistributed here; see the original repository
at https://github.com/jsyoon0823/GAIN for implementation details.

---

## Configuration

All parameters are set in `scripts/config.R`. This folder corresponds to the
20% missingness setting. To reproduce the 40% missingness setting,
change DGP parameters following the `additional simulation details` in the supplementary material.

## Prerequisites

### R Packages

```r
install.packages(c(
  "tidyverse", "mice", "missForest", "energy",
  "kernlab", "readxl", "MASS"
))

install.packages("torch")
torch::install_torch()
```
---

## Local Tutorial

`run_example.R` runs the complete pipeline sequentially on concrete dataset. 
This is the recommended starting point to see how the pipeline runs.

**Methods evaluated:** Emputation, MICE, EM, Mean, MissForest.
GAIN is excluded from the local tutorial for simplicity (no Python environment required).

Please set the working directory to the `simulation` folder:
   ```r
   setwd("path/to/simulation")
   ```
When running `run_example.R`, intermediate files (per-iteration imputation outputs) are written to `run/concrete/dgp_*/train_*/` and evaluation results are written to `results/concrete/dgp_*/train_*/`.

---

## Cluster reproduction

The full pipeline can be easily run on a SLURM cluster. The five scripts in
`slurm/` need to be submitted in the order below.

### Get Commands for a Scenario
 
Run the helper script to print the exact submission commands:
 
```bash
bash slurm/run_scenario.sh 1
```

This prints the four `sbatch` commands for scenario 1 (concrete / dgp=mcar / train=mcar)
with the correct variable values already filled in. 

---

## Output Structure
 
The pipeline writes to two folders:
 
- `run/` — intermediate files generated during the pipeline (per-iteration imputation
  outputs). Structure: `run/<dataset>/dgp_<dgp>/train_<train>/iter_###.rda` (missing data),
  `iter_###_emputation.rda` (Emputation), `iter_###_missForest.rda` (MissForest),
  `iter_###_gain.csv` (GAIN).
- `results/` — final evaluation outputs at
  `results/<dataset>/dgp_<dgp>/train_<train>/eval.rda`.


