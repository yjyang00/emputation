# Data Application — README

This folder contains code to reproduce the data application results in the 
main paper and supplementary material.

---

## Files

- `bootstrap.R`          — Main script for bootstrap
- `sensitivity.R`        — Main script for sensitivity analysis
- `Figure2.R`            — Reproduce Figure 2 in the main paper
- `sensitivity_plot.R`   — Reproduce Figure S4 in the supplementary material

---

## Data Access

Researchers seeking the full dataset should submit a data request at https://naccdata.org in accordance with NACC's data use policies.

---

## Prerequisites

```r
install.packages(c("tidyverse", "mice", "ggplot2", "patchwork"))
install.packages("torch")
torch::install_torch()
```

---

## Execution

Both `bootstrap.R` and `sensitivity.R` can be easily run as SLURM array jobs on a
cluster, with each job handling one bootstrap replicate identified by a seed.

### Bootstrap replicates

`bootstrap.R` takes two arguments: `<seed>` and `<output_dir>`.

```bash
# Example single run (set mechanism = "ccmv" or "mcar" in bootstrap.R before submitting)
Rscript bootstrap.R 1 results/bootstrap
```
To reproduce the full results, run 500 replicates, either sequentially in a loop or in parallel via a SLURM array job.

### Sensitivity analysis replicates

`sensitivity.R` takes two arguments: <seed> and <output_dir>.
Run 500 replicates under CCMV emputation.

```bash
# Example single run
Rscript sensitivity.R 1 results/sensitivity
```

To reproduce the full results, run 500 replicates sequentially or in parallel via a SLURM array job.

### Figures

Run Figure2.R to reproduce Figure 2 in the main paper for bootstrap.
Run sensitivity_plot.R to reproduce Figure S4 in the supplementary material for 
sensitivity analysis.
