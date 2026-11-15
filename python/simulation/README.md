# Simulation code (Python)

Reproduces the simulation study of the paper: imputation under MCAR, CCMV and MAR missingness on three UCI
datasets (concrete, ccpp, wine), at 20% and 40% missingness and for varying complete-case proportions.
Python translation of the R simulation code in `../../simulation`.

## Experiments and outputs

| Experiment | Settings | Output (in `output/figures/`) | Paper |
|---|---|---|---|
| `miss20` | MCAR (rho = 0.25), CCMV (alpha_r = -4), MAR (rate 0.2); cc = 20% | `imputation_results_20_greyscale.pdf`, `mar_20_greyscale.pdf`, `pointwise_accuracy_20.{csv,tex}` | Figure 2, Figure S2a, Table S1 |
| `miss40` | MCAR (rho = 0.5), CCMV (alpha_r = -0.5), MAR (rate 0.4); cc = 20% | `imputation_results_40_greyscale.pdf`, `mar_40_greyscale.pdf`, `pointwise_accuracy_40.{csv,tex}` | Figure S1, Figure S2b, Table S2 |
| `ccprop` | MAR (rate 0.4), cc = 5, 10, 20, 40, 60% | `ccprop_energy.pdf` | Figure S3 |

Each experiment has 100 iterations per (dataset, setting). In each iteration, the missing data are generated
once and every method imputes the same incomplete dataset: emputation trained under MCAR and CCMV
(`Emp_mcar`, `Emp_ccmv`; 500 epochs, lr 1e-4, B = 2, two hidden layers of width 500, batch norm, full
batch) and the benchmarks EM, MICE, missForest (MF), GAIN and mean imputation (EM and MICE only for
`ccprop`). Metrics (Supplement A): RMSE and MAE of the averaged imputation, MADC, energy distance
(`energy::eqdist.e` scale) and MMD^2 (`kernlab::kmmd`), with 10 imputations for the multiple-imputation methods.

The aggregate stage also writes `comparison_with_R.{csv,pdf}` (Python results against the R results of the
paper, from `reference_results/r_summary.csv`) and `variant_check.csv` (see *Benchmark settings*).

## Layout

- The three datasets are read from the R simulation's data folder, `../../simulation/data` (`Folds_pp.xlsx`
  from its first sheet, as in the R code); set `EMPUTATION_DATA_DIR` to use another folder.
- `simstudy/` — the simulation library:
  `config.py` (experiments, hyperparameters, seeds), `datasets.py`, `dgp.py` (MCAR/MAR/CCMV generators and
  the CCMV patterns of `../../simulation/src/R/specs`), `baselines/` (EM, MICE-PMM, missForest, GAIN, mean), `metrics.py`,
  `tasks.py` (pipeline stages), `figures.py`.
- `run_simulation.py` — command-line driver of all stages.
- `slurm/` — SLURM jobs (`submit_all.sh` submits everything; `emputation_gpu.sbatch` for the GPU partition).
- `reference_results/r_summary.csv` — the R results shipped in `../../simulation/results` (converted from `.rda`).
- `run_example.py` — small local run of the pipeline on the concrete data.

## Running on the cluster

1. Clone the repository on the cluster. The job scripts activate the Python environment `~/yikun_venv` (set
   `VENV=/path/to/env` to use another one), which needs torch, numpy, scipy, pandas, scikit-learn and
   matplotlib; the scripts use the package in `../emputation-python/` without installation.

2. Submit from `python/simulation`:

   ```bash
   bash slurm/submit_all.sh                    # all experiments
   bash slurm/submit_all.sh miss20             # only Figure 2 / S2a / Table S1
   DRY_RUN=1 bash slurm/submit_all.sh          # print the sbatch commands
   GPU=wine bash slurm/submit_all.sh           # the large wine fits on the GPU partition
   ```

   The script chains `data.sbatch` -> `emputation.sbatch` / `baselines.sbatch` arrays (one array per
   experiment and dataset; logs in `out/`) -> `aggregate.sbatch`, which writes the summaries to
   `output/summary/` and the figures and tables to `output/figures/`. Partition names default to `short` and
   `medium` (`SHORT=... MEDIUM=...` to change).

3. Unfinished tasks (time-outs, preemption) can be listed and resubmitted:

   ```bash
   python run_simulation.py missing --experiment miss20 --dataset wine --stage-name emputation
   ```

   Every task skips work whose output already exists, so rerunning `aggregate.sbatch` afterwards is enough.

### Resources

The cost is dominated by the emputation fits, which use full-batch training as in the paper. One epoch
processes every training term twice (B = 2), and the number of terms depends on the data:

| | terms per fit | memory | requested |
|---|---|---|---|
| concrete | 1k-110k | < 3 GB | 8 CPUs, 8 GB, 4 h |
| ccpp | 10k-170k | < 5 GB | 8 CPUs, 12 GB, 6 h |
| wine, CCMV process / cc 60% | 10k-50k | < 2 GB | 8 CPUs, 8 GB |
| wine, MAR (20%) / cc 5-10% | 0.5-1.0M | 13-25 GB | 16 CPUs, 24-40 GB |
| wine, MCAR (20%), MAR (40%), cc 20-40% | 1.6-1.8M | 40-45 GB | 16 CPUs, 56-64 GB |
| wine, MCAR (40%) | 2.8M | about 66 GB | 16 CPUs, 88 GB |

Memory is about 12 KB per network row (= 2 x terms). `submit_all.sh` requests these resources per
experiment, dataset and setting. If large-memory nodes are scarce, `WINE_CHUNK_SIZE=300000 bash
slurm/submit_all.sh` trains the large wine fits with gradient accumulation over chunks of 300k terms (16 GB);
with batch normalization the batch statistics then come from 600k-row chunks instead of the full batch, a
close approximation. GPUs make the fits much faster: `GPU=wine bash slurm/submit_all.sh` runs the large wine
fits on the GPU partition (`slurm/emputation_gpu.sbatch`: one GPU per fit, `module load CUDA`), and `GPU=all`
every fit; on GPUs, fits with more than `GPU_CHUNK_SIZE` (default 250k) training terms accumulate gradients over
chunks of that size.

## Benchmark settings

- **MICE** re-implements the defaults of R's `mice()` for numeric data (predictive mean matching with 5
  donors and a Bayesian regression draw, 5 iterations, chains started from random draws).
- **missForest** uses scikit-learn random forests with the settings of R's `missForest()` (100 trees,
  `mtry = floor(sqrt(p))`, at most 10 iterations, same stopping rule). The default `randomForest` tree rule
  (missForest <= 1.5) matched the published MF results; `--missforest-backend ranger` uses the rule of
  missForest >= 1.6.
- **GAIN** is a PyTorch port of the reference implementation (github.com/jsyoon0823/GAIN) with batch size 128,
  hint rate 0.9, 1000 iterations and alpha = 100 as in `scripts/config.R` (the paper's text says alpha = 10;
  `GAIN_ALPHA=10` changes it).
- The `miss20` baseline jobs also run the alternatives `MF_ranger` and `GAIN_a10`; `variant_check.csv`
  reports which variant is closer to the published R results.

## Local use

```bash
python run_simulation.py tasks                     # array sizes
python run_example.py --iters 2 --epochs 20        # small local run on concrete (not the paper's settings)
python run_simulation.py figures --source reference  # the paper's figures from the R results
```
