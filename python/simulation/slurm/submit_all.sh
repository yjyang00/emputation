#!/bin/bash
# Submit the whole simulation study with job dependencies:
#   data -> {emputation arrays, baseline arrays} -> aggregate + figures
#
# Usage (from the simulation folder):
#   bash slurm/submit_all.sh                  # miss20 miss40 ccprop (all simulation figures/tables)
#   bash slurm/submit_all.sh miss20           # Figure 2, Figure S2a, Table S1 only
#   DRY_RUN=1 bash slurm/submit_all.sh        # print the sbatch commands without submitting
#
# Environment overrides:
#   SHORT=short MEDIUM=medium   partition names (short: <= 12 h, medium: multi-day)
#   VENV=/path/to/env           Python environment (default: ~/yikun_venv)
#   WINE_CHUNK_SIZE=300000      train the large wine fits with gradient accumulation over
#                               chunks of 300k terms (~16 GB instead of up to ~90 GB; batch-norm
#                               statistics then come from chunks of 600k rows, see README)
#   GPU=all                     run the emputation fits on the GPU partition (slurm/emputation_gpu.sbatch,
#                               one GPU per fit); GPU=wine: only the large wine fits, the rest on CPUs
#   GPU_PARTITION=gpu           name of the GPU partition
#   GPU_CHUNK_SIZE=250000       on GPUs, fits with more training terms accumulate gradients over chunks of
#                               this many terms (~6 GB of GPU memory); GPU_CHUNK_SIZE= (empty) trains every
#                               fit on the full batch (the largest wine fit then needs ~70 GB of GPU memory)
#   GAIN_ALPHA=10               GAIN alpha stated in the paper (default 100, as in config.R)
#   MF_BACKEND=ranger           missForest >= 1.6 trees (default: randomForest, missForest <= 1.5)
#
# Resubmitting unfinished tasks (time-outs, preemption):
#   ids=$(python run_simulation.py missing --experiment miss20 --dataset wine --stage-name emputation)
#   EXPERIMENT=miss20 DATASET=wine sbatch --array=$ids --partition medium --cpus-per-task 16 \
#       --mem-per-cpu 3500 --time 1-12:00 slurm/emputation.sbatch
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p out
EXPS=${*:-miss20 miss40 ccprop}
SHORT=${SHORT:-short}
MEDIUM=${MEDIUM:-medium}

submit() {  # submit "$@" and print the job id
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "  [dry run] ${SUBMIT_ENV:-} sbatch $*" >&2
        echo "DRY${RANDOM}"
    else
        env ${SUBMIT_ENV:-} sbatch --parsable "$@"
    fi
}

export EXPERIMENTS="${EXPS}"   # read by slurm/data.sbatch (sbatch exports the environment)
data_job=$(SUBMIT_ENV="" submit slurm/data.sbatch)
echo "data: ${data_job}"

# Emputation arrays: experiment dataset array-range cpus mem-per-cpu(MB) time partition.
# Wine is split by setting because the number of training terms (and so memory)
# varies by two orders of magnitude; ranges follow the setting order of
# simstudy/config.py (200 tasks = 100 iterations x 2 training mechanisms each).
EMP_JOBS=(
    "miss20 concrete 1-600    8 1000 0-04:00 ${SHORT}"
    "miss20 ccpp     1-600    8 1500 0-06:00 ${SHORT}"
    "miss20 wine     1-200   16 3500 1-12:00 ${MEDIUM}"   # dgp mcar: ~1.7M terms, ~41 GB
    "miss20 wine     201-400  8 1000 0-06:00 ${SHORT}"    # dgp ccmv: ~46k terms
    "miss20 wine     401-600 16 2500 1-12:00 ${MEDIUM}"   # dgp mar:  ~1.0M terms, ~25 GB
    "miss40 concrete 1-600    8 1000 0-04:00 ${SHORT}"
    "miss40 ccpp     1-600    8 1500 0-06:00 ${SHORT}"
    "miss40 wine     1-200   16 5500 2-00:00 ${MEDIUM}"   # dgp mcar: ~2.8M terms, ~66 GB
    "miss40 wine     201-400  8 1000 0-04:00 ${SHORT}"    # dgp ccmv: ~10k terms
    "miss40 wine     401-600 16 3500 1-12:00 ${MEDIUM}"   # dgp mar:  ~1.7M terms, ~41 GB
    "ccprop concrete 1-1000   8 1000 0-04:00 ${SHORT}"
    "ccprop ccpp     1-1000   8 1500 0-06:00 ${SHORT}"
    "ccprop wine     1-200   16 1500 1-00:00 ${MEDIUM}"   # cc  5%: ~0.5M terms
    "ccprop wine     201-400 16 2000 1-12:00 ${MEDIUM}"   # cc 10%: ~0.85M terms
    "ccprop wine     401-600 16 3500 1-12:00 ${MEDIUM}"   # cc 20%: ~1.65M terms
    "ccprop wine     601-800 16 4000 1-12:00 ${MEDIUM}"   # cc 40%: ~1.8M terms
    "ccprop wine     801-1000 8 1000 0-04:00 ${SHORT}"    # cc 60%: ~35k terms
)
BASE_JOBS=(
    "miss20 concrete 1-300 4 1000 0-01:00"
    "miss20 ccpp     1-300 4 1500 0-02:00"
    "miss20 wine     1-300 4 1500 0-02:00"
    "miss40 concrete 1-300 4 1000 0-01:00"
    "miss40 ccpp     1-300 4 1500 0-02:00"
    "miss40 wine     1-300 4 1500 0-02:00"
    "ccprop concrete 1-500 4 1000 0-01:00"
    "ccprop ccpp     1-500 4 1500 0-02:00"
    "ccprop wine     1-500 4 1500 0-02:00"
)

wanted() { [[ " ${EXPS} " == *" $1 "* ]]; }
deps=()
for spec in "${EMP_JOBS[@]}"; do
    read -r exp ds range cpus mem time part <<< "${spec}"
    wanted "${exp}" || continue
    large=$([[ "${ds}" == "wine" && "${cpus}" == "16" ]] && echo 1 || echo 0)
    envs="EXPERIMENT=${exp} DATASET=${ds}"
    if [[ "${GPU:-}" == "all" || ( "${GPU:-}" == "wine" && "${large}" == 1 ) ]]; then
        script=slurm/emputation_gpu.sbatch label=" on GPU"
        envs="${envs} CHUNK_SIZE=${GPU_CHUNK_SIZE-250000}"
        res=(--partition="${GPU_PARTITION:-gpu}" --ntasks=1 --gpus-per-task=1 --cpus-per-task=4
             --mem-per-cpu=4000 --time="$([[ "${large}" == 1 ]] && echo 0-04:00 || echo 0-01:00)")
    else
        script=slurm/emputation.sbatch label=""
        if [[ "${large}" == 1 && -n "${WINE_CHUNK_SIZE:-}" ]]; then
            envs="${envs} CHUNK_SIZE=${WINE_CHUNK_SIZE}"
            mem=1000
        fi
        res=(--cpus-per-task="${cpus}" --mem-per-cpu="${mem}" --time="${time}" --partition="${part}")
    fi
    id=$(SUBMIT_ENV="${envs}" submit --dependency=afterok:${data_job} --job-name "emp_${exp}_${ds}" \
         --array="${range}" "${res[@]}" "${script}")
    echo "emputation ${exp}/${ds} [${range}]${label}: ${id}"
    deps+=("${id}")
done
for spec in "${BASE_JOBS[@]}"; do
    read -r exp ds range cpus mem time <<< "${spec}"
    wanted "${exp}" || continue
    id=$(SUBMIT_ENV="EXPERIMENT=${exp} DATASET=${ds}" submit --dependency=afterok:${data_job} \
         --job-name "base_${exp}_${ds}" --array="${range}" --cpus-per-task="${cpus}" \
         --mem-per-cpu="${mem}" --time="${time}" --partition="${SHORT}" slurm/baselines.sbatch)
    echo "baselines ${exp}/${ds} [${range}]: ${id}"
    deps+=("${id}")
done
dep_list=$(IFS=:; echo "${deps[*]}")
agg=$(SUBMIT_ENV="" submit --dependency=afterany:${dep_list} slurm/aggregate.sbatch)
echo "aggregate + figures: ${agg}"
echo "Figures and tables will be written to output/figures/."
