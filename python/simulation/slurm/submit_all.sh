#!/bin/bash
# Submit the whole simulation study.
#
# Clusters cap the number of jobs a user may have queued (MaxSubmitJobs; every array task counts),
# and the study has ~10,000 tasks. This script therefore writes the job arrays to submit, with their
# resources, into a plan (out/plan_<date>.tsv) and submits one small long-running job,
# slurm/feeder.sbatch, that submits them as the queue frees up (slurm/feeder.py): every 15 minutes it
# submits, in plan order and in arrays of at most 100 tasks, the tasks that are neither finished nor
# queued, until sbatch refuses because of the limit. After the last array it submits
# slurm/aggregate.sbatch (summaries, figures and tables), which waits for all arrays.
#
# Usage (from the simulation folder):
#   bash slurm/submit_all.sh                  # miss20 miss40 ccprop (all simulation figures/tables)
#   bash slurm/submit_all.sh miss20           # Figure 2, Figure S2a, Table S1 only
#   DRY_RUN=1 bash slurm/submit_all.sh        # print the plan without submitting anything
#   cat out/emp_feed_<jobid>.out              # what the feeder has submitted so far
#
# Rerunning is safe: finished and queued tasks are skipped. Once the arrays have run, rerunning the
# script resubmits exactly the tasks that failed (time-outs, node failures).
#
# Environment overrides:
#   SHORT=short MEDIUM=medium   partition names (short, <= 12 h: all jobs; medium, multi-day: the feeder)
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
#   FEED_EVERY=15 FEED_CHUNK=100  minutes between the feeder's rounds; tasks per submitted array
#   FEED_RESERVE=20             places in the queue the feeder keeps free for your other jobs
#   FEED_MAX_SUBMIT=<n>         your submit limit (sacctmgr show assoc user=$USER format=MaxSubmit);
#                               without it, the feeder learns the limit from the first refusal
#   FEED_TIME=2-00:00           time limit of a feeder job (it hands over to a new one before the limit)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p out
SIM=$(pwd)
EXPS=${*:-miss20 miss40 ccprop}
SHORT=${SHORT:-short}
MEDIUM=${MEDIUM:-medium}
PLAN=${SIM}/out/plan_$(date +%Y%m%d_%H%M%S).tsv

# Emputation arrays: experiment dataset array-range cpus mem-per-cpu(MB) time.
# Every fit takes well under 12 h, so all run on the short partition with a 10 h limit. Run times
# below are from the pilot: a fit takes ~50 s + 3.7 ms per training term on 8 CPUs, and 2.4 ms per
# term for the large wine fits on 16 CPUs, plus 3-5 min per task for start-up and evaluation.
# Wine is split by setting because the number of training terms (and so memory and time)
# varies by two orders of magnitude; ranges follow the setting order of simstudy/config.py
# (200 tasks = 100 iterations x 2 training mechanisms each).
EMP_JOBS=(
    "miss20 concrete 1-600    8 1000 0-10:00"   # <= 85k terms, <= 10 min
    "miss20 ccpp     1-600    8 1500 0-10:00"   # <= 170k terms, <= 15 min
    "miss20 wine     1-200   16 3500 0-10:00"   # dgp mcar: 1.3-1.7M terms, ~41 GB, 55-75 min
    "miss20 wine     201-400  8 1000 0-10:00"   # dgp ccmv: 46k terms, ~10 min
    "miss20 wine     401-600 16 2500 0-10:00"   # dgp mar:  0.9-1.0M terms, ~25 GB, 40-45 min
    "miss40 concrete 1-600    8 1000 0-10:00"   # <= 110k terms, <= 10 min
    "miss40 ccpp     1-600    8 1500 0-10:00"   # <= 100k terms, <= 10 min
    "miss40 wine     1-200   16 5500 0-10:00"   # dgp mcar: 2.5-2.8M terms, ~67 GB, 105-120 min
    "miss40 wine     201-400  8 1000 0-10:00"   # dgp ccmv: 10k terms, ~6 min
    "miss40 wine     401-600 16 3500 0-10:00"   # dgp mar:  1.7-1.8M terms, ~42 GB, 70-75 min
    "ccprop concrete 1-1000   8 1000 0-10:00"   # <= 95k terms, <= 10 min
    "ccprop ccpp     1-1000   8 1500 0-10:00"   # <= 125k terms, <= 11 min
    "ccprop wine     1-200   16 1500 0-10:00"   # cc  5%: 0.4-0.5M terms, ~25 min
    "ccprop wine     201-400 16 2000 0-10:00"   # cc 10%: 0.8-0.9M terms, ~40 min
    "ccprop wine     401-600 16 3500 0-10:00"   # cc 20%: 1.6-1.7M terms, ~70 min
    "ccprop wine     601-800 16 4000 0-10:00"   # cc 40%: 1.8-1.85M terms, ~80 min
    "ccprop wine     801-1000 8 1000 0-10:00"   # cc 60%: 35k terms, ~8 min
)
BASE_JOBS=(   # benchmarks: < 1 GB and a few minutes per task in the pilot
    "miss20 concrete 1-300 4 1000 0-10:00"
    "miss20 ccpp     1-300 4 1500 0-10:00"
    "miss20 wine     1-300 4 1500 0-10:00"
    "miss40 concrete 1-300 4 1000 0-10:00"
    "miss40 ccpp     1-300 4 1500 0-10:00"
    "miss40 wine     1-300 4 1500 0-10:00"
    "ccprop concrete 1-500 4 1000 0-10:00"
    "ccprop ccpp     1-500 4 1500 0-10:00"
    "ccprop wine     1-500 4 1500 0-10:00"
)

unit() {  # name kind range dir script env after sbatch-options... -> one line of the plan
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" "${*:8}"
}
for exp in ${EXPS}; do
    case ${exp} in miss20|miss40|ccprop) ;; *) echo "unknown experiment ${exp}" >&2; exit 1 ;; esac
done
{
for exp in ${EXPS}; do
    for spec in "${EMP_JOBS[@]}"; do
        read -r e ds range cpus mem time <<< "${spec}"
        [[ "${e}" == "${exp}" ]] || continue
        large=$([[ "${ds}" == "wine" && "${cpus}" == "16" ]] && echo 1 || echo 0)
        if [[ "${GPU:-}" == "all" || ( "${GPU:-}" == "wine" && "${large}" == 1 ) ]]; then
            script=slurm/emputation_gpu.sbatch chunk=${GPU_CHUNK_SIZE-250000}
            res=(--partition="${GPU_PARTITION:-gpu}" --ntasks=1 --gpus-per-task=1 --cpus-per-task=4
                 --mem-per-cpu=4000 --time="$([[ "${large}" == 1 ]] && echo 0-04:00 || echo 0-01:00)")
        else
            script=slurm/emputation.sbatch chunk=""
            if [[ "${large}" == 1 && -n "${WINE_CHUNK_SIZE:-}" ]]; then
                chunk=${WINE_CHUNK_SIZE} mem=1000
            fi
            res=(--partition="${SHORT}" --cpus-per-task="${cpus}" --mem-per-cpu="${mem}" --time="${time}")
        fi
        unit "emp_${exp}_${ds}" emputation "${range}" "${SIM}" "${script}" \
            "EXPERIMENT=${exp};DATASET=${ds};CHUNK_SIZE=${chunk}" - "${res[@]}"
    done
    for spec in "${BASE_JOBS[@]}"; do
        read -r e ds range cpus mem time <<< "${spec}"
        [[ "${e}" == "${exp}" ]] || continue
        unit "base_${exp}_${ds}" baselines "${range}" "${SIM}" slurm/baselines.sbatch \
            "EXPERIMENT=${exp};DATASET=${ds};GAIN_ALPHA=${GAIN_ALPHA:-};MF_BACKEND=${MF_BACKEND:-}" - \
            --partition="${SHORT}" --cpus-per-task="${cpus}" --mem-per-cpu="${mem}" --time="${time}"
    done
done
unit emp_agg job - "${SIM}" slurm/aggregate.sbatch - "afterany:emp_*,base_*" --partition="${SHORT}"
} > "${PLAN}"

if [[ -n "${DRY_RUN:-}" ]]; then   # the plan without the folder column
    awk -F'\t' -v OFS='\t' '{print $1, $2, $3, $5, $6, $7, ($8 == "" ? "-" : $8)}' "${PLAN}" |
        (column -t -s $'\t' 2>/dev/null || cat)
    rm "${PLAN}"
    exit 0
fi
export FEED_SBATCH="--partition=${MEDIUM} --time=${FEED_TIME:-2-00:00}"   # also used for later feeder jobs
feed=$(PLAN="${PLAN}" sbatch --parsable ${FEED_SBATCH} slurm/feeder.sbatch)
echo "plan: ${PLAN}"
echo "feeder job ${feed} submits it as the queue frees up; progress: cat out/emp_feed_${feed}.out"
