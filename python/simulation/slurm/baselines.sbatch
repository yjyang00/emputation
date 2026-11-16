#!/bin/bash
#SBATCH --job-name emp_base         # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition short           # Slurm partition to use
#SBATCH --array=1-300               # How many tasks in the array (see `python run_simulation.py tasks`)
#SBATCH --cpus-per-task=4           # CPU threads (random forests of missForest, BLAS)
#SBATCH --time 0-02:00              # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=1500          # Memory limit for each tasks (in MB)
#SBATCH -o out/emp_base_%A_%a.out   # File to which STDOUT will be written
#SBATCH -e out/emp_base_%A_%a.err   # File to which STDERR will be written
#SBATCH --mail-type=ALL             # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu    # Email to which notifications will be sent

# One array task = the benchmark methods (Mean, EM, MICE, missForest, GAIN;
# EM and MICE for ccprop) on one (setting, iteration), each evaluated.
#   EXPERIMENT=miss20 DATASET=concrete sbatch slurm/baselines.sbatch
# Optional: GAIN_ALPHA=10 (the value stated in the paper; default 100 as in
# config.R), MF_BACKEND=ranger (missForest >= 1.6 trees; default randomForest, missForest <= 1.5).

module load Python
source ${VENV:-$HOME/yikun_venv}/bin/activate   # your environment; or VENV=/path/to/env
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
i=${SLURM_ARRAY_TASK_ID}
python ./run_simulation.py baselines --experiment ${EXPERIMENT} --dataset ${DATASET} --task-id ${i} \
    ${GAIN_ALPHA:+--gain-alpha ${GAIN_ALPHA}} ${MF_BACKEND:+--missforest-backend ${MF_BACKEND}}
