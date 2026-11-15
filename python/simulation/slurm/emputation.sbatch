#!/bin/bash
#SBATCH --job-name emp_fit          # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition short           # Slurm partition to use
#SBATCH --array=1-600               # How many tasks in the array (see `python run_simulation.py tasks`)
#SBATCH --cpus-per-task=8           # CPU threads used by PyTorch for one emputation fit
#SBATCH --time 0-06:00              # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=1500          # Memory limit for each tasks (in MB)
#SBATCH -o out/emp_fit_%A_%a.out    # File to which STDOUT will be written
#SBATCH -e out/emp_fit_%A_%a.err    # File to which STDERR will be written
#SBATCH --mail-type=ALL             # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu    # Email to which notifications will be sent

# One array task = one emputation fit (setting, iteration, training mechanism)
# followed by its evaluation. Submitted by slurm/submit_all.sh, which sets the
# array range, CPUs, memory and time per dataset; to submit by hand:
#   EXPERIMENT=miss20 DATASET=concrete sbatch slurm/emputation.sbatch
# Optional: CHUNK_SIZE=300000 (gradient accumulation over chunks of terms,
# caps memory). For the GPU partition use slurm/emputation_gpu.sbatch.

module load Python
source ${VENV:-$HOME/yikun_venv}/bin/activate   # your environment; or VENV=/path/to/env
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
i=${SLURM_ARRAY_TASK_ID}
python ./run_simulation.py emputation --experiment ${EXPERIMENT} --dataset ${DATASET} --task-id ${i} \
    ${CHUNK_SIZE:+--chunk-size ${CHUNK_SIZE}}
