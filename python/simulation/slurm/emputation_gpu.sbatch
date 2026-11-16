#!/bin/bash
#SBATCH --job-name emp_gpu          # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition gpu             # Slurm partition to use (GPU nodes)
#SBATCH --array=1-600               # How many tasks in the array (see `python run_simulation.py tasks`)
#SBATCH --ntasks=1                  # One task per array element
#SBATCH --gpus-per-task=1           # One GPU per emputation fit
#SBATCH --cpus-per-task=4           # CPU threads for data handling and the evaluation metrics
#SBATCH --time 0-04:00              # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=4000          # Memory limit for each tasks (in MB)
#SBATCH -o out/emp_gpu_%A_%a.out    # File to which STDOUT will be written
#SBATCH -e out/emp_gpu_%A_%a.err    # File to which STDERR will be written
#SBATCH --mail-type=ALL             # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu    # Email to which notifications will be sent

# GPU version of emputation.sbatch: one array task = one emputation fit on one
# GPU, followed by its evaluation on the CPUs. Submitted by slurm/submit_all.sh
# with GPU=all or GPU=wine; to submit by hand:
#   EXPERIMENT=miss40 DATASET=wine CHUNK_SIZE=250000 sbatch slurm/emputation_gpu.sbatch
# CHUNK_SIZE: fits with more training terms accumulate gradients over chunks of
# this many terms (250k terms take about 6 GB of GPU memory); unset = exact full batch.

module load CUDA
module load Python
source ${VENV:-$HOME/yikun_venv}/bin/activate   # your environment; or VENV=/path/to/env
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
i=${SLURM_ARRAY_TASK_ID}
python ./run_simulation.py emputation --experiment ${EXPERIMENT} --dataset ${DATASET} --task-id ${i} \
    --device cuda ${CHUNK_SIZE:+--chunk-size ${CHUNK_SIZE}}
