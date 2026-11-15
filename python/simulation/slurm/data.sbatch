#!/bin/bash
#SBATCH --job-name emp_data         # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition short           # Slurm partition to use
#SBATCH --ntasks 1                  # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 0-01:00              # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=4000          # Memory limit for each tasks (in MB)
#SBATCH -o out/emp_data_%j.out      # File to which STDOUT will be written
#SBATCH -e out/emp_data_%j.err      # File to which STDERR will be written
#SBATCH --mail-type=ALL             # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu    # Email to which notifications will be sent

# Generates the missingness masks of all iterations (seconds to minutes).
# Submit from the simulation folder:  sbatch slurm/data.sbatch
# Optional: EXPERIMENTS="miss20 miss40 ccprop" sbatch slurm/data.sbatch

module load Python
source ${VENV:-$HOME/yikun_venv}/bin/activate   # your environment; or VENV=/path/to/env
for e in ${EXPERIMENTS:-miss20 miss40 ccprop}; do
    python ./run_simulation.py data --experiment ${e}
done
