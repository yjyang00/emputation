#!/bin/bash
#SBATCH --job-name emp_feed         # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition medium          # Slurm partition to use (multi-day; the job is idle most of the time)
#SBATCH --ntasks 1                  # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 2-00:00              # Wall time limit in D-HH:MM (before it, the feeder hands over to a new job)
#SBATCH --mem-per-cpu=4000          # Memory limit for each tasks (in MB)
#SBATCH -o out/emp_feed_%j.out      # File to which STDOUT will be written
#SBATCH -e out/emp_feed_%j.err      # File to which STDERR will be written
#SBATCH --mail-type=END,FAIL        # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu    # Email to which notifications will be sent

# Submits the job arrays of a plan, written by slurm/submit_all.sh, as room in the
# queue frees up (see slurm/feeder.py). Started by slurm/submit_all.sh; to continue
# a plan by hand, from the simulation folder:
#   PLAN=out/plan_<date>.tsv sbatch slurm/feeder.sbatch

module load Python
source ${VENV:-$HOME/yikun_venv}/bin/activate   # your environment; or VENV=/path/to/env
python ./slurm/feeder.py "${PLAN}"
