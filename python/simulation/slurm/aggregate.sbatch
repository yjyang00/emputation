#!/bin/bash
#SBATCH --job-name emp_agg          # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition short           # Slurm partition to use
#SBATCH --ntasks 1                  # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 0-01:00              # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=4000          # Memory limit for each tasks (in MB)
#SBATCH -o out/emp_agg_%j.out       # File to which STDOUT will be written
#SBATCH -e out/emp_agg_%j.err       # File to which STDERR will be written
#SBATCH --mail-type=ALL             # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu    # Email to which notifications will be sent

# Prints the progress report (finished tasks, run times, fit sizes), collects
# all metric files into output/summary/*.csv, and draws the figures and tables
# into output/figures/ (Figures 2, S1, S2, S3; Tables S1, S2; comparison with the
# R results of the paper), plus the paper's figures redrawn from the R results
# (output/figures/from_R_reference/). Read the report in out/emp_agg_<jobid>.out.

module load Python
source ${VENV:-$HOME/yikun_venv}/bin/activate   # your environment; or VENV=/path/to/env
python ./run_simulation.py report
python ./run_simulation.py aggregate
python ./run_simulation.py figures
python ./run_simulation.py figures --source reference
