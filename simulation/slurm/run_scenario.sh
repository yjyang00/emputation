#!/bin/bash
# slurm/run_scenario.sh
#
# Helper script: prints the sbatch commands to run for a given scenario.
# Copy and run each command in order, waiting for each stage to finish
# before submitting the next.
#
# Usage:
#   bash slurm/run_scenario.sh <scenario_id>
#
# Example:
#   bash slurm/run_scenario.sh 1

SCENARIO_ID=$1

if [ -z "$SCENARIO_ID" ]; then
  echo "Usage: bash slurm/run_scenario.sh <scenario_id>"
  echo "  scenario_id: integer from 1 to 18"
  exit 1
fi

if ! [[ "$SCENARIO_ID" =~ ^[0-9]+$ ]] || [ "$SCENARIO_ID" -lt 1 ] || [ "$SCENARIO_ID" -gt 18 ]; then
  echo "Error: scenario_id must be an integer between 1 and 18."
  exit 1
fi

# -----------------------------------------------------------------------
# Scenario lookup table (mirrors cfg$scenarios in scripts/config.R)
# -----------------------------------------------------------------------
case $SCENARIO_ID in
   1) DATASET=concrete; DGP=mcar; TRAIN=mcar ;;
   2) DATASET=concrete; DGP=ccmv; TRAIN=ccmv ;;
   3) DATASET=concrete; DGP=mcar; TRAIN=ccmv ;;
   4) DATASET=concrete; DGP=ccmv; TRAIN=mcar ;;
   5) DATASET=concrete; DGP=mar;  TRAIN=mcar ;;
   6) DATASET=concrete; DGP=mar;  TRAIN=ccmv ;;
   7) DATASET=ccpp;     DGP=mcar; TRAIN=mcar ;;
   8) DATASET=ccpp;     DGP=ccmv; TRAIN=ccmv ;;
   9) DATASET=ccpp;     DGP=mcar; TRAIN=ccmv ;;
  10) DATASET=ccpp;     DGP=ccmv; TRAIN=mcar ;;
  11) DATASET=ccpp;     DGP=mar;  TRAIN=mcar ;;
  12) DATASET=ccpp;     DGP=mar;  TRAIN=ccmv ;;
  13) DATASET=wine;     DGP=mcar; TRAIN=mcar ;;
  14) DATASET=wine;     DGP=ccmv; TRAIN=ccmv ;;
  15) DATASET=wine;     DGP=mcar; TRAIN=ccmv ;;
  16) DATASET=wine;     DGP=ccmv; TRAIN=mcar ;;
  17) DATASET=wine;     DGP=mar;  TRAIN=mcar ;;
  18) DATASET=wine;     DGP=mar;  TRAIN=ccmv ;;
esac

echo ""
echo "Scenario $SCENARIO_ID: dataset=$DATASET  dgp=$DGP  train=$TRAIN"
echo ""
echo "Submit the following commands in order."
echo "Wait for each stage to complete (squeue -u \$USER) before submitting the next."
echo ""
echo "-------------------------------------------------------"
echo "Stage 1 — Emputation (01_make_missing + 02_run_emputation):"
echo ""
echo "  SCENARIO_ID=$SCENARIO_ID sbatch slurm/emputation_array.txt"
echo ""
echo "-------------------------------------------------------"
echo "Stage 2 — GAIN and MissForest (after Stage 1 finishes):"
echo ""
echo "  SCENARIO_ID=$SCENARIO_ID sbatch slurm/missforest_array.txt"
echo "  DATASET=$DATASET DGP=$DGP TRAIN=$TRAIN sbatch slurm/gain.txt"
echo ""
echo "-------------------------------------------------------"
echo "Stage 3 — Evaluation (after Stage 2 finishes):"
echo ""
echo "  SCENARIO_ID=$SCENARIO_ID sbatch slurm/eval.txt"
echo ""
echo "-------------------------------------------------------"
echo "Stage 4 — Aggregation (after Stage 3 finishes):"
echo ""
echo "  SCENARIO_ID=$SCENARIO_ID sbatch slurm/aggregate_eval.txt"
echo ""
echo "-------------------------------------------------------"
echo "Output: results/$DATASET/dgp_$DGP/train_$TRAIN/eval.rda"
echo ""
