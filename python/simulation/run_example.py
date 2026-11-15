#!/usr/bin/env python3
"""Small local run of the simulation pipeline on the concrete data (Python version of run_example.R).

Runs the `miss20` experiment for the concrete dataset end to end: missing data,
emputation trained under MCAR and CCMV, the benchmarks (Mean, EM, MICE,
missForest, GAIN), aggregation and figures. With the paper's settings (500
epochs) one emputation fit takes tens of minutes on a laptop, so the default
trains for fewer epochs; the full study runs on a cluster (slurm/submit_all.sh).

Usage:
    python run_example.py [--iters 2] [--epochs 20] [--output-dir output_example]
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the emputation package is used from this folder, without installation
sys.path.insert(0, str(HERE.parent / "emputation-python"))

from simstudy import figures, tasks  # noqa: E402
from simstudy.config import EXPERIMENTS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, default=2, help="iterations per setting (the paper uses 100)")
    parser.add_argument("--epochs", type=int, default=20, help="emputation epochs (the paper uses 500)")
    parser.add_argument("--output-dir", default=str(HERE / "output_example"))
    args = parser.parse_args()

    exp = EXPERIMENTS["miss20"]
    layout = tasks.Layout(args.output_dir)
    tasks.run_data(layout, exp.name, ["concrete"])
    for s in range(len(exp.settings)):
        for it in range(1, args.iters + 1):
            for t in range(len(exp.trains)):
                task_id = (s * exp.n_iter + it - 1) * len(exp.trains) + t + 1
                tasks.run_emputation(layout, exp.name, "concrete", task_id,
                                     emputation_kwargs=dict(num_epochs=args.epochs, print_every_nepoch=10 ** 6))
            tasks.run_baselines(layout, exp.name, "concrete", s * exp.n_iter + it,
                                methods=["Mean", "EM", "MICE", "MF", "GAIN"])
    tasks.aggregate(layout, [exp.name])
    figures.make_all(layout, experiments=[exp.name])
    print(f"\nSummary: {layout.summary_dir / 'summary_miss20.csv'}\nFigures: {layout.figures_dir}")


if __name__ == "__main__":
    main()
