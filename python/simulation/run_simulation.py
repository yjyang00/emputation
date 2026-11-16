#!/usr/bin/env python3
"""Command-line driver of the simulation study.

Stages (run from the `simulation` folder):

    python run_simulation.py tasks       --experiment miss20
    python run_simulation.py data        --experiment miss20
    python run_simulation.py emputation  --experiment miss20 --dataset wine --task-id 17
    python run_simulation.py baselines   --experiment miss20 --dataset wine --task-id 17
    python run_simulation.py report      # finished tasks, run times, fit sizes
    python run_simulation.py aggregate
    python run_simulation.py figures

Experiments: miss20 (Figure 2, Figure S2a, Table S1), miss40 (Figure S1,
Figure S2b, Table S2) and ccprop (Figure S3). `--task-id` is the 1-based
SLURM array index; `tasks` prints the array ranges.
"""

import argparse
import os
import sys
from pathlib import Path

# the emputation package is used from this folder, without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "emputation-python"))

import torch  # noqa: E402

from simstudy import figures, tasks  # noqa: E402
from simstudy.config import EXPERIMENTS  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["tasks", "data", "emputation", "baselines", "aggregate", "figures",
                                          "missing", "export", "report"])
    parser.add_argument("--experiment", choices=list(EXPERIMENTS))
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS),
                        help="experiments to aggregate/plot (default: all)")
    parser.add_argument("--dataset", choices=["concrete", "ccpp", "wine"])
    parser.add_argument("--task-id", type=int, help="1-based array index (default: $SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--output-dir", default=None, help="output root (default: simulation/output)")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads (default: $SLURM_CPUS_PER_TASK or all)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (emputation stage)")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="emputation: accumulate gradients over chunks of this many terms (caps memory)")
    parser.add_argument("--methods", nargs="+", help="baselines to run (default: all of the experiment)")
    parser.add_argument("--gain-alpha", type=float, default=None, help="GAIN alpha (default: 100, as in config.R)")
    parser.add_argument("--missforest-backend", choices=["randomForest", "ranger"], default=None,
                        help="missForest trees (default: randomForest, i.e. missForest <= 1.5)")
    parser.add_argument("--save-imputations", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stage-name", choices=["emputation", "baselines"], default="emputation",
                        help="missing: which array stage to check")
    parser.add_argument("--iters", default="1-20", help="export: iterations, e.g. 1-20")
    parser.add_argument("--export-dir", default=None, help="export: output folder")
    parser.add_argument("--source", choices=["output", "reference"], default="output",
                        help="figures: simulation output, or the R results shipped in reference_results/")
    args = parser.parse_args(argv)

    threads = args.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or None
    if threads:
        torch.set_num_threads(threads)
    layout = tasks.Layout(args.output_dir)
    task_id = args.task_id or (int(os.environ["SLURM_ARRAY_TASK_ID"]) if "SLURM_ARRAY_TASK_ID" in os.environ else None)

    def need(*names):
        missing = [n for n in names if getattr(args, n.replace("-", "_")) is None and not (n == "task-id" and task_id)]
        if missing:
            parser.error(f"stage {args.stage!r} requires " + ", ".join("--" + n for n in missing))

    if args.stage == "tasks":
        for exp in ([EXPERIMENTS[args.experiment]] if args.experiment else EXPERIMENTS.values()):
            print(f"{exp.name}: emputation array 1-{exp.n_emputation_tasks()}, "
                  f"baselines array 1-{exp.n_baseline_tasks()} (per dataset)")
    elif args.stage == "data":
        need("experiment")
        tasks.run_data(layout, args.experiment, [args.dataset] if args.dataset else None)
    elif args.stage == "emputation":
        need("experiment", "dataset", "task-id")
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                parser.error("--device cuda requested, but PyTorch sees no GPU "
                             "(is the CUDA module loaded and is torch a CUDA build?)")
            gpu = torch.cuda.get_device_properties(0)
            print(f"GPU: {gpu.name}, {gpu.total_memory / 2 ** 30:.0f} GB", flush=True)
        tasks.run_emputation(layout, args.experiment, args.dataset, task_id, device=args.device,
                             chunk_size=args.chunk_size, save_imputations=args.save_imputations,
                             overwrite=args.overwrite)
    elif args.stage == "baselines":
        need("experiment", "dataset", "task-id")
        tasks.run_baselines(layout, args.experiment, args.dataset, task_id, methods=args.methods,
                            gain_alpha=args.gain_alpha, missforest_backend=args.missforest_backend,
                            n_jobs=threads, save_imputations=args.save_imputations, overwrite=args.overwrite)
    elif args.stage == "aggregate":
        tasks.aggregate(layout, args.experiments)
    elif args.stage == "report":
        tasks.progress_report(layout, args.experiments)
    elif args.stage == "missing":
        need("experiment", "dataset")
        print(tasks.compress_ranges(tasks.missing_tasks(layout, args.experiment, args.dataset, args.stage_name)))
    elif args.stage == "export":
        need("experiment", "dataset", "export-dir")
        lo, _, hi = args.iters.partition("-")
        tasks.export_csv(layout, args.experiment, args.dataset, range(int(lo), int(hi or lo) + 1), args.export_dir)
    elif args.stage == "figures":
        figures.make_all(layout, source=args.source, experiments=args.experiments)


if __name__ == "__main__":
    main()
