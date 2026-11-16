"""Stages of the simulation pipeline.

1. ``run_data``: generate the missingness masks of every iteration.
2. ``run_emputation``: one array task = one emputation fit (setting, iteration,
   training mechanism), evaluated right away.
3. ``run_baselines``: one array task = all benchmark methods on one
   (setting, iteration), each evaluated right away.
4. ``aggregate``: collect the per-task metric files into summary tables.

Output layout (``root`` defaults to ``simulation/output``)::

    root/<experiment>/<dataset>/<setting>/masks.npz
    root/<experiment>/<dataset>/<setting>/metrics/iter_###_<method>.csv
    root/<experiment>/<dataset>/<setting>/imputations/iter_###_<method>.npz   (optional)
    root/summary/{metrics,summary}_<experiment>.csv
    root/results/results_<20|40|ccprop>/<dataset>_dgp<setting>.csv            (layout of Code/simulation/results)
"""

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from emputation import emputation

from .baselines import BASELINES
from .config import EMPUTATION, EXPERIMENTS, GAIN, M_IMP, MISSFOREST_BACKEND, VARIANTS, seed_for
from .datasets import load_dataset
from .dgp import generate_mask
from .metrics import METRICS, Evaluator

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "output"


class Layout:
    """File locations of a simulation run."""

    def __init__(self, root=None):
        self.root = Path(root) if root else DEFAULT_ROOT

    def setting_dir(self, exp, dataset, key):
        return self.root / exp / dataset / key

    def masks(self, exp, dataset, key):
        return self.setting_dir(exp, dataset, key) / "masks.npz"

    def metrics(self, exp, dataset, key, it, method):
        return self.setting_dir(exp, dataset, key) / "metrics" / f"iter_{it:03d}_{method}.csv"

    def imputations(self, exp, dataset, key, it, method):
        return self.setting_dir(exp, dataset, key) / "imputations" / f"iter_{it:03d}_{method}.npz"

    @property
    def summary_dir(self):
        return self.root / "summary"

    @property
    def figures_dir(self):
        return self.root / "figures"

    @property
    def results_dir(self):
        return self.root / "results"


def _atomic_write(path, write):
    """Write through a temporary file so readers never see partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# --------------------------------------------------------------------- data
def make_masks(exp, dataset, setting):
    """Missingness masks of all iterations, shape (n_iter, n, d); deterministic."""
    X, _ = load_dataset(dataset)
    seeds = np.random.SeedSequence(seed_for("data", exp.name, dataset, setting.key)).spawn(exp.n_iter)
    return np.stack([generate_mask(X, dataset, setting, np.random.default_rng(s)) for s in seeds])


def load_masks(layout, exp, dataset, setting):
    """Masks of a setting, generated and cached on first use."""
    path = layout.masks(exp.name, dataset, setting.key)
    if not path.exists():
        masks = make_masks(exp, dataset, setting)
        _atomic_write(path, lambda f: np.savez_compressed(f, masks=np.packbits(masks, axis=-1),
                                                          shape=np.array(masks.shape)))
    with np.load(path) as f:
        shape = tuple(f["shape"])
        return np.unpackbits(f["masks"], axis=-1, count=shape[-1]).astype(bool).reshape(shape)


def run_data(layout, exp_name, datasets=None):
    """Generate (or load) the masks of an experiment and print their summary."""
    exp = EXPERIMENTS[exp_name]
    rows = []
    for dataset in datasets or exp.datasets:
        for setting in exp.settings:
            masks = load_masks(layout, exp, dataset, setting)
            rows.append(dict(experiment=exp.name, dataset=dataset, setting=setting.key,
                             missing_rate=masks.mean(),
                             complete_rows=(~masks.any(axis=2)).mean(),
                             patterns_per_iter=np.mean([len(np.unique(m, axis=0)) for m in masks])))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return table


# ------------------------------------------------------------------ methods
def _unit(layout, exp, dataset, setting, it):
    X, _ = load_dataset(dataset)
    mask = load_masks(layout, exp, dataset, setting)[it - 1]
    return X, mask, np.where(mask, np.nan, X)


def _record(layout, exp, dataset, setting, it, method, imputations, evaluator, extra, save_imputations):
    rng = np.random.default_rng(seed_for("metrics", exp.name, dataset, setting.key, it, method))
    row = dict(experiment=exp.name, dataset=dataset, setting=setting.key, dgp=setting.dgp,
               cc_prop=setting.cc_prop if setting.dgp != "ccmv" else np.nan, iter=it, method=method)
    row.update(evaluator.evaluate(imputations, rng))
    row.update(extra)
    _atomic_write(layout.metrics(exp.name, dataset, setting.key, it, method),
                  lambda f: pd.DataFrame([row]).to_csv(f, index=False))
    if save_imputations:
        _atomic_write(layout.imputations(exp.name, dataset, setting.key, it, method),
                      lambda f: np.savez_compressed(f, imputations=np.asarray(imputations, dtype=np.float32)))
    print(" ".join(f"{k}={row[k]:.4g}" for k in METRICS), flush=True)
    return row


def run_emputation(layout, exp_name, dataset, task_id, device="cpu", chunk_size=None,
                   save_imputations=False, overwrite=False, emputation_kwargs=None):
    """Fit emputation for one (setting, iteration, mechanism) and evaluate its imputations."""
    exp = EXPERIMENTS[exp_name]
    setting, it, train = exp.emputation_task(task_id)
    method = f"Emp_{train}"
    out = layout.metrics(exp.name, dataset, setting.key, it, method)
    print(f"[emputation] {exp.name}/{dataset}/{setting.key} iter={it} train={train}", flush=True)
    if out.exists() and not overwrite:
        print(f"  exists, skipping: {out}")
        return None
    X, mask, X_na = _unit(layout, exp, dataset, setting, it)
    kwargs = dict(EMPUTATION, **(emputation_kwargs or {}))
    start = time.time()
    fit = emputation(X_na, mechanism=train, device=device, chunk_size=chunk_size,
                     seed=seed_for("emputation", exp.name, dataset, setting.key, it, train), **kwargs)
    imputations = fit.sample(X_na, m=M_IMP)
    extra = dict(seconds=time.time() - start, n_terms=fit.terms_.n_terms, final_loss=fit.loss_history[-1, 0])
    return _record(layout, exp, dataset, setting, it, method, list(imputations), Evaluator(X, mask), extra,
                   save_imputations)


def run_baselines(layout, exp_name, dataset, task_id, methods=None, gain_alpha=None,
                  missforest_backend=None, n_jobs=None, save_imputations=False, overwrite=False):
    """Run and evaluate the benchmark methods on one (setting, iteration)."""
    exp = EXPERIMENTS[exp_name]
    setting, it = exp.baseline_task(task_id)
    print(f"[baselines] {exp.name}/{dataset}/{setting.key} iter={it}", flush=True)
    X, mask, X_na = _unit(layout, exp, dataset, setting, it)
    evaluator = Evaluator(X, mask)
    rows = []
    for method in methods or exp.baselines:
        out = layout.metrics(exp.name, dataset, setting.key, it, method)
        if out.exists() and not overwrite:
            print(f"  {method}: exists, skipping")
            continue
        base, override = VARIANTS.get(method, (method, {}))
        params = {}
        if base == "GAIN":
            params = dict(GAIN, alpha=GAIN["alpha"] if gain_alpha is None else gain_alpha)
        elif base == "MF":
            params = dict(backend=missforest_backend or MISSFOREST_BACKEND, n_jobs=n_jobs)
        params.update(override)
        rng = np.random.default_rng(seed_for("baseline", exp.name, dataset, setting.key, it, method))
        start = time.time()
        imputations = BASELINES[base](X_na, m=M_IMP, rng=rng, **params)
        print(f"  {method} ({time.time() - start:.1f}s): ", end="")
        rows.append(_record(layout, exp, dataset, setting, it, method, imputations, evaluator,
                            dict(seconds=time.time() - start), save_imputations))
    return rows


# ---------------------------------------------------------------- aggregate
def collect_metrics(layout, exp_name):
    """All per-task metric rows of an experiment as one DataFrame."""
    files = sorted((layout.root / exp_name).glob("*/*/metrics/iter_*.csv"))
    if not files:
        raise FileNotFoundError(f"No metric files under {layout.root / exp_name}.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def summarize(metrics):
    """Mean and standard deviation over iterations per (dataset, setting, method)."""
    keys = ["experiment", "dataset", "setting", "dgp", "cc_prop", "method"]
    grouped = metrics.groupby(keys, dropna=False)
    summary = grouped[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "n_iter", grouped.size())
    return summary.reset_index()


def progress_report(layout, exp_names=None):
    """Print, per (experiment, dataset, setting, method), the number of finished
    tasks, their run time and the size of the emputation fits (training terms)."""
    frames = []
    for name in exp_names or list(EXPERIMENTS):
        try:
            frames.append(collect_metrics(layout, name))
        except FileNotFoundError:
            pass
    if not frames:
        print("[report] no finished tasks yet")
        return None
    df = pd.concat(frames, ignore_index=True)
    if "n_terms" not in df:
        df["n_terms"] = np.nan
    table = df.groupby(["experiment", "dataset", "setting", "method"]).agg(
        done=("iter", "count"), sec_median=("seconds", "median"), sec_max=("seconds", "max"),
        terms_median=("n_terms", "median"), RMSE=("RMSE", "mean"), Energy=("Energy", "mean"))
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(table.to_string(float_format=lambda v: f"{v:.4g}"))
    return table


R_METHOD_ORDER = ("EM", "MICE", "MF", "GAIN", "Mean", "Emp_mcar", "Emp_ccmv")  # rows of R's summary_df


def export_results(layout, exp_name, summary):
    """Write one table per (dataset, setting) in the layout of the R results (Code/simulation/results):
    results/results_<20|40|ccprop>/<dataset>_dgp<setting>.csv, one row per method (R's order; the benchmark
    variants last) with the columns of R's summary_df (mean of each metric over iterations), followed by the
    standard deviations and the number of iterations."""
    folder = layout.results_dir / f"results_{exp_name.removeprefix('miss')}"
    folder.mkdir(parents=True, exist_ok=True)
    summary = summary.rename(columns={f"{m}_std": f"{m}_sd" for m in METRICS})
    cols = ["method"] + [f"{m}_mean" for m in METRICS] + [f"{m}_sd" for m in METRICS] + ["n_iter"]
    rank = {m: k for k, m in enumerate(R_METHOD_ORDER)}
    for (dataset, setting), table in summary.groupby(["dataset", "setting"]):
        table = table.assign(rank=table.method.map(rank).fillna(len(rank))).sort_values(["rank", "method"])
        table[cols].to_csv(folder / f"{dataset}_dgp{setting}.csv", index=False)
    return folder


def aggregate(layout, exp_names=None):
    """Write metrics_<exp>.csv and summary_<exp>.csv; report missing tasks."""
    out = {}
    for name in exp_names or list(EXPERIMENTS):
        exp = EXPERIMENTS[name]
        try:
            metrics = collect_metrics(layout, name)
        except FileNotFoundError as err:
            print(f"[aggregate] {err}")
            continue
        summary = summarize(metrics)
        layout.summary_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(layout.summary_dir / f"metrics_{name}.csv", index=False)
        summary.to_csv(layout.summary_dir / f"summary_{name}.csv", index=False)
        folder = export_results(layout, name, summary)
        incomplete = summary[summary.n_iter < exp.n_iter]
        print(f"[aggregate] {name}: {len(metrics)} rows, {len(summary)} (dataset, setting, method) cells; "
              f"per-setting tables in {folder}")
        if len(incomplete):
            print("  cells with fewer than {} iterations:\n{}".format(
                exp.n_iter, incomplete[["dataset", "setting", "method", "n_iter"]].to_string(index=False)))
        out[name] = summary
    return out


# ------------------------------------------------------------------ helpers
def missing_tasks(layout, exp_name, dataset, stage):
    """1-based array indices of `stage` ("emputation" or "baselines") without output."""
    exp = EXPERIMENTS[exp_name]
    ids = []
    if stage == "emputation":
        for t in range(1, exp.n_emputation_tasks() + 1):
            setting, it, train = exp.emputation_task(t)
            if not layout.metrics(exp.name, dataset, setting.key, it, f"Emp_{train}").exists():
                ids.append(t)
    else:
        for t in range(1, exp.n_baseline_tasks() + 1):
            setting, it = exp.baseline_task(t)
            if any(not layout.metrics(exp.name, dataset, setting.key, it, m).exists() for m in exp.baselines):
                ids.append(t)
    return ids


def compress_ranges(ids):
    """[1, 2, 3, 7, 9, 10] -> "1-3,7,9-10" (the format of sbatch --array)."""
    parts, start = [], None
    for k, v in enumerate(ids):
        if start is None:
            start = v
        if k + 1 == len(ids) or ids[k + 1] != v + 1:
            parts.append(str(start) if start == v else f"{start}-{v}")
            start = None
    return ",".join(parts)


def export_csv(layout, exp_name, dataset, iters, out_dir):
    """Write incomplete datasets as CSV files (NA = missing) for the R side of the consistency study."""
    exp = EXPERIMENTS[exp_name]
    X, columns = load_dataset(dataset)
    for setting in exp.settings:
        masks = load_masks(layout, exp, dataset, setting)
        folder = Path(out_dir) / exp.name / dataset / setting.key
        folder.mkdir(parents=True, exist_ok=True)
        for it in iters:
            pd.DataFrame(np.where(masks[it - 1], np.nan, X), columns=columns).to_csv(
                folder / f"iter_{it:03d}.csv", index=False, na_rep="NA")
    print(f"[export] {exp.name}/{dataset}: iterations {compress_ranges(list(iters))} -> {out_dir}")
