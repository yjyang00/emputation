"""Figures and tables of the simulation study.

* ``imputation_results_{20,40}_greyscale.pdf``: Figure 2 / Figure S1 (MCAR and
  CCMV processes; metrics relative to the emputation risk matching the process).
* ``mar_{20,40}_greyscale.pdf``: Figure S2 (MAR; relative to emputation-MCAR).
* ``ccprop_energy.pdf``: Figure S3 (energy distance vs complete-case proportion).
* ``pointwise_accuracy_{20,40}.{csv,tex}``: Tables S1 and S2.
* ``comparison_with_R.csv`` / ``comparison_with_R.pdf``: Python results against
  the R results shipped in ``reference_results/r_summary.csv``.

The bar figures re-create simulation/visualization.R; the y-axis uses ggplot's
``pseudo_log_trans(base = 10)``, ``asinh(y / 2) / log(10)``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter  # noqa: E402

REFERENCE_CSV = Path(__file__).resolve().parents[1] / "reference_results" / "r_summary.csv"
DATA_ORDER = ("ccpp", "concrete", "wine")
REL_METRICS = (("MADC", "Rel. MADC"), ("Energy", "Rel. Energy"), ("MMD2", "Rel. MMD2"))
GREYS = ("#111111", "#444444", "#777777", "#AAAAAA", "#CCCCCC")
INK = "#262626"  # grey15: bar outlines and stripes
BAR_POS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.2, 7.2)

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                     "hatch.linewidth": 0.8, "pdf.fonttype": 42})


def _pseudo_log(y):
    return np.arcsinh(np.asarray(y, dtype=float) / 2) / np.log(10)


def _pseudo_log_inv(z):
    return 2 * np.sinh(np.asarray(z, dtype=float) * np.log(10))


def _figure_frame(fig, axes, col_labels, row_labels, strip_in=0.2, row_strip_gap_in=0.42):
    """Facet strips (columns on top, rows on the left) in the ggplot style of the paper."""
    W, H = fig.get_size_inches()
    style = dict(facecolor="#F5F5F5", edgecolor="#8C8C8C", linewidth=0.5)
    for j, label in enumerate(col_labels):
        p = axes[0, j].get_position()
        fig.add_artist(Rectangle((p.x0, p.y1), p.width, strip_in / H, transform=fig.transFigure, **style))
        fig.text(p.x0 + p.width / 2, p.y1 + strip_in / H / 2, label, ha="center", va="center",
                 fontsize=8.5, fontweight="bold", color="#1A1A1A")
    for i, label in enumerate(row_labels):
        p = axes[i, 0].get_position()
        x0 = p.x0 - (row_strip_gap_in + strip_in) / W
        fig.add_artist(Rectangle((x0, p.y0), strip_in / W, p.height, transform=fig.transFigure, **style))
        fig.text(x0 + strip_in / W / 2, p.y0 + p.height / 2, label, ha="center", va="center", rotation=90,
                 fontsize=8.5, fontweight="bold", color="#1A1A1A")


def bar_grid(summary, kind, path):
    """Figure 2 / S1 (``kind="main"``) or Figure S2 (``kind="mar"``) from a summary table.

    Args:
        summary: rows with columns dataset, dgp, method and ``<metric>_mean``
            for one missingness level.
        kind: "main" (MCAR and CCMV processes) or "mar".
        path: output file.
    """
    if kind == "main":
        groups = (("ccmv", 0.0), ("mcar", 9.0))
        order = ("EM", "MICE", "MF", "GAIN", "Mean", "Emp_ref", "Emp_alt")
        labels = ("EM", "MICE", "MF", "GAIN", "Mean", "Emp (reference)", "Emp (mismatched)")
        reference = {"ccmv": "Emp_ccmv", "mcar": "Emp_mcar"}
    elif kind == "mar":
        groups = (("mar", 0.0),)
        order = ("EM", "MICE", "Mean", "MF", "GAIN", "Emp_ref", "Emp_alt")
        labels = ("EM", "MICE", "Mean", "MF", "GAIN", "Emp (train: mcar)", "Emp (train: ccmv)")
        reference = {"mar": "Emp_mcar"}
    else:
        raise ValueError("kind must be 'main' or 'mar'.")
    fills = dict(zip(order[:5], GREYS), Emp_ref="#FFFFFF", Emp_alt="#FFFFFF")

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.2), sharex=True, sharey="row", squeeze=False)
    fig.subplots_adjust(left=0.115, right=0.99, top=0.945, bottom=0.135, wspace=0.07, hspace=0.06)
    xmax = groups[-1][1] + BAR_POS[-1]
    for i, (metric, _) in enumerate(REL_METRICS):
        top = 0.0
        for j, dataset in enumerate(DATA_ORDER):
            ax = axes[i, j]
            for dgp, offset in groups:
                sub = summary[(summary.dataset == dataset) & (summary.dgp == dgp)].set_index("method")
                ref_method = reference[dgp]
                alt_method = "Emp_ccmv" if ref_method == "Emp_mcar" else "Emp_mcar"
                if ref_method not in sub.index:
                    continue
                ref = sub.loc[ref_method, f"{metric}_mean"]
                for key, pos in zip(order, BAR_POS):
                    method = {"Emp_ref": ref_method, "Emp_alt": alt_method}.get(key, key)
                    if method not in sub.index:
                        continue
                    value = sub.loc[method, f"{metric}_mean"] / ref
                    top = max(top, value)
                    ax.bar(pos + offset, value, width=0.72, color=fills[key], edgecolor=INK, linewidth=0.5,
                           hatch="//////" if key == "Emp_ref" else None, zorder=2)
            if len(groups) > 1:
                ax.axvline(8.6, color="#8C8C8C", linestyle=":", linewidth=1.0, zorder=1)
            ax.axhline(1, color="#CC0000", linestyle=(0, (4, 3)), linewidth=0.8, zorder=3)
            ax.set_yscale("function", functions=(_pseudo_log, _pseudo_log_inv))
            ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color("#8C8C8C")
                spine.set_linewidth(0.6)
            ax.tick_params(axis="x", length=0, labelsize=8, labelcolor="#333333")
            ax.tick_params(axis="y", labelsize=7, labelcolor="#404040", color="#737373", width=0.5, length=2.5)
        upper = _pseudo_log_inv(_pseudo_log(top) * 1.05)
        breaks = [b for b in (0, 1, 10, 100, 500, 1000, 2000) if b <= upper]
        axes[i, 0].set_ylim(0, upper)
        axes[i, 0].yaxis.set_major_locator(FixedLocator(breaks))
        axes[i, 0].yaxis.set_minor_locator(FixedLocator([]))
        axes[i, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    centers = [offset + 4.1 for _, offset in groups]
    for ax in axes[-1]:
        ax.set_xlim(BAR_POS[0] - 0.36 - 0.45, xmax + 0.36 + 0.45)
        ax.set_xticks(centers, [f"DGP: {dgp}" for dgp, _ in groups], fontweight="bold")
    _figure_frame(fig, axes, DATA_ORDER, [lab for _, lab in REL_METRICS])
    handles = [Patch(facecolor=fills[k], edgecolor=INK, linewidth=0.5, hatch="//////" if k == "Emp_ref" else None,
                     label=lab) for k, lab in zip(order, labels)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8,
               handlelength=1.3, handleheight=1.3, columnspacing=1.0, handletextpad=0.4,
               bbox_to_anchor=(0.55, 0.0))
    fig.savefig(path)
    plt.close(fig)


def ccprop_lines(summary, path):
    """Figure S3: energy distance against the complete-case proportion."""
    colors = {"Emp_mcar": "#2166AC", "Emp_ccmv": "#2166AC", "MICE": "#D6604D", "EM": "#1B9E77"}
    styles = {"Emp_ccmv": (0, (4, 2))}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), squeeze=False)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.9, bottom=0.2, wspace=0.12)
    for ax, (dataset, title) in zip(axes[0], (("concrete", "Concrete"), ("wine", "Wine"), ("ccpp", "CCPP"))):
        sub = summary[summary.dataset == dataset]
        for method in ("EM", "MICE", "Emp_ccmv", "Emp_mcar"):
            rows = sub[sub.method == method].sort_values("cc_prop")
            if rows.empty:
                continue
            ax.plot(100 * rows.cc_prop, rows.Energy_mean, color=colors[method], linewidth=1.4,
                    linestyle=styles.get(method, "-"), marker="o", markersize=4.5, zorder=3)
        ax.set_xticks([5, 10, 20, 40, 60], ["5%", "10%", "20%", "40%", "60%"])
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(which="major", color="#EBEBEB", linewidth=0.8, zorder=0)
        ax.grid(which="minor", axis="y", color="#EBEBEB", linewidth=0.4, zorder=0)
        ax.tick_params(labelsize=8.5, labelcolor="#4D4D4D", color="#333333")
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.set_title(title, fontsize=10, fontweight="bold", color="#1A1A1A",
                     bbox=dict(facecolor="#EBEBEB", edgecolor="none", boxstyle="square,pad=0.35"), pad=8)
    axes[0, 0].set_ylabel("Energy distance", fontsize=10)
    axes[0, 1].set_xlabel("Complete case proportion", fontsize=10)
    blank = Line2D([], [], color="none")
    handles = [blank, Line2D([], [], color="#2166AC", marker="o"), Line2D([], [], color="#D6604D", marker="o"),
               Line2D([], [], color="#1B9E77", marker="o"), blank, Line2D([], [], color="black"),
               Line2D([], [], color="black", linestyle=(0, (4, 2)))]
    names = [r"$\bf{Method}$", "Emputation", "MICE", "EM", r"$\bf{Assumption}$", "MCAR", "CCMV"]
    fig.legend(handles, names, loc="lower center", ncol=7, frameon=False, fontsize=9, handlelength=2.2,
               columnspacing=1.2)
    fig.savefig(path)
    plt.close(fig)


TABLE_METHODS = (("Emp_mcar", "Emp-MCAR"), ("Emp_ccmv", "Emp-CCMV"), ("EM", "EM"), ("GAIN", "GAIN"),
                 ("MF", "MissForest"), ("MICE", "MICE"), ("Mean", "Mean"))


def pointwise_table(summary, csv_path, tex_path):
    """Tables S1 / S2: RMSE and MAE; the two smallest values of each row in bold."""
    rows = []
    for dataset, dname in (("concrete", "Concrete"), ("wine", "Wine"), ("ccpp", "CCPP")):
        for dgp in ("mcar", "ccmv", "mar"):
            sub = summary[(summary.dataset == dataset) & (summary.dgp == dgp)].set_index("method")
            if sub.empty:
                continue
            row = {"Data": dname, "DGP": dgp.upper()}
            for metric in ("RMSE", "MAE"):
                for method, label in TABLE_METHODS:
                    row[f"{metric} {label}"] = sub[f"{metric}_mean"].get(method, np.nan)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(csv_path, index=False, float_format="%.4f")
    lines = [r"\begin{tabular}{llccccccc|ccccccc}", r"\toprule",
             r"& & \multicolumn{7}{c|}{RMSE} & \multicolumn{7}{c}{MAE} \\",
             "Data & DGP & " + " & ".join(label for _, label in TABLE_METHODS) + " & "
             + " & ".join(label for _, label in TABLE_METHODS) + r" \\", r"\midrule"]
    for _, row in table.iterrows():
        cells = [row["Data"], row["DGP"]]
        for metric in ("RMSE", "MAE"):
            vals = np.array([row[f"{metric} {label}"] for _, label in TABLE_METHODS], dtype=float)
            best = set(np.argsort(np.where(np.isnan(vals), np.inf, vals))[:2])
            cells += [("--" if np.isnan(v) else (rf"\textbf{{{v:.4f}}}" if k in best else f"{v:.4f}"))
                      for k, v in enumerate(vals)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(tex_path).write_text("\n".join(lines) + "\n")
    return table


# ------------------------------------------------------------- R reference
def load_reference(path=REFERENCE_CSV):
    """R summary results shipped with the code, in the layout of :func:`summary_for_level`."""
    return pd.read_csv(path)


def summary_for_level(summary, level):
    """Rows of a Python summary (miss20 or miss40) in the reference layout."""
    sub = summary[summary.experiment == f"miss{level}"].copy()
    sub["level"] = level
    return sub


def compare_with_reference(py_summary, reference):
    """Python vs R means per (level, dataset, dgp, method, metric).

    ``z`` is the difference in units of the Monte Carlo standard error of the
    Python mean times sqrt(2) (the R results report only means, so their
    standard error is taken to be that of the Python run).
    """
    rows = []
    for _, r in reference.iterrows():
        p = py_summary[(py_summary.level == r.level) & (py_summary.dataset == r.dataset)
                       & (py_summary.dgp == r.dgp) & (py_summary.method == r.method)]
        if p.empty:
            continue
        p = p.iloc[0]
        for metric in ("RMSE", "MAE", "MADC", "Energy", "MMD2"):
            se = p[f"{metric}_std"] / np.sqrt(p.n_iter) if p.n_iter > 1 else np.nan
            rows.append(dict(level=r.level, dataset=r.dataset, dgp=r.dgp, method=r.method, metric=metric,
                             R=r[f"{metric}_mean"], Python=p[f"{metric}_mean"], n_iter=p.n_iter,
                             ratio=p[f"{metric}_mean"] / r[f"{metric}_mean"],
                             z=(p[f"{metric}_mean"] - r[f"{metric}_mean"]) / (np.sqrt(2) * se)))
    return pd.DataFrame(rows)


def variant_check(py_summary, reference):
    """Benchmark variants against the published R results (level 20).

    For GAIN (alpha = 100 vs 10) and missForest (randomForest vs ranger trees):
    mean absolute log-ratio to the R values over datasets, processes and
    metrics; the variant with the smaller value matches the paper better.
    """
    rows = []
    for base, variants in (("GAIN", ("GAIN", "GAIN_a10")), ("MF", ("MF", "MF_ranger"))):
        ref = reference[(reference.level == 20) & (reference.method == base)]
        for variant in variants:
            py = py_summary[(py_summary.level == 20) & (py_summary.method == variant)]
            merged = ref.merge(py, on=["dataset", "dgp"], suffixes=("_R", "_Py"))
            if merged.empty:
                continue
            logs = [np.abs(np.log(merged[f"{m}_mean_Py"] / merged[f"{m}_mean_R"]))
                    for m in ("RMSE", "MAE", "MADC", "Energy", "MMD2")]
            rows.append(dict(method=base, variant=variant, cells=len(merged),
                             mean_abs_log_ratio=float(np.mean(np.concatenate(logs)))))
    return pd.DataFrame(rows)


def comparison_plot(comp, path, xlabel="R (paper)"):
    """Python means against R means, one panel per metric (log-log, identity line)."""
    metrics = ("RMSE", "MAE", "MADC", "Energy", "MMD2")
    markers = {"EM": "s", "MICE": "D", "MF": "^", "GAIN": "v", "Mean": "x", "Emp_mcar": "o", "Emp_ccmv": "P"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.2))
    for ax, metric in zip(axes, metrics):
        sub = comp[comp.metric == metric]
        for method, rows in sub.groupby("method"):
            marker = markers.get(method, "o")
            style = dict(color="#333333") if marker == "x" else dict(facecolor="none", edgecolor="#333333")
            ax.scatter(rows.R, rows.Python, s=14, marker=marker, label=method, linewidth=0.7, **style)
        lo, hi = np.nanmin(sub[["R", "Python"]].values), np.nanmax(sub[["R", "Python"]].values)
        ax.plot([lo, hi], [lo, hi], color="#CC0000", linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Python")
    axes[-1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# -------------------------------------------------------------------- main
def make_all(layout, source="output", experiments=None):
    """Write all figures and tables to ``layout.figures_dir``.

    Args:
        source: "output" (aggregated Python results in ``layout.summary_dir``)
            or "reference" (the R results in reference_results/, which
            reproduces the paper's Figures 2, S1 and S2 and Tables S1-S2).
    """
    out = layout.figures_dir if source == "output" else layout.figures_dir / "from_R_reference"
    out.mkdir(parents=True, exist_ok=True)
    reference = load_reference()
    written = []
    for level in (20, 40):
        name = f"miss{level}"
        if experiments and name not in experiments:
            continue
        if source == "reference":
            summary = reference[reference.level == level]
        else:
            path = layout.summary_dir / f"summary_{name}.csv"
            if not path.exists():
                print(f"[figures] {path} not found; run the aggregate stage first.")
                continue
            summary = summary_for_level(pd.read_csv(path), level)
        bar_grid(summary, "main", out / f"imputation_results_{level}_greyscale.pdf")
        bar_grid(summary, "mar", out / f"mar_{level}_greyscale.pdf")
        pointwise_table(summary, out / f"pointwise_accuracy_{level}.csv", out / f"pointwise_accuracy_{level}.tex")
        written += [f"imputation_results_{level}_greyscale.pdf", f"mar_{level}_greyscale.pdf",
                    f"pointwise_accuracy_{level}.csv/.tex"]
    if source == "output":
        path = layout.summary_dir / "summary_ccprop.csv"
        if (not experiments or "ccprop" in experiments) and path.exists():
            ccprop_lines(pd.read_csv(path), out / "ccprop_energy.pdf")
            written.append("ccprop_energy.pdf")
        frames = [summary_for_level(pd.read_csv(layout.summary_dir / f"summary_miss{lv}.csv"), lv)
                  for lv in (20, 40) if (layout.summary_dir / f"summary_miss{lv}.csv").exists()]
        if frames:
            variants = variant_check(pd.concat(frames, ignore_index=True), reference)
            if len(variants):
                variants.to_csv(out / "variant_check.csv", index=False)
                written.append("variant_check.csv")
            comp = compare_with_reference(pd.concat(frames, ignore_index=True), reference)
            comp.to_csv(out / "comparison_with_R.csv", index=False)
            comparison_plot(comp, out / "comparison_with_R.pdf")
            written += ["comparison_with_R.csv", "comparison_with_R.pdf"]
    print(f"[figures] wrote to {out}: " + ", ".join(written))
    return written
