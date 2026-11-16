#!/usr/bin/env python3
"""Failed tasks of the simulation arrays, and the memory the tasks used.

    python3 slurm/failed_tasks.py              # jobs of the last 7 days
    python3 slurm/failed_tasks.py 2026-11-15   # jobs since this date

Run from anywhere on the login node (standard library only, no environment
needed). For each row of EMP_JOBS and BASE_JOBS in slurm/submit_all.sh (array
name and index range) it takes the latest attempt of every task from sacct and
prints a table: tasks done, queued, out of memory, timed out, failed; the
memory requested and the peak memory (MaxRSS) of the completed tasks. Below the
table, for every row with failures, the array indices per state, a suggested
mem-per-cpu where tasks ran out of memory, and the last line of the error log
of a few FAILED tasks.

SLURM reports a task killed for exceeding its memory as OUT_OF_MEMORY. A CUDA
out-of-memory error on the GPU partition, or a MemoryError raised by Python,
ends as FAILED instead; the error-log lines show which.

To fix: raise mem-per-cpu of the row in slurm/submit_all.sh, stop the feeder if
it still runs (scancel --name=emp_feed), and rerun submit_all.sh, which submits
exactly the tasks without output.
"""

import glob
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
QUEUED = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED"}
UNIT = {"": 2 ** -20, "K": 2 ** -10, "M": 1, "G": 2 ** 10, "T": 2 ** 20}  # -> MB


def mb(text, cpus=1):
    """sacct memory ('41234567K', '1000Mc', '88G', '0') -> MB, or None."""
    m = re.fullmatch(r"([\d.]+)([KMGT]?)([cn]?)", text.strip())
    if not m:
        return None
    return float(m.group(1)) * UNIT[m.group(2)] * (cpus if m.group(3) == "c" else 1)


def gb(value):
    return "-" if value is None else f"{value / 1024:.1f}"


def expand(spec):
    """'1-3,7' -> [1, 2, 3, 7]."""
    ids = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        ids.extend(range(int(lo), int(hi or lo) + 1))
    return ids


def ranges(ids):
    """[1, 2, 3, 7] -> '1-3,7'."""
    ids, parts = sorted(ids), []
    for i in ids:
        if parts and parts[-1][1] == i - 1:
            parts[-1][1] = i
        else:
            parts.append([i, i])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in parts)


def plan_rows():
    """The rows of EMP_JOBS and BASE_JOBS in submit_all.sh."""
    rows, kind = [], None
    for line in (SIM / "slurm" / "submit_all.sh").read_text().splitlines():
        if re.match(r"(EMP|BASE)_JOBS=\(", line):
            kind = "emp" if line.startswith("EMP") else "base"
        elif line.startswith(")"):
            kind = None
        elif kind:
            m = re.search(r'"(\w+)\s+(\w+)\s+(\d+)-(\d+)\s+(\d+)\s+(\d+)\s+\S+"', line)
            if m:
                rows.append({"name": f"{kind}_{m[1]}_{m[2]}", "lo": int(m[3]), "hi": int(m[4]),
                             "cpus": int(m[5]), "mem": int(m[6])})
    return rows


def sacct_tasks(since):
    """Latest attempt of every task of the emp_*/base_* arrays: (name, index) -> task."""
    out = subprocess.run(["sacct", "-n", "-P", "-S", since, "-o", "JobID,JobName,State,MaxRSS,ReqMem,AllocCPUS"],
                         check=True, stdout=subprocess.PIPE, universal_newlines=True).stdout
    tasks = {}  # "123_17" -> task
    for line in out.splitlines():
        jid, name, state, rss, req, cpus = line.split("|")[:6]
        task, _, step = jid.partition(".")
        pending = re.fullmatch(r"(\d+)_\[([\d,-]+)(%\d+)?\]", task)  # pending tasks "123_[19-300%50]"
        if pending and re.match(r"(emp|base)_", name):
            for i in expand(pending[2]):
                tasks[f"{pending[1]}_{i}"] = {"name": name, "job": int(pending[1]), "idx": i, "state": "PENDING",
                                              "rss": None, "req": None, "cpus": 0}
        m = re.fullmatch(r"(\d+)_(\d+)", task)  # skips single jobs
        if not m:
            continue
        if not step:
            if re.match(r"(emp|base)_", name):
                cpus = int(cpus or 0)
                tasks[task] = {"name": name, "job": int(m[1]), "idx": int(m[2]), "state": state.split()[0],
                               "rss": None, "req": mb(req, cpus or 1), "cpus": cpus}
        elif task in tasks:  # batch/extern steps: the memory used, and OUT_OF_MEMORY on older SLURM versions
            t = tasks[task]
            used = mb(rss)
            if used is not None:
                t["rss"] = max(t["rss"] or 0, used)
            if state.startswith("OUT_OF_MEMORY"):
                t["state"] = "OUT_OF_MEMORY"
    latest = {}
    for t in tasks.values():
        k = (t["name"], t["idx"])
        if k not in latest or t["job"] > latest[k]["job"]:
            latest[k] = t
    return latest


def last_error_line(t):
    for path in glob.glob(str(SIM / "out" / f"*_{t['job']}_{t['idx']}.err")):
        lines = [x.strip() for x in Path(path).read_text(errors="replace").splitlines() if x.strip()]
        return f"{Path(path).relative_to(SIM)}: {lines[-1] if lines else '(empty)'}"
    return f"(no log out/*_{t['job']}_{t['idx']}.err)"


def main(since):
    rows = plan_rows()
    groups = defaultdict(list)
    for (name, idx), t in sacct_tasks(since).items():
        row = next((k for k, r in enumerate(rows) if r["name"] == name and r["lo"] <= idx <= r["hi"]), None)
        groups[row if row is not None else name].append(t)
    if not groups:
        print(f"no tasks of emp_*/base_* arrays in sacct since {since}")
        return 0

    head = f"{'array':<22}{'range':>9}{'CPUs':>5}{'MB/CPU':>7}{'GB':>6}{'peak GB':>8}" \
           f"{'done':>6}{'queued':>7}{'OOM':>5}{'TIMEOUT':>8}{'FAILED':>7}{'other':>6}"
    print(f"latest attempt of each task since {since} (peak GB: MaxRSS of the completed tasks)\n{head}")
    keys = [k for k in range(len(rows)) if k in groups] + sorted(k for k in groups if isinstance(k, str))
    details = []
    for k in keys:
        ts = groups[k]
        n = Counter(t["state"] for t in ts)
        done = [t["rss"] for t in ts if t["state"] == "COMPLETED" and t["rss"] is not None]
        peak = max(done) if done else None
        other = sum(c for s, c in n.items()
                    if s not in QUEUED | {"COMPLETED", "OUT_OF_MEMORY", "TIMEOUT", "FAILED"})
        if isinstance(k, int):
            r = rows[k]
            label, cols = (r["name"], f"{r['lo']}-{r['hi']}"), f"{r['cpus']:>5}{r['mem']:>7}{r['cpus'] * r['mem'] / 1024:>6.0f}"
        else:
            r, label, cols = None, (k, "?"), f"{'':>18}"
        print(f"{label[0]:<22}{label[1]:>9}{cols}{gb(peak):>8}{n['COMPLETED']:>6}"
              f"{sum(n[s] for s in QUEUED):>7}{n['OUT_OF_MEMORY']:>5}{n['TIMEOUT']:>8}{n['FAILED']:>7}{other:>6}")

        bad = defaultdict(list)
        for t in ts:
            if t["state"] not in QUEUED | {"COMPLETED"}:
                bad[t["state"]].append(t)
        if not bad:
            continue
        lines = [f"{label[0]} {label[1]}:"]
        for state, failed in sorted(bad.items()):
            lines.append(f"  {state} ({len(failed)}): {ranges(t['idx'] for t in failed)}")
        oom = bad.get("OUT_OF_MEMORY", [])
        if oom and r:
            had = max((t["req"] for t in oom if t["req"]), default=r["cpus"] * r["mem"])
            need = max(1.3 * had, 1.15 * (peak or 0))  # the tasks needed more than they had
            per_cpu = 500 * math.ceil(need / r["cpus"] / 500)
            lines.append(f"  -> they had {gb(had)} GB; set mem-per-cpu of this row to >= {per_cpu} "
                         f"({r['cpus']} x {per_cpu} MB = {r['cpus'] * per_cpu / 1024:.0f} GB)")
        if "TIMEOUT" in bad:
            lines.append("  -> raise the time limit of this row")
        for t in bad.get("FAILED", [])[:3]:
            lines.append(f"  {last_error_line(t)}")
        details.append("\n".join(lines))
    if details:
        print("\n" + "\n\n".join(details))
        print("\nAfter editing slurm/submit_all.sh: scancel --name=emp_feed (if the feeder still runs), "
              "then bash slurm/submit_all.sh")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1].startswith("-")):
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1] if len(sys.argv) == 2 else (date.today() - timedelta(days=7)).isoformat()))
