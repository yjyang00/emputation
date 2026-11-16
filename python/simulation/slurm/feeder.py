#!/usr/bin/env python3
"""Submit the job arrays of a plan without exceeding the cluster's submit limit.

Clusters cap the number of jobs a user may have queued (MaxSubmitJobs; every
array task counts as a job), and the simulation study has ~10,000 tasks.
`slurm/submit_all.sh` writes the arrays to submit into a plan file and starts
this script as a small long-running job (`slurm/feeder.sbatch`). Every
FEED_EVERY minutes (default 15) it submits, in plan order and in arrays of at
most FEED_CHUNK tasks (default 100), the tasks that are

- not finished (simulation arrays: the task's metrics files do not exist),
- not queued (no pending or running job with the same name and array index),
- not submitted from this plan before (failed tasks are not retried; rerunning
  submit_all.sh resubmits them),

until sbatch refuses because of the submit limit. The limit is then known (or
set by FEED_MAX_SUBMIT), and the feeder keeps FEED_RESERVE (default 20) places
in the queue free for other jobs. A unit with `after` (the aggregate job) is
submitted once all units it refers to are submitted, with a dependency on
those of their jobs that are still queued. The feeder exits when the whole plan
is submitted, and hands over to a new feeder job shortly before its time limit.
It also generates the missingness masks that do not exist yet, so that the
tasks do not each generate them.

Plan file: one unit per line, with the tab-separated fields
    name  kind  range  dir  script  env  after  sbatch-options
kind: emputation | baselines (simulation arrays) | array | job; range: array
indices (- for a job); dir: folder to submit from; env: KEY=VALUE;... (or -);
after: afterok|afterany:NAME-PATTERN,... (or -).

    python slurm/feeder.py out/plan_<date>.tsv      (run by slurm/feeder.sbatch)
"""

import fnmatch
import getpass
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
# the simulation code and the emputation package are used from this repository, without installation
sys.path[:0] = [str(SIM), str(SIM.parent / "emputation-python")]

from simstudy.config import EXPERIMENTS  # noqa: E402
from simstudy.tasks import Layout, compress_ranges, load_masks, missing_tasks  # noqa: E402

EVERY = 60 * float(os.environ.get("FEED_EVERY", 15))
CHUNK = int(os.environ.get("FEED_CHUNK", 100))
RESERVE = int(os.environ.get("FEED_RESERVE", 20))
SELF = os.environ.get("SLURM_JOB_ID")  # this feeder job (None outside SLURM)
# refusals that go away by waiting: the submit limit, a busy or unreachable controller
RETRY = re.compile(r"SubmitJob|temporarily|timed out|socket|unable to contact|try again", re.IGNORECASE)


class Retry(Exception):
    """The scheduler refused for now; try again in the next round."""


@dataclass
class Unit:
    name: str
    kind: str
    ids: list  # array indices ([] for a single job)
    dir: str
    script: str
    env: dict
    after: tuple  # (dependency type, job name patterns) or None
    options: list
    key: str


def matches(name, patterns):
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def expand(spec):
    """'1-3,7' -> [1, 2, 3, 7]."""
    ids = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        ids.extend(range(int(lo), int(hi or lo) + 1))
    return ids


def seconds(text):
    """SLURM duration '[D-]HH:MM:SS' or 'MM:SS' -> seconds (ValueError for UNLIMITED)."""
    days, _, hms = text.rpartition("-")
    parts = [int(p) for p in hms.split(":")]
    h, m, s = [0] * (3 - len(parts)) + parts
    return int(days or 0) * 86400 + h * 3600 + m * 60 + s


def read_plan(path):
    units = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, kind, rng, folder, script, env, after, options = line.split("\t")
        if kind not in ("emputation", "baselines", "array", "job"):
            raise ValueError(f"unknown kind {kind!r} in the plan line: {line}")
        typ, _, patterns = after.partition(":")
        units.append(Unit(name, kind, [] if rng == "-" else expand(rng), folder, script,
                          dict(kv.split("=", 1) for kv in env.split(";")) if env != "-" else {},
                          (typ, patterns.split(",")) if after != "-" else None, options.split(),
                          key=f"{name}:{rng}"))
    return units


def slurm(cmd, cwd=None, env=None):
    """Run a SLURM command and return its output; Retry if waiting may help, else RuntimeError."""
    # without the SLURM_* variables of the feeder job, which would leak into the submitted jobs
    full_env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_") or k == "SLURM_CONF"}
    r = subprocess.run(cmd, cwd=cwd, env={**full_env, **(env or {})}, capture_output=True, text=True)
    if r.returncode == 0:
        return r.stdout.strip()
    msg = " ".join(r.stderr.split()) or f"exit code {r.returncode}"
    if RETRY.search(msg):
        raise Retry(msg)
    raise RuntimeError(f"{' '.join(cmd)} (in {cwd or os.getcwd()}) failed: {msg}")


def queue():
    """(job id, array index or None, job name) of the user's pending and running jobs."""
    try:
        out = slurm(["squeue", "-u", getpass.getuser(), "-h", "-r", "-o", "%F|%K|%j"])
    except RuntimeError as err:
        raise Retry(str(err)) from err
    rows = []
    for line in out.splitlines():
        job, task, name = line.split("|", 2)
        rows.append((job.strip(), int(task) if task.strip().isdigit() else None, name.strip()))
    return rows


class Feeder:
    def __init__(self, plan):
        self.plan = str(Path(plan).resolve())
        self.units = read_plan(plan)
        self.state_file = Path(self.plan).with_suffix(".json")
        self.state = (json.loads(self.state_file.read_text()) if self.state_file.exists()
                      else {"submitted": {}, "jobs": {}, "limit": None})
        if os.environ.get("FEED_MAX_SUBMIT"):
            self.state["limit"] = int(os.environ["FEED_MAX_SUBMIT"])
        self.layout = Layout()

    def save(self):
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        os.replace(tmp, self.state_file)

    def make_masks(self):
        for exp, ds in sorted({(u.env["EXPERIMENT"], u.env["DATASET"]) for u in self.units
                               if u.kind in ("emputation", "baselines")}):
            for setting in EXPERIMENTS[exp].settings:
                if not self.layout.masks(exp, ds, setting.key).exists():
                    log(f"generating the masks of {exp}/{ds}/{setting.key}")
                    load_masks(self.layout, EXPERIMENTS[exp], ds, setting)

    def todo(self, unit, queued, missing):
        """Array indices of `unit` still to submit ([None] for a job not submitted yet)."""
        if unit.kind == "job":
            return [] if unit.key in self.state["submitted"] else [None]
        skip = queued[unit.name].union(self.state["submitted"].get(unit.key, []))
        ids = [i for i in unit.ids if i not in skip]
        if ids and unit.kind in ("emputation", "baselines"):
            k = (unit.env["EXPERIMENT"], unit.env["DATASET"], unit.kind)
            if k not in missing:
                missing[k] = set(missing_tasks(self.layout, *k))
            ids = [i for i in ids if i in missing[k]]
        return ids

    def dependency(self, k, done, jobs):
        """sbatch arguments for the `after` of unit k; None while a unit it refers to is not fully submitted."""
        unit = self.units[k]
        if unit.after is None:
            return []
        typ, patterns = unit.after
        if not all(done[u.key] for u in self.units[:k] if matches(u.name, patterns)):
            return None
        ids = sorted({j for name, js in jobs.items() if matches(name, patterns) for j in js}, key=int)
        return [f"--dependency={typ}:{':'.join(ids)}"] if ids else []

    def submit(self, unit, ids, dep, count, queued, jobs):
        """Submit `ids` of `unit` in arrays of at most CHUNK tasks while the queue has room.
        Returns the indices not submitted, the queue size and why it stopped (None if it did not)."""
        while ids:
            size = min(len(ids), CHUNK) if unit.ids else 1
            limit = self.state["limit"]
            if limit:
                size = min(size, limit - RESERVE - count)
                if size < 1:
                    return ids, count, f"the submit limit is {limit} jobs; {RESERVE} places are kept free"
            chunk = ids[:size] if unit.ids else []
            args = ([f"--array={compress_ranges(chunk)}"] if chunk else []) + dep
            try:
                job = slurm(["sbatch", "--parsable", f"--job-name={unit.name}", *args, *unit.options, unit.script],
                            cwd=unit.dir, env=unit.env).split(";")[0]
            except Retry as err:
                if "SubmitJob" in str(err):  # the limit lies between the current queue size and that + size
                    try:
                        count = len(queue())
                    except Retry:
                        pass
                    self.state["limit"] = min(count, self.state["limit"] or count)
                    self.save()
                return ids, count, str(err)
            log(f"submitted {unit.name} {' '.join(args)}: job {job}")
            self.state["submitted"].setdefault(unit.key, []).extend(chunk or [0])
            self.state["jobs"].setdefault(unit.key, []).append(job)
            self.save()
            queued[unit.name].update(chunk)
            jobs[unit.name].add(job)
            count += size
            ids = ids[size:]
        return ids, count, None

    def round(self):
        """One pass over the plan; True when the whole plan is submitted."""
        rows = queue()
        count = len(rows)
        queued, jobs = defaultdict(set), defaultdict(set)
        for job, task, name in rows:
            if job != SELF:
                queued[name].add(task)
                jobs[name].add(job)
        missing, done, left, stop = {}, {}, 0, None
        for k, unit in enumerate(self.units):
            ids = self.todo(unit, queued, missing)
            dep = self.dependency(k, done, jobs)
            if ids and dep is not None and stop is None:
                ids, count, stop = self.submit(unit, ids, dep, count, queued, jobs)
            done[unit.key] = not ids
            left += len(ids)
        log(f"{count} jobs queued, {left} tasks of the plan still to submit" + (f" ({stop})" if left and stop else ""))
        return left == 0

    def hand_over(self):
        """Continue in a new feeder job that starts when this one ends."""
        cmd = ["sbatch", "--parsable", f"--dependency=afterany:{SELF}", *os.environ.get("FEED_SBATCH", "").split(),
               "slurm/feeder.sbatch"]
        for _ in range(15):
            try:
                log(f"time limit near: the plan continues in feeder job {slurm(cmd, cwd=SIM, env={'PLAN': self.plan})}")
                return 0
            except Retry as err:
                log(f"the next feeder job was refused ({err}); retrying in 1 min")
                time.sleep(60)
        log(f"could not submit the next feeder job; continue the plan with: cd {SIM} && PLAN={self.plan} "
            f"sbatch {' '.join(cmd[3:])}")
        return 1


def main(plan):
    feeder = Feeder(plan)
    log(f"plan {feeder.plan}: {len(feeder.units)} arrays and jobs; state in {feeder.state_file.name}")
    end = None
    if SELF:
        try:  # hand over 10 minutes before the time limit
            end = time.time() + seconds(slurm(["squeue", "-h", "-j", SELF, "-o", "%L"])) - 600
        except (ValueError, Retry, RuntimeError):
            pass
    feeder.make_masks()
    while True:
        try:
            if feeder.round():
                log("the whole plan is submitted")
                return 0
        except Retry as err:
            log(f"squeue failed ({err}); next round in {EVERY / 60:g} min")
        if end and time.time() + EVERY > end:
            return feeder.hand_over()
        time.sleep(EVERY)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
