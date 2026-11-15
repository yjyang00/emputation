"""Configuration of the simulation study (Section 7 and Supplement A-B).

Three experiments reproduce the simulation figures and tables:

* ``miss20``: 20% missingness under MCAR, CCMV and MAR
  (Figure 2, Figure S2a, Table S1).
* ``miss40``: 40% missingness under MCAR, CCMV and MAR
  (Figure S1, Figure S2b, Table S2).
* ``ccprop``: MAR with 40% missingness and complete-case proportions
  5%-60% (Figure S3).

Each experiment crosses datasets x settings x iterations. In each
(dataset, setting, iteration) unit the missing data are generated once and
every method -- emputation trained under MCAR and CCMV, and the
benchmarks -- imputes the same incomplete dataset.
"""

import hashlib
from dataclasses import dataclass

N_ITER = 100  # Monte Carlo repetitions
M_IMP = 10  # imputations per multiple-imputation method
DATASETS = ("concrete", "ccpp", "wine")
TRAIN_MECHANISMS = ("mcar", "ccmv")

# Emputation hyperparameters (scripts/config.R; Section 7 of the paper):
# full-batch training, i.e. one optimizer step per epoch on all terms.
EMPUTATION = dict(num_epochs=500, lr=1e-4, B=2, hidden_dim=500, num_layer=3, batch_norm=True,
                  batch_size=None)

# GAIN hyperparameters from scripts/config.R. The paper's text reports
# alpha = 10, the R configuration alpha = 100 (the default of the original
# GAIN code); override with `run_simulation.py baselines --gain-alpha`.
GAIN = dict(batch_size=128, hint_rate=0.9, alpha=100.0, iterations=1000)

# missForest trees: "randomForest" reproduces missForest <= 1.5 (randomForest,
# nodesize 5), "ranger" the default backend of missForest >= 1.6 (min.bucket 5).
# randomForest matches the published MF results (concrete, MCAR 20%: RMSE
# 0.555 vs 0.557 in the paper; ranger 0.604).
MISSFOREST_BACKEND = "randomForest"

# Variants of two benchmarks whose settings in the original runs are ambiguous
# (GAIN alpha: 100 in config.R vs 10 in the paper; missForest backend). They
# run in the miss20 baseline jobs; the figures stage reports which variant is
# closer to the published R results (variant_check.csv). The figures use MF
# and GAIN.
VARIANTS = {"MF_ranger": ("MF", {"backend": "ranger"}), "GAIN_a10": ("GAIN", {"alpha": 10.0})}


@dataclass(frozen=True)
class Setting:
    """A missingness-generating process.

    Attributes:
        dgp: "mcar", "mar" or "ccmv".
        param: MCAR cell probability rho, MAR overall missing rate, or the
            CCMV intercept alpha_r.
        cc_prop: fraction of rows kept complete before MCAR/MAR missingness
            is applied (not used by CCMV).
        label: short name used in file paths.
    """

    dgp: str
    param: float
    cc_prop: float = 0.2
    label: str = ""

    @property
    def key(self):
        return self.label or self.dgp


@dataclass(frozen=True)
class Experiment:
    name: str
    settings: tuple
    baselines: tuple = ("Mean", "EM", "MICE", "MF", "GAIN")
    trains: tuple = TRAIN_MECHANISMS
    datasets: tuple = DATASETS
    n_iter: int = N_ITER

    def setting(self, key):
        for s in self.settings:
            if s.key == key:
                return s
        raise KeyError(f"Experiment {self.name} has no setting {key!r}.")

    # Array-task layout: setting-major, then iteration, then mechanism.
    def n_emputation_tasks(self):
        return len(self.settings) * self.n_iter * len(self.trains)

    def n_baseline_tasks(self):
        return len(self.settings) * self.n_iter

    def emputation_task(self, task_id):
        """Decode a 1-based array index into (setting, iteration, mechanism)."""
        if not 1 <= task_id <= self.n_emputation_tasks():
            raise ValueError(f"task_id must be in 1..{self.n_emputation_tasks()} for {self.name}.")
        k, t = divmod(task_id - 1, len(self.trains))
        s, i = divmod(k, self.n_iter)
        return self.settings[s], i + 1, self.trains[t]

    def baseline_task(self, task_id):
        """Decode a 1-based array index into (setting, iteration)."""
        if not 1 <= task_id <= self.n_baseline_tasks():
            raise ValueError(f"task_id must be in 1..{self.n_baseline_tasks()} for {self.name}.")
        s, i = divmod(task_id - 1, self.n_iter)
        return self.settings[s], i + 1


CC_GRID = (0.05, 0.10, 0.20, 0.40, 0.60)

EXPERIMENTS = {
    "miss20": Experiment("miss20", settings=(Setting("mcar", 0.25), Setting("ccmv", -4.0),
                                             Setting("mar", 0.20)),
                         baselines=("Mean", "EM", "MICE", "MF", "GAIN", "MF_ranger", "GAIN_a10")),
    "miss40": Experiment("miss40", settings=(Setting("mcar", 0.50), Setting("ccmv", -0.5),
                                             Setting("mar", 0.40))),
    "ccprop": Experiment("ccprop", baselines=("EM", "MICE"),
                         settings=tuple(Setting("mar", 0.40, cc_prop=cc, label=f"mar_cc{round(100 * cc):02d}")
                                        for cc in CC_GRID)),
}


def seed_for(*keys):
    """Reproducible 32-bit seed derived from a sequence of keys."""
    digest = hashlib.sha256("/".join(str(k) for k in keys).encode()).digest()
    return int.from_bytes(digest[:4], "little")
