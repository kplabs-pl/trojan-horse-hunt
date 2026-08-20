"""Check the deterministic reproducibility claims of this package.

Runs the parts that must give identical results on any machine: trigger generation and the
evaluation metric. Needs no GPU, no git-LFS content and no Kaggle downloads.

    python ci/verify_triggers_reproducibility.py

Exits non-zero if any check fails. What is deliberately *not* checked here is anything that
depends on model training, which is non-deterministic by design -- see REPRODUCIBILITY.md.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
GROUND_TRUTH = REPO / "data" / "ground_truths.csv"
SUBMISSIONS = REPO / "top_solutions" / "submission_files"
N_MODELS = 45

failures: list[str] = []


def report(ok: bool, name: str, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_triggers() -> None:
    """generate_all.py must reproduce every published trigger, and must not touch configs."""
    trigger_yamls = {
        i: (REPO / "triggers" / f"trigger_model_{i}" / f"trigger_model_{i}.yaml") for i in range(1, N_MODELS + 1)
    }
    exp_yamls = {
        i: (REPO / "experiments" / f"experiment_model_{i}" / "experiment.yaml") for i in range(1, N_MODELS + 1)
    }
    before_trigger = {i: digest(p) for i, p in trigger_yamls.items() if p.exists()}
    before_exp = {i: digest(p) for i, p in exp_yamls.items() if p.exists()}

    proc = subprocess.run(
        [sys.executable, str(REPO / "triggers" / "generate_all.py")],
        cwd=REPO, capture_output=True, text=True,
    )
    report(proc.returncode == 0, "generate_all.py runs",
           "" if proc.returncode == 0 else (proc.stderr.strip().splitlines() or ["?"])[-1])
    if proc.returncode != 0:
        return

    tol = 1e-7
    gt = pd.read_csv(GROUND_TRUTH, index_col="model_id").drop(columns=["Usage"])
    mismatched, worst = [], 0.0
    for i in range(1, N_MODELS + 1):
        csv = REPO / "triggers" / f"trigger_model_{i}" / f"trigger_model_{i}.csv"
        if not csv.exists():
            mismatched.append(f"{i}(missing)")
            continue
        produced = pd.read_csv(csv).to_numpy().T.astype(np.float64)
        published = gt.loc[i].to_numpy().reshape(3, 75).astype(np.float64)
        deviation = float(np.abs(produced - published).max())
        worst = max(worst, deviation)
        if deviation > tol:
            mismatched.append(f"{i}(max|d|={deviation:.2e})")
    report(not mismatched,
           f"all {N_MODELS} triggers match ground_truths.csv to the precision it stores",
           f"max deviation {worst:.1e} (tolerance {tol:.0e})" if not mismatched
           else "differs: " + ", ".join(mismatched))

    drifted = [i for i, d in before_trigger.items() if digest(trigger_yamls[i]) != d]
    report(not drifted, "regenerated trigger yaml files unchanged",
           "" if not drifted else f"changed: {drifted}")

    clobbered = [i for i, d in before_exp.items() if digest(exp_yamls[i]) != d]
    report(not clobbered, "hand-tuned experiment.yaml files left untouched",
           "" if not clobbered else f"overwritten: {clobbered}")


def check_leaderboard() -> None:
    """The metric in this repo must reproduce the official public/private scores."""
    sol = pd.read_csv(GROUND_TRUTH)
    value_cols = sol.columns.drop(["model_id", "Usage"]).tolist()

    def score(sub: pd.DataFrame, split: str) -> float:
        ids = sol.loc[sol.Usage == split, "model_id"].tolist()
        truth = sol.loc[sol.Usage == split].set_index("model_id")[value_cols]
        pred = sub.set_index("model_id").loc[ids, value_cols]
        abs_diff = (truth - pred).abs().values
        with_zero = truth.copy()
        with_zero["zero"] = 0
        rng = (with_zero.max(axis=1) - with_zero.min(axis=1)).values
        rng[rng == 0] = np.finfo(np.float32).eps
        nmae = abs_diff / rng[:, None]
        nmae[nmae > 1] = 1
        return float(nmae.mean())

    checked, worst, bad = 0, 0.0, []
    for team_dir in sorted(p for p in SUBMISSIONS.iterdir() if p.is_dir()):
        csvs = sorted(team_dir.glob("*.csv"))
        if not csvs:
            continue
        m = re.search(r"Pub_(\d+)_(\d+)_Priv_(\d+)_(\d+)", csvs[0].name)
        if not m:
            continue
        sub = pd.read_csv(csvs[0])
        for split, expected in (("Public", float(f"{m.group(1)}.{m.group(2)}")),
                                ("Private", float(f"{m.group(3)}.{m.group(4)}"))):
            got = score(sub, split)
            checked += 1
            worst = max(worst, abs(got - expected))
            if abs(got - expected) >= 1e-5:
                bad.append(f"{team_dir.name} {split}: {got:.5f} vs {expected:.5f}")
    report(checked > 0 and not bad,
           f"{checked} official leaderboard scores reproduced within 1e-5",
           f"max deviation {worst:.2e}" if not bad else "; ".join(bad))


def check_figures() -> None:
    """Figure scripts that need neither torch nor git-LFS content."""
    jobs = [(f"figures_for_article/Figure_{n}.py", [f"figures_for_article/Figure_{n}.pdf"])
            for n in (6, 7, 8, 9, 10)]
    jobs.append(("figures_for_article/Figure_11_12_13.py",
                 [f"figures_for_article/Figure_{n}.pdf" for n in (11, 12, 13)]))
    for script, outputs in jobs:
        for out in outputs:
            (REPO / out).unlink(missing_ok=True)
        proc = subprocess.run([sys.executable, script], cwd=REPO, capture_output=True, text=True)
        missing = [o for o in outputs if not (REPO / o).exists()]
        ok = proc.returncode == 0 and not missing
        detail = ""
        if proc.returncode != 0:
            detail = (proc.stderr.strip().splitlines() or ["?"])[-1]
        elif missing:
            detail = f"no output: {missing}"
        report(ok, f"{Path(script).name} produces {', '.join(Path(o).name for o in outputs)}", detail)


if __name__ == "__main__":
    print(f"verifying {REPO}\n")
    check_triggers()
    check_leaderboard()
    check_figures()
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: " + ", ".join(failures))
        sys.exit(1)
    print("all checks passed")
