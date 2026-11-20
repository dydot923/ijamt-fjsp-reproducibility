"""Paired effect sizes and uncertainty from the original frozen experiment CSVs."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figures" / "data"
OUT = ROOT / "replication" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(values, statistic=np.mean, reps=20000, seed=90421):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(reps, len(values)), replace=True)
    estimates = statistic(samples, axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def sign_p(values):
    values = np.asarray(values, dtype=float)
    nonzero = values[np.abs(values) > 1e-12]
    wins = int(np.sum(nonzero > 0))
    losses = int(np.sum(nonzero < 0))
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / 2**n) if n else 1.0
    return wins, losses, p


def holm(ps):
    indices = sorted(range(len(ps)), key=lambda i: ps[i])
    adjusted = [0.0] * len(ps)
    floor = 0.0
    for rank, index in enumerate(indices):
        floor = max(floor, min(1.0, (len(ps) - rank) * ps[index]))
        adjusted[index] = floor
    return adjusted


def analyze(frame, group_col, control, metric="hv"):
    pivot = frame.pivot(index="seed", columns=group_col, values=metric)
    rows = []
    for candidate in pivot.columns:
        if candidate == control:
            continue
        pair = pivot[[control, candidate]].dropna()
        diff = pair[control].to_numpy() - pair[candidate].to_numpy()
        lo, hi = bootstrap_ci(diff)
        wins, losses, p = sign_p(diff)
        rows.append({"control": control, "candidate": candidate, "n": len(diff), "mean_diff": float(diff.mean()),
                     "median_diff": float(np.median(diff)), "ci_lo": float(lo), "ci_hi": float(hi),
                     "wins": wins, "losses": losses, "sign_p": p})
    adjusted = holm([row["sign_p"] for row in rows])
    for row, p in zip(rows, adjusted):
        row["holm_p"] = p
    return rows


def main():
    static = pd.read_csv(DATA / "static_runs.csv")
    ablation = pd.read_csv(DATA / "ablation_runs.csv")
    rows = ([{"batch": "main", **row} for row in analyze(static, "method", "proposed")]
            + [{"batch": "ablation", **row} for row in analyze(ablation, "variant", "full")])
    with (OUT / "paired_effects.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6g}"))


if __name__ == "__main__":
    main()
