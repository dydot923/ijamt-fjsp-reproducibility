"""Summarize the new controlled stress experiment without cross-scenario HV comparisons."""

from pathlib import Path
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "replication" / "results" / "stress_runs.csv"
OUT = DATA.parent


def ci(values):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(43120)
    estimates = rng.choice(x, size=(20000, len(x)), replace=True).mean(axis=1)
    return np.quantile(estimates, [.025, .975])


def sign_p(values):
    x = np.asarray(values, dtype=float)
    wins = int(sum(x > 1e-12)); losses = int(sum(x < -1e-12))
    n = wins + losses
    p = min(1., 2. * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.
    return wins, losses, p


def main():
    frame = pd.read_csv(DATA)
    assert len(frame) == 72 and frame.status.eq("ok").all()
    assert frame.invalid_schedules.eq(0).all()
    summary = frame.groupby(["scenario", "method"]).agg(
        n=("hv", "size"), hv_mean=("hv", "mean"), hv_sd=("hv", "std"),
        time_mean=("runtime_s", "mean"), cmax_mean=("best_cmax", "mean"),
        lmax_mean=("best_lmax", "mean"), audit_count=("audited_schedules", "sum")
    ).reset_index()
    summary.to_csv(OUT / "stress_summary.csv", index=False, encoding="utf-8-sig")
    lines = []
    for scenario in ("S0", "S1", "S2"):
        pivot = frame[frame.scenario == scenario].pivot(index="seed", columns="method", values="hv")
        d = pivot.proposed - pivot.shared_nsga
        lo, hi = ci(d)
        wins, losses, p = sign_p(d)
        lines.append({"scenario": scenario, "mean_hv_delta": d.mean(), "bootstrap_lo": lo, "bootstrap_hi": hi,
                      "wins": wins, "losses": losses, "sign_p": p})
    pd.DataFrame(lines).to_csv(OUT / "stress_paired.csv", index=False, encoding="utf-8-sig")
    for method in ("proposed", "shared_nsga"):
        p = frame[frame.method == method].pivot(index="seed", columns="scenario", values="best_cmax")
        print(method, "S1-S0 Cmax", (p.S1-p.S0).mean(), "S2-S0 Cmax", (p.S2-p.S0).mean())
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(lines).to_string(index=False, float_format=lambda v: f"{v:.5f}"))


if __name__ == "__main__": main()
