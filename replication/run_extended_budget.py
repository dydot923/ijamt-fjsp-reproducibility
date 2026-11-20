"""Exploratory extended-generation comparison on S0, with independently audited outputs."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fjsp.baselines import run_standard_nsga_ii
from fjsp.config import AlgorithmConfig
from fjsp.dataset_io import load_any_dataset
from fjsp.objectives import approximate_hypervolume
from run_stress import DATA, OUT, SEEDS, check_schedule, make_scenario


def one(seed):
    data = make_scenario(load_any_dataset(DATA), "S0")
    cfg = AlgorithmConfig(seed=seed, generations=90, population_size=40, verbose=False)
    started = time.perf_counter()
    front = run_standard_nsga_ii(data, cfg)
    runtime = time.perf_counter() - started
    issues = [check_schedule(data, item.schedule, item.objectives) for item in front]
    return {"seed": seed, "method": "shared_nsga_90", "runtime_s": runtime,
            "front_size": len(front), "audited_schedules": len(issues),
            "invalid_schedules": sum(bool(x) for x in issues),
            "front": [[float(v) for v in item.objectives] for item in front]}


def main():
    previous = json.loads((OUT / "stress_runs.json").read_text(encoding="utf-8"))
    base = [r for r in previous if r["scenario"] == "S0" and r["method"] in {"proposed", "shared_nsga"}]
    extra = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(one, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            item = future.result(); extra.append(item)
            print(item["seed"], item["runtime_s"], item["invalid_schedules"], flush=True)
    assert len(extra) == 12 and not any(r["invalid_schedules"] for r in extra)
    all_rows = base + extra
    points = [p for r in all_rows for p in r["front"]]
    ideal = [min(p[i] for p in points) for i in range(3)]
    nadir = [max(p[i] for p in points) for i in range(3)]
    reference = [nadir[i] + max(1.0, .2 * (nadir[i]-ideal[i]), .05 * abs(nadir[i])) for i in range(3)]
    for r in all_rows:
        r["hv_extended_ref"] = approximate_hypervolume(r["front"], ideal, reference)
    (OUT / "extended_budget_runs.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "extended_budget_bounds.json").write_text(json.dumps({"ideal": ideal, "reference": reference}, indent=2), encoding="utf-8")
    df = pd.DataFrame([{k:v for k,v in r.items() if k != "front"} for r in all_rows])
    df.to_csv(OUT / "extended_budget_runs.csv", index=False, encoding="utf-8-sig")
    print(df.groupby("method")[["runtime_s", "hv_extended_ref", "audited_schedules", "invalid_schedules"]].agg(["mean", "std"]).round(5))


if __name__ == "__main__": main()
