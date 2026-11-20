"""Re-run controlled setup/maintenance stress experiments with an audited decoder.

The engineering dataset has zero setup times and no machine maintenance windows.
S1 and S2 are explicit synthetic perturbations, not field measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from fjsp.baselines import run_standard_nsga_ii
from fjsp.config import AlgorithmConfig
from fjsp.dataset_io import load_any_dataset
from fjsp.nsga_vns import IGA_NSGA_VNS
from fjsp.objectives import approximate_hypervolume

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "replication" / "data" / "v42_dataset.json"
OUT = ROOT / "replication" / "results"
METHODS = ("proposed", "shared_nsga")
SCENARIOS = ("S0", "S1", "S2")
SEEDS = tuple(range(310001, 310013))


def make_scenario(data, scenario):
    if scenario in {"S1", "S2"}:
        for operation in data.operations.values():
            for option in operation.candidates:
                machine_number = int(option.machine_id[1:])
                option.setup_time = round(4.0 + 0.08 * option.processing_time + 0.5 * (machine_number % 3), 1)
    if scenario == "S2":
        for machine_id in ("M01", "M04", "M08"):
            data.machines[machine_id].maintenance_windows = [(100.0, 135.0), (210.0, 245.0)]
    return data


def check_schedule(data, schedule, claimed):
    violations = []
    entries = schedule.entries
    if set(entries) != set(data.operation_ids):
        violations.append("operation_set")
    for operation_id, entry in entries.items():
        try:
            option = data.option(operation_id, entry.machine_id)
        except KeyError:
            violations.append("machine_eligibility")
            continue
        if abs(entry.processing_time - option.processing_time) > 1e-7 or abs(entry.end - entry.start - option.processing_time) > 1e-7:
            violations.append("processing_duration")
        if abs(entry.energy - option.total_energy) > 1e-7:
            violations.append("energy")
        if abs(entry.start - entry.setup_start - entry.setup_time) > 1e-7 or entry.setup_start < -1e-7:
            violations.append("setup_interval")
        for left, right in data.machines[entry.machine_id].maintenance_windows:
            if entry.setup_start < right - 1e-7 and left < entry.end - 1e-7:
                violations.append("maintenance")
    for job in data.jobs.values():
        previous = None
        for operation in job.operations:
            entry = entries.get(operation.operation_id)
            if entry is None:
                continue
            if entry.setup_start + 1e-7 < (previous.end if previous else job.release_time):
                violations.append("precedence")
            previous = entry
    for machine_id in data.machine_ids:
        previous = None
        for entry in schedule.entries_for_machine(machine_id):
            if previous and previous.end > entry.setup_start + 1e-7:
                violations.append("machine_overlap")
            option = data.option(entry.operation_id, entry.machine_id)
            expected_setup = 0.0 if previous and data.jobs[previous.job_id].family == data.jobs[entry.job_id].family else option.setup_time
            if abs(entry.setup_time - expected_setup) > 1e-7:
                violations.append("sequence_setup")
            previous = entry
    recalc = (
        max((entry.end for entry in entries.values()), default=0.0),
        sum(entry.energy for entry in entries.values()),
        max((sum(entry.end - entry.setup_start for entry in schedule.entries_for_machine(machine_id)) for machine_id in data.machine_ids), default=0.0),
    )
    if any(abs(a - b) > 1e-6 for a, b in zip(recalc, claimed)):
        violations.append("objective_mismatch")
    return sorted(set(violations))


def run_one(item):
    scenario, method, seed = item
    data = make_scenario(load_any_dataset(DATA), scenario)
    config = AlgorithmConfig(seed=seed, verbose=False)
    start = time.perf_counter()
    try:
        if method == "proposed":
            solver = IGA_NSGA_VNS(data, config)
            front = solver.run()
            evaluations = solver.evaluation_count
        else:
            front = run_standard_nsga_ii(data, config)
            evaluations = None
        elapsed = time.perf_counter() - start
        issues = [check_schedule(data, item.schedule, item.objectives) for item in front]
        return {
            "scenario": scenario, "method": method, "seed": seed, "runtime_s": elapsed,
            "evaluations": evaluations, "front_size": len(front),
            "audited_schedules": len(issues), "invalid_schedules": sum(bool(x) for x in issues),
            "violation_types": sorted({x for row in issues for x in row}),
            "front": [[float(v) for v in item.objectives] for item in front],
            "status": "ok",
        }
    except Exception as exc:
        return {"scenario": scenario, "method": method, "seed": seed, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def summarize(rows):
    for scenario in SCENARIOS:
        subset = [row for row in rows if row["scenario"] == scenario and row["status"] == "ok"]
        points = [p for row in subset for p in row["front"]]
        if not points:
            continue
        ideal = [min(p[i] for p in points) for i in range(3)]
        nadir = [max(p[i] for p in points) for i in range(3)]
        reference = [nadir[i] + max(1.0, 0.2 * (nadir[i] - ideal[i]), 0.05 * abs(nadir[i])) for i in range(3)]
        for row in subset:
            row["hv"] = approximate_hypervolume(row["front"], ideal, reference)
            for i, name in enumerate(("cmax", "tec", "lmax")):
                row[f"best_{name}"] = min((p[i] for p in row["front"]), default=math.nan)
        (OUT / f"bounds_{scenario}.json").write_text(json.dumps({"ideal": ideal, "reference": reference}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(scenario, method, seed) for scenario in SCENARIOS for method in METHODS for seed in SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(row["scenario"], row["method"], row["seed"], row["status"], row.get("invalid_schedules", "-"), flush=True)
            (OUT / "stress_runs_checkpoint.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    summarize(rows)
    (OUT / "stress_runs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["scenario", "method", "seed", "status", "runtime_s", "evaluations", "front_size", "audited_schedules", "invalid_schedules", "violation_types", "hv", "best_cmax", "best_tec", "best_lmax"]
    with (OUT / "stress_runs.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda x: (x["scenario"], x["method"], x["seed"])):
            item = dict(row)
            item["violation_types"] = ";".join(item.get("violation_types", []))
            writer.writerow(item)


if __name__ == "__main__":
    main()
