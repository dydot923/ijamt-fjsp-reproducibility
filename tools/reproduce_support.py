"""Fresh re-evaluation of manuscript Tables 11-12 and their fixed inputs."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ROOT.parent / "04_修订与核查/工具/python_deps"
if DEPENDENCIES.exists():
    sys.path.insert(0, str(DEPENDENCIES))
sys.path.insert(0, str(ROOT / "replication"))
import run_budget_study as study
import run_matched_study as matched
from fjsp.dataset_io import load_any_dataset

OUT = ROOT / "replication/results_support"
BASE = ROOT / "frozen/replication/results_v2"


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf8")
    temporary.replace(path)


def canonical(row):
    if isinstance(row, dict):
        return {k: canonical(v) for k, v in row.items() if k not in ("runtime_s", "elapsed")}
    if isinstance(row, list):
        return [canonical(v) for v in row]
    return row


def factorial_one(task):
    original_data = study.make_data

    def data(instance):
        if instance in ("CF0", "CFM"):
            result = load_any_dataset(study.DATA / "v42_dataset.json")
            if instance == "CFM":
                for machine in ("M01", "M04", "M08"):
                    result.machines[machine].maintenance_windows = [(100., 135.), (210., 245.)]
            return result
        return original_data(instance)

    study.make_data = data
    study.instrument_neighbor = matched.instrument_matched
    try:
        result = study.run_one(task)
    finally:
        study.make_data = original_data
    assert result["invalid_schedules"] == 0
    save(OUT / "completed" / f"{task[0]}_{task[2]}.json", result)
    return {"instance": task[0], "seed": task[2], "seconds": result["runtime_s"]}


def single_step(rebuilt=False):
    import run_single_step as single
    single.OUT = OUT
    # These 10 parent encodings are fixed inputs to the diagnostic, not fresh
    # search runs. Every one of the 1,200 candidate proposals is decoded anew.
    if rebuilt:
        parents = [json.loads(p.read_text("utf8")) for p in sorted((OUT / "parent-completed").glob("*.json"))]
    else:
        all_parents = json.loads((BASE / "main_runs.json").read_text("utf8"))
        parents = [r for r in all_parents if r["method"] == "H-NSGA" and r["seed"] <= 410010]
    assert len(parents) == 10
    save(OUT / "main_runs.json", parents)
    single.main()
    with (OUT / "single_step.csv").open(encoding="utf-8-sig", newline="") as f:
        fresh = list(csv.DictReader(f))
    with (BASE / "single_step.csv").open(encoding="utf-8-sig", newline="") as f:
        archived = list(csv.DictReader(f))
    assert fresh == archived
    assert len(fresh) == 1200
    report = {"fresh_candidate_decodes": len(fresh), "fixed_parent_inputs": len(parents),
              "all_fields_equal_to_archived_candidates": True, "parent_searches_freshly_reproduced": rebuilt,
              "input_parent_file_sha256": hashlib.sha256((OUT / "main_runs.json").read_bytes()).hexdigest()}
    save(ROOT / "reports/support-single-step.json", report)
    print(json.dumps(report), flush=True)


def exploratory_one(task):
    group, record_task = task
    if group == "parents":
        result = study.run_one(record_task)
        path = OUT / "parent-completed" / f"E0_{record_task[2]}.json"
    else:
        import run_local_budget
        result = run_local_budget.run_alias(record_task)
        path = OUT / "tuning-completed" / f"{record_task[1]}_{record_task[2]}.json"
    assert result["invalid_schedules"] == 0
    save(path, result)
    return {"group": group, "instance": record_task[0], "method": record_task[1],
            "seed": record_task[2], "seconds": result["runtime_s"]}


def exploratory(group, workers):
    destination = OUT / ("parent-completed" if group == "parents" else "tuning-completed")
    destination.mkdir(exist_ok=True)
    if group == "parents":
        tasks = [("E0", "H-NSGA", s, 4000) for s in range(410001, 410011)]
        previous = [r for r in json.loads((BASE / "main_runs.json").read_text("utf8"))
                    if r["method"] == "H-NSGA" and r["seed"] <= 410010]
        name = lambda task: f"E0_{task[2]}.json"
    else:
        tasks = [("E0T", f"H-MVNS-L{t}", s, 4000) for t in (1, 2, 4, 6) for s in range(440001, 440007)]
        previous = json.loads((BASE / "local_budget_runs.json").read_text("utf8"))
        name = lambda task: f"{task[1]}_{task[2]}.json"
    missing = [t for t in tasks if not (destination / name(t)).exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(exploratory_one, (group, t)) for t in missing]
        for i, future in enumerate(as_completed(futures), len(tasks) - len(missing) + 1):
            print(json.dumps({"completed": i, "total": len(tasks), **future.result()}), flush=True)
    old = {(r["method"], r["seed"]): r for r in previous}
    rows = [json.loads((destination / name(t)).read_text("utf8")) for t in tasks]
    differences = [{"method": r["method"], "seed": r["seed"]} for r in rows
                   if canonical(r) != canonical(old[(r["method"], r["seed"])])]
    if group == "parents":
        save(OUT / "main_runs.json", rows)
    else:
        save(OUT / "local_budget_runs.json", rows)
    report = {"fresh_runs": len(rows), "new_encoding_evaluations": sum(r["evaluations"] for r in rows),
              "invalid_schedules": sum(r["invalid_schedules"] for r in rows), "deterministic_differences": differences}
    save(ROOT / f"reports/support-{group}.json", report)
    print(json.dumps(report), flush=True)
    assert not differences
    if group == "parents":
        single_step(rebuilt=True)


def factorial(workers):
    tasks = [(instance, "H-MVNS", seed, 4000) for instance in ("CF0", "E1", "CFM", "E2")
             for seed in range(430001, 430013)]
    missing = [task for task in tasks if not (OUT / "completed" / f"{task[0]}_{task[2]}.json").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(factorial_one, task) for task in missing]
        for n, future in enumerate(as_completed(futures), len(tasks) - len(missing) + 1):
            result = future.result()
            print(json.dumps({"completed": n, "total": len(tasks), **result}), flush=True)
    archived = json.loads((BASE / "constraint_factorial_runs.json").read_text("utf8"))
    archived += [r for r in json.loads((BASE / "matched_runs.json").read_text("utf8")) if r["instance"] in ("E1", "E2")]
    old = {(r["instance"], r["seed"]): r for r in archived}
    rows = [json.loads((OUT / "completed" / f"{t[0]}_{t[2]}.json").read_text("utf8")) for t in tasks]
    differences = []
    for row in rows:
        previous = old[(row["instance"], row["seed"])]
        if canonical(row) != canonical(previous):
            differences.append({"instance": row["instance"], "seed": row["seed"],
                                "front_equal": row["front"] == previous["front"],
                                "representative_equal": row["representative"] == previous["representative"]})
    save(OUT / "factorial_runs.json", rows)
    report = {"fresh_runs": len(rows), "new_encoding_evaluations": sum(r["evaluations"] for r in rows),
              "invalid_schedules": sum(r["invalid_schedules"] for r in rows),
              "deterministic_differences": differences}
    save(ROOT / "reports/support-factorial.json", report)
    print(json.dumps(report), flush=True)
    assert not differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--mode", choices=("single", "factorial", "parents", "tuning", "all"), default="all")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / "completed").mkdir(exist_ok=True)
    if args.mode == "single":
        single_step()
    if args.mode in ("parents", "all"):
        exploratory("parents", args.workers)
    if args.mode in ("factorial", "all"):
        factorial(args.workers)
    if args.mode in ("tuning", "all"):
        exploratory("tuning", args.workers)


if __name__ == "__main__":
    main()
