"""Paired whole-search regression: full evaluation versus exact prefix reuse."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT.parent / "04_修订与核查/工具/python_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "replication"))
import run_upgrade_confirm as driver
from fjsp.nsga_vns import _evaluate


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items() if k not in ("runtime_s", "elapsed", "decoder_counts")}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value


def full_factory(counters):
    def evaluate(individual, data, context, baseline):
        counters["full_calls"] += 1
        counters["reinserted_operations"] += len(data.operations)
        return _evaluate(individual, data, context, baseline)
    return evaluate


def main():
    saved_factory = driver.make_evaluator
    saved_out = driver.OUT
    reports = []
    try:
        for instance in ("E0", "E1", "E2", "Mk01", "Mk10"):
            for method in ("H-NSGA", "H-MVNS-2", "RP-NSGA"):
                task = (instance, method, 771001, 200)
                results = []
                for mode, factory in (("full", full_factory), ("prefix", saved_factory)):
                    output = ROOT / "replication/results_integration" / mode
                    (output / "completed").mkdir(parents=True, exist_ok=True)
                    driver.OUT = output
                    driver.make_evaluator = factory
                    driver.one(task)
                    path = output / "completed" / f"{instance}_{method}_771001_200.json"
                    results.append(json.loads(path.read_text("utf8")))
                assert canonical(results[0]) == canonical(results[1]), (instance, method)
                reports.append({"instance": instance, "method": method, "seed": 771001,
                                "budget": 200, "identical_search_outputs": True,
                                "full_counts": results[0]["decoder_counts"], "prefix_counts": results[1]["decoder_counts"]})
                print(json.dumps(reports[-1]), flush=True)
    finally:
        driver.make_evaluator = saved_factory
        driver.OUT = saved_out
    report = {"status": "PASS", "paired_trajectories": len(reports), "fresh_search_runs": 2 * len(reports),
              "new_encoding_evaluations": len(reports) * 400,
              "purpose": "Evaluator equivalence diagnostic; excluded from confirmatory quality sample sizes",
              "cases": reports, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / "reports/integration-reproduction.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf8")
    print(json.dumps({"status": "PASS", "paired_trajectories": 15, "evaluations": 6000}), flush=True)


if __name__ == "__main__":
    main()
