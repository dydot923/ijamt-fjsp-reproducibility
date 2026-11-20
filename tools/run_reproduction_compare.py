"""Compare fresh deterministic run evidence with frozen original outputs."""
from pathlib import Path
import argparse
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "frozen/replication/results_v3"
FRESH = ROOT / "replication/results_v3"
TIMING_KEYS = {"runtime_s", "elapsed", "full_ns", "prefix_ns"}


def compare(a, b, path="", differences=None):
    if differences is None:
        differences = []
    if isinstance(a, dict) and isinstance(b, dict):
        ak, bk = set(a) - TIMING_KEYS, set(b) - TIMING_KEYS
        if ak != bk:
            differences.append({"path": path, "kind": "keys", "frozen": sorted(ak), "fresh": sorted(bk)})
        for k in sorted(ak & bk):
            compare(a[k], b[k], path + "/" + k, differences)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            differences.append({"path": path, "kind": "length", "frozen": len(a), "fresh": len(b)})
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + "/" + str(i), differences)
    elif a != b:
        numeric = isinstance(a, (int, float)) and isinstance(b, (int, float))
        close = numeric and math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-9)
        differences.append({"path": path, "kind": "numerical_roundoff" if close else "value", "frozen": a, "fresh": b})
    return differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    original = {p.name: p for p in (FROZEN / "completed").glob("*.json")}
    current = {p.name: p for p in (FRESH / "completed").glob("*.json")}
    assert len(original) == 678
    assert current.keys() <= original.keys(), current.keys() - original.keys()
    rows = []
    for name in sorted(current):
        a = json.loads(original[name].read_text("utf8"))
        b = json.loads(current[name].read_text("utf8"))
        differences = compare(a, b)
        rows.append({"file": name, "exact_deterministic_match": not differences,
                     "differences": differences[:30], "total_difference_count": len(differences),
                     "fresh_evaluations": b["evaluations"], "fresh_invalid_schedules": b["invalid_schedules"],
                     "fresh_audited_schedules": b["audited_schedules"],
                     "fresh_runtime_s": b["runtime_s"], "frozen_runtime_s": a["runtime_s"]})
    result = dict(expected_runs=678, completed=len(current), missing=sorted(original.keys() - current.keys()),
                  exact_deterministic_matches=sum(r["exact_deterministic_match"] for r in rows),
                  evaluations=sum(r["fresh_evaluations"] for r in rows),
                  invalid_schedules=sum(r["fresh_invalid_schedules"] for r in rows),
                  audited_terminal_schedules=sum(r["fresh_audited_schedules"] for r in rows),
                  timing_fields_excluded=sorted(TIMING_KEYS), runs=rows)
    (ROOT / "reports/reproduction-confirm-comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("runs", "missing")}, ensure_ascii=True))
    if args.require_complete:
        assert len(current) == 678 and result["exact_deterministic_matches"] == 678
        assert result["evaluations"] == 3108000 and result["invalid_schedules"] == 0
    oracle = FRESH / "oracle"
    if (oracle / "runs.json").exists():
        orows = []
        for name in ("design.json", "summary.json", "runs.json"):
            a = json.loads((FROZEN / "oracle" / name).read_text("utf8"))
            b = json.loads((oracle / name).read_text("utf8"))
            differences = compare(a, b)
            orows.append({"file": name, "exact_deterministic_match": not differences,
                          "differences": differences[:30], "total_difference_count": len(differences)})
        oracle_report = dict(files=orows, fresh_runs=len(json.loads((oracle / "runs.json").read_text("utf8"))))
        (ROOT / "reports/reproduction-oracle-comparison.json").write_text(json.dumps(oracle_report, indent=2), "utf8")
        print(json.dumps({"oracle_runs": oracle_report["fresh_runs"],
                          "files_exact_match": [r["exact_deterministic_match"] for r in orows]}))
        if args.require_complete:
            assert oracle_report["fresh_runs"] == 144 and all(r["exact_deterministic_match"] for r in orows)


if __name__ == "__main__":
    main()
