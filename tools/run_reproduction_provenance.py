"""Check new-project source/input identity against the immutable originals."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "03_复现工程/replication"
if not SOURCE.exists():
    SOURCE = ROOT / "frozen/replication"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = []
    current = ROOT / "replication"
    for p in sorted(current.glob("*.py")) + sorted((current / "fjsp").glob("*.py")) + sorted((current / "data").rglob("*.json")):
        relative = p.relative_to(current)
        baseline = SOURCE / relative
        assert baseline.exists(), relative
        row = {"file": relative.as_posix(), "original_sha256": digest(baseline), "current_sha256": digest(p)}
        row["identical"] = row["original_sha256"] == row["current_sha256"]
        records.append(row)
    differences = [r["file"] for r in records if not r["identical"]]
    # This is an acknowledged analytical-bound edge-case correction, not a
    # search change; the frozen bounds JSON remains untouched in actual runs.
    expected = {"upgrade_bounds.py"}
    if (ROOT / "reports/reproduction-legacy-portability-audit.json").exists():
        expected.add("run_budget_study.py")
    assert set(differences) == expected, differences
    def search_body(path):
        tree = ast.parse(path.read_text("utf8"))
        tree.body = [node for node in tree.body if not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ("convert_benchmarks", "main"))]
        return ast.dump(tree, include_attributes=False)
    assert search_body(current / "run_budget_study.py") == search_body(SOURCE / "run_budget_study.py")
    design = json.loads((current / "results_v3/run-design.json").read_text("utf8"))
    original = json.loads((ROOT / "frozen/replication/results_v3/run-design.json").read_text("utf8"))
    for key in ("tasks", "protocol_sha256", "script_sha256", "reference_survival_sha256", "evaluator_sha256", "bounds_sha256"):
        assert design[key] == original[key], key
    record = dict(source_files_checked=len(records), differences=differences, records=records,
                  confirmatory_grid_and_driver_evaluator_survival_hashes_identical=True,
                  run_budget_study_search_ast_identical=True,
                  note="Frozen normalization JSON used for all runs. Worker count/timestamps are execution identity only.")
    (ROOT / "reports/reproduction-source-provenance.json").write_text(json.dumps(record, indent=2), "utf8")
    print(json.dumps({"status": "PASS", "files_checked": len(records), "differences": differences}))


if __name__ == "__main__":
    main()
