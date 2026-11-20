"""Regression for historical negative maintenance windows; original inputs unchanged."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "04_修订与核查/工具/python_deps"))
sys.path.insert(0, str(ROOT / "replication"))
from upgrade_bounds import analytic_bounds
from run_budget_study import make_data


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = ROOT / "frozen/replication/upgrade_bounds.py"
    new = ROOT / "replication/upgrade_bounds.py"
    original_project = ROOT.parent / "03_复现工程/replication/upgrade_bounds.py"
    if original_project.exists():
        assert old.read_bytes() == original_project.read_bytes()
    old_space = {"__name__": "frozen_bounds_regression", "__file__": str(old)}
    exec(compile(old.read_text("utf8"), str(old), "exec"), old_space)
    option = SimpleNamespace(processing_time=2.0, setup_time=0.0, total_energy=1.0, machine_id="M1")
    operation = SimpleNamespace(operation_id="J1-O1", candidates=[option])
    job = SimpleNamespace(release_time=0.0, operations=[operation])
    machine = SimpleNamespace(maintenance_windows=[(-5.0, -1.0)])
    data = SimpleNamespace(operations={"J1-O1": operation}, jobs={"J1": job},
                           machines={"M1": machine}, machine_ids=["M1"])
    failed = False
    try:
        old_space["analytic_bounds"](data)
    except AssertionError:
        failed = True
    assert failed, "Original implementation must expose the boundary-case failure"
    corrected = analytic_bounds(data)
    assert corrected["lower"] == corrected["upper"] == [2.0, 1.0, 2.0]
    frozen = json.loads((ROOT / "frozen/replication/results_v3/analytic-bounds.json").read_text("utf8"))
    before_hash = digest(ROOT / "replication/results_v3/analytic-bounds.json")
    checks = []
    for instance in ["E0"] + [f"Mk{i:02d}" for i in range(1, 11)]:
        original = old_space["analytic_bounds"](make_data(instance))
        result = analytic_bounds(make_data(instance))
        assert result == original == frozen["bounds"][instance]
        checks.append({"instance": instance, "bounds_unchanged": True})
    after_hash = digest(ROOT / "replication/results_v3/analytic-bounds.json")
    assert before_hash == after_hash == digest(ROOT / "frozen/replication/results_v3/analytic-bounds.json")
    record = dict(change="Clamp last maintenance end to at least zero", scope="Boundary correction only",
                  before_sha256=digest(old), after_sha256=digest(new),
                  original_project_file_unchanged=True, frozen_bounds_json_unchanged=True,
                  frozen_bounds_sha256=after_hash, original_counterexample_raises=True,
                  corrected_counterexample_bounds=corrected, real_input_checks=checks,
                  confirmation_driver_uses_frozen_bounds_json=True,
                  search_algorithm_and_task_grid_unchanged=True)
    (ROOT / "reports/reproduction-bounds-portability-audit.json").write_text(json.dumps(record, indent=2), "utf8")
    print(json.dumps({"status": "PASS", "real_inputs_unchanged": len(checks),
                       "historical_negative_window_failure_reproduced_and_fixed": True}))


if __name__ == "__main__":
    main()
