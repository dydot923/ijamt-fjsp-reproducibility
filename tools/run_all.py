"""Resumable full reproduction; manuscript editing is a separate reviewed step."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--derived-only", action="store_true", help="Rebuild metrics, audits and figures from completed fresh runs")
    args = parser.parse_args()
    assert 1 <= args.workers <= 14
    environment = dict(os.environ)
    existing = ROOT.parent / "04_修订与核查/工具/python_deps"
    if existing.exists():
        environment["PYTHONPATH"] = str(existing)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    jobs = []
    if not args.derived_only:
        jobs += [
            ["tools/run_reproduction.py", "confirm", "--workers", str(args.workers)],
            ["tools/run_reproduction.py", "oracle"],
            ["tools/reproduce_support.py", "--mode", "all", "--workers", str(min(args.workers, 3))],
            ["tools/theory_audit.py"], ["tools/reproduce_integration.py"],
            ["qa/theory-upgrade-evidence/verify_theory.py"],
            # This phase is deliberately serial and follows all parallel jobs.
            ["replication/time_prefix_component.py"],
        ]
    jobs += [["tools/run_reproduction_compare.py", "--require-complete"],
             ["tools/run_reproduction_bounds_check.py"], ["tools/run_reproduction_provenance.py"],
             ["replication/compute_capacity_envelope.py"],
             ["tools/statistical_audit.py", "--source", "fresh"], ["tools/rebuild_artifacts.py"]]
    records = []
    for index, arguments in enumerate(jobs, 1):
        command = [sys.executable, "-B", "-X", "utf8", str(ROOT / arguments[0]), *arguments[1:]]
        print(f"[{index}/{len(jobs)}] {' '.join(arguments)}", flush=True)
        log = ROOT / "logs" / f"pipeline-{index:02d}-{Path(arguments[0]).stem}.log"
        started = time.time()
        with log.open("w", encoding="utf8") as output:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT)
        records.append({"command": arguments, "returncode": result.returncode, "elapsed_s": time.time() - started, "log": str(log.relative_to(ROOT))})
        (ROOT / "reports/pipeline-status.json").write_text(json.dumps(records, indent=2), "utf8")
        if result.returncode:
            print(f"Failed; inspect {log}", flush=True)
            raise SystemExit(result.returncode)
    print("PASS: completed reproduction outputs, numerical audits, tables and figures", flush=True)


if __name__ == "__main__":
    main()
