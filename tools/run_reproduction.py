"""Fresh resumable reproduction; writes only inside this new code project."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DEPS = ROOT.parent / "04_修订与核查/工具/python_deps"
FROZEN = ROOT / "frozen/replication/results_v3"
CURRENT = ROOT / "replication/results_v3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate():
    rows = [json.loads(p.read_text("utf8")) for p in (FROZEN / "completed").glob("*.json")]
    times = [r["runtime_s"] for r in rows]
    result = dict(runs=len(rows), archived_task_cpu_hours=sum(times) / 3600,
                  ideal_eight_worker_hours=sum(times) / 28800,
                  median_runtime_s=statistics.median(times), max_runtime_s=max(times),
                  source="frozen recorded runtime_s; estimate only, not algorithm efficiency evidence")
    (ROOT / "reports/reproduction-runtime-estimate.json").write_text(json.dumps(result, indent=2), "utf8")
    print(json.dumps(result, indent=2), flush=True)


def launch(kind, workers):
    env = dict(os.environ)
    if PYTHON_DEPS.exists():
        env["PYTHONPATH"] = str(PYTHON_DEPS)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    script = "run_upgrade_confirm.py" if kind == "confirm" else "run_exact_oracle.py"
    args = [sys.executable, "-B", "-X", "utf8", str(ROOT / "replication" / script)]
    if kind == "confirm":
        args += ["--group", "all", "--workers", str(workers)]
    log = ROOT / "logs" / f"reproduction-{kind}.log"
    record = dict(started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  command=args, python=sys.version, executable=sys.executable, platform=platform.platform(),
                  workers=workers if kind == "confirm" else 1, thread_limit=1,
                  original_completed_count=len(list((CURRENT / "completed").glob("*.json"))),
                  protocol_sha256=sha(ROOT / "plan/upgrade-experiment-lock.md"),
                  input_bounds_sha256=sha(CURRENT / "analytic-bounds.json"))
    record_path = ROOT / "reports" / f"reproduction-{kind}-process.json"
    print(json.dumps(record, ensure_ascii=True), flush=True)
    with log.open("a", encoding="utf8") as output:
        output.write("\n" + json.dumps(record, ensure_ascii=True) + "\n")
        output.flush()
        process = subprocess.Popen(args, cwd=ROOT / "replication", env=env, stdout=output, stderr=subprocess.STDOUT)
        record["pid"] = process.pid
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), "utf8")
        returncode = process.wait()
    record.update(returncode=returncode, finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  final_completed_count=len(list((CURRENT / "completed").glob("*.json"))))
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), "utf8")
    print(json.dumps(record, ensure_ascii=True), flush=True)
    if returncode:
        raise SystemExit(returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("estimate", "confirm", "oracle"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "estimate":
        estimate()
    else:
        assert 1 <= args.workers <= 14
        launch(args.mode, args.workers)


if __name__ == "__main__":
    main()
