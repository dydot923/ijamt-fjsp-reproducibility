# Running and reproducing the study

## Software environment

Use Python3.12. `requirements.txt` pins the scientific versions recorded in the verified environment. Install `requirements-reproduction.txt` for the optional full figure/table orchestration, which also imports python-docx. The quick demonstration and tests use only the scientific requirements. No Microsoft Word, original drive letter or author-specific installation folder is required for them.

## 1 Run the independent example

```powershell
.\.venv\Scripts\python.exe -B scripts/run_demo.py
.\.venv\Scripts\python.exe -B scripts/run_demo.py --instance T4 --method H-NSGA --seed 530001
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The demonstration executes the solver; it does not print a fabricated result. It enumerates the independent machine-order DAG front and the decoder-reachable front, checks schedule feasibility and tests deterministic fields against the archived small-instance run. Runtime is excluded from deterministic comparison.

The tests additionally verify all six constructed input generators and exact fronts,25machine-move prefix/full schedule comparisons, and all three study methods on T1 with200new evaluations each. Their purpose is to check packaging and computation, not to claim new performance evidence.

## 2 Restore additional local data when available

For the full scientific project, first make a separate working copy of this repository. Obtain `ijamt-fjsp-data-results-v0.1.0.zip` from the matching versioned GitHub Release once published, or use the locally prepared companion asset. Extract its **contents into the root** of that copy. The restored tree includes:

```text
replication/data/v42_dataset.json
replication/data/benchmarks/Mk01.json ... Mk10.json
replication/results_v3/analytic-bounds.json
replication/results_v3/completed/        # independently repeated confirmation records
replication/results_v3/oracle/
replication/results_support/
replication/results_integration/
replication/results_20261029_capacity/
replication/results_v2/                 # original exploratory records
frozen/replication/                     # original comparison implementation and records
tables/Table01.csv ... Table13.csv
```

The original code-only manifest does not cover the companion asset; verify its `DATA_MANIFEST.json` separately. These additional inputs and results are ignored by Git to keep the source tree small. Public redistribution was confirmed for this prepared release; publish the ZIP as a versioned data asset alongside the code.

## 3 Recompute analyses from the archived runs

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-reproduction.txt
.\.venv\Scripts\python.exe -B -X utf8 tools/run_all.py --derived-only
```

This compares existing repeated records with the frozen originals, checks numerical bounds, recomputes the capacity envelope and statistics, and rebuilds tables/figures. It does not start a new678-run confirmation experiment. Generated reports are written under `reports/` and logs under `logs/`.

Font availability affects figure lettering. The numerical verification does not require the author's Times New Roman/Chinese fonts. The plotting code can fall back to installed fonts. This repository does not bundle proprietary font files.

## 4 Execute a fresh repetition

The orchestration entry is:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 tools/run_all.py --workers 8
```

The driver resumes completed tasks. With the full bundle already restored,678completed records are present, so the main search stage will skip them. To start a genuinely fresh repetition, use a separate working copy and preserve the restored results elsewhere in that copy before running. In particular, retain `replication/results_v3/analytic-bounds.json` while moving the existing `completed/`, `oracle/`, run-design/result summaries, support and integration outputs aside. Do not remove `frozen/`, which is comparison evidence. A fresh run has substantial compute cost; the recorded archived durations provide only a rough estimate for another CPU.

For an isolated main confirmation run after preparing an empty current output area:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 tools/run_reproduction.py confirm --workers 8
.\.venv\Scripts\python.exe -B -X utf8 tools/run_reproduction_compare.py --require-complete
```

The main grid has678runs and3,108,000new encoding evaluations. Serial component timing follows parallel searches; do not run the timing stage concurrently with other solver jobs. A same-seed repetition checks deterministic implementation consistency and does not double the statistical sample size.

## Paper mapping

| Evidence | Entry |
|---|---|
| Conservative full decoder | `replication/fjsp/decoder.py` |
| Objective-specific candidate generation | `replication/run_matched_study.py`, `replication/fjsp/nsga_vns.py` |
| Exact common-prefix reuse | `replication/prefix_evaluator.py` |
| Six independent exact fronts | `replication/run_exact_oracle.py` |
| Main independent-budget grid | `replication/run_upgrade_confirm.py` |
| Eligibility/energy capacity LP | `replication/compute_capacity_envelope.py`, `tools/theory_audit.py` |
| Prespecified and sensitivity statistics | `tools/statistical_audit.py` |
| Serial paired component timing | `replication/time_prefix_component.py` |

The publication copy retains the verified scientific source bytes. Packaging, documentation and the small example/test entry points are new. It does not include manuscript-editing tools, author declarations, private document backups or agent task transcripts.
