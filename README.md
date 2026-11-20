# Three objective flexible job shop scheduling

Research code for **Objective-matched neighborhoods and exact prefix evaluation for three-objective flexible job-shop scheduling**, by Yuyang Du and Yongbin Zhang, Beijing Institute of Graphic Communication.

This repository contains the scheduling implementation, exact common-prefix evaluator, experimental drivers, independent small-instance enumeration, and six constructed validation instances. A runnable example checks a fresh 200-evaluation search against its archived small-instance result. The associated manuscript is being prepared for submission; no journal acceptance, article DOI or public data DOI is asserted here.

[中文说明](README.zh-CN.md) · [Reproduction guide](docs/REPRODUCTION.md) · [Data scope](docs/DATA.md) · [Publication guide](docs/PUBLISH.zh-CN.md)

The complete data/results bundle is distributed as **`ijamt-fjsp-data-results-v0.1.0.zip`** with the [matching v0.1.0 release](https://github.com/dydot923/ijamt-fjsp-reproducibility/releases/tag/v0.1.0). It contains the engineering and benchmark inputs, archived numerical records and material required for the full reproduction pipeline. Obtain both the code and this data asset for complete reproduction.

## Quick start

Use Python 3.12. Run these commands from the folder containing this README.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B scripts/run_demo.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Linux or macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B scripts/run_demo.py
.venv/bin/python -B -m unittest discover -s tests -v
```

The default demonstration uses constructed case T1, H-MVNS-2, seed 530001 and exactly 200 newly evaluated encodings. It verifies schedule feasibility, enumerates the exact front without the insertion decoder, checks decoder reachability, and compares the deterministic search output with the archived reference. Output is saved to `outputs/demo-T1-H-MVNS-2-530001.json`. A demonstration is not a rerun of all manuscript experiments.

Use `python scripts/check_release.py` to check the distributed file hashes and publication scope without installing scientific dependencies.

## Repository contents

| Path | Content |
|---|---|
| `replication/fjsp/` | Data model, conservative decoder, objective functions and solver implementation |
| `replication/prefix_evaluator.py` | Exact common-reading-prefix reuse |
| `replication/run_*.py` | Budget-controlled, small-instance and supplementary experiment drivers |
| `replication/analyze_*.py` | Indicator and statistical analysis sources |
| `replication/compute_capacity_envelope.py`, `upgrade_bounds.py` | Eligibility/energy LP and analytical bounds |
| `tools/` | Independent mathematical/statistical checks and full reproduction orchestration |
| `examples/T1.json` to `T6.json` | Six constructed validation inputs; not production measurements |
| `reference/small_instances/` | Archived exact fronts and 144 small-instance search records |
| `figures/`, `tables/` | Numerical figure/table reconstruction sources |
| `plan/upgrade-experiment-lock.md` | Frozen experimental protocol used by the main driver |
| `source_provenance.json` | Hashes of the copied, previously verified scientific sources |

Generated results, large input/reference bundles, environments, caches and manuscript files are excluded by `.gitignore`. The paper's engineering input and third-party Brandimarte routing files are distributed in the **separate data asset**, not embedded in this compact code tree. The code tree alone does not reproduce the678engineering/benchmark confirmation runs. See [Data scope](docs/DATA.md) for the complete companion materials.

## Full reproduction

The original experiment scripts and independent auditors are retained. To reproduce the full study, extract `ijamt-fjsp-data-results-v0.1.0.zip` into this repository root to restore the original relative paths. Install `requirements-reproduction.txt` and follow [the full reproduction guide](docs/REPRODUCTION.md). The orchestration driver resumes existing runs; use a separate copy with empty current output directories for a genuinely fresh repetition.

## Interpretation

The model concerns complete static schedules. Energy sums the supplied values on selected processing arcs; it does not include idle, transport, start-up or maintenance energy. The Mk energy extensions are constructed, dimensionless values. Maximum machine occupancy sums setup and processing durations, whereas makespan also reflects precedence and calendar waiting. Exact evaluation equivalence and component-level timing do not establish universal search superiority or whole-algorithm acceleration. The manuscript's budget, metric and statistical qualifications apply to the reported comparisons.

The 2026-11-17 independent repetition reproduced the archived deterministic confirmation outputs. Same-seed repetitions are implementation checks and do not increase the number of independent experimental units. Timing is machine dependent.

## Citation and funding

Use the metadata in `CITATION.cff` and cite [release v0.1.0](https://github.com/dydot923/ijamt-fjsp-reproducibility/releases/tag/v0.1.0) for this version. No article or dataset DOI has been assigned.

This study was supported by the National Key Research and Development Program of China (Grant No. 2023YFE0113800).

## Licensing

A software license has not yet been selected by the rights holders. See [licensing status](docs/LICENSING.md). The absence of a license does not grant an open-source reuse license. Third-party input data require separate provenance and redistribution terms.
