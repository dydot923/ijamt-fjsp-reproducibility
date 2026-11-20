"""Independent manuscript numerical audit; no imports from original analysis.

Uses atomic front records, an independent rectangle-union HV implementation,
explicit signed-rank subset sums, and independently reconstructed table cells.
--source fresh can be run once the new 678-task grid is complete.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import itertools
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "04_修订与核查/工具/python_deps"))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import linprog
from docx import Document

METHODS = ["H-NSGA", "H-MVNS-6", "H-MVNS-2", "RP-NSGA"]
INSTANCES = ["E0"] + [f"Mk{i:02}" for i in range(1, 11)]


def read(path):
    return json.loads(Path(path).read_text("utf8"))


def nd(points):
    # Frozen numerical policy rounds reference candidates to 9 decimals.
    a = np.unique(np.round(np.asarray(points, float), 9), axis=0)
    a = a[np.argsort(a[:, 0], kind="stable")]
    keep = []
    for p in a:
        if keep:
            delta = np.asarray(keep) - p
            if np.any(np.max(delta, axis=1) <= 1e-9):
                continue
            removed = (np.min(delta, axis=1) >= -1e-9) & (np.max(delta, axis=1) > 1e-9)
            keep = [q for q, drop in zip(keep, removed) if not drop]
        keep.append(p)
    return np.asarray(keep)


def fps(points, limit):
    a = np.unique(np.round(np.asarray(points, float), 9), axis=0)
    if len(a) <= limit:
        return a
    chosen = list(dict.fromkeys(np.argmin(a, axis=0).tolist()))
    while len(chosen) < limit:
        # Full recomputation rather than the source's incremental distance cache.
        distances = np.min(np.sum((a[:, None] - a[chosen][None]) ** 2, axis=2), axis=1)
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return a[chosen]


def hv(points):
    # Union of boxes [p, (1,1,1)], integrate x slabs then sweep y slabs.
    a = np.unique(np.round(np.asarray(points, float), 12), axis=0)
    a = a[np.all(a < 1, axis=1)]
    cuts = np.unique(np.r_[a[:, 0], 1.0])
    volume = 0.
    for left, right in zip(cuts[:-1], cuts[1:]):
        yz = a[a[:, 0] <= left, 1:]
        # Vectorize the inner rectangle sweep; duplicate y coordinates give
        # zero-width slabs, leaving exactly the same rectangle union.
        yz = yz[np.argsort(yz[:, 0])]
        widths = np.diff(np.r_[yz[:, 0], 1.0])
        heights = 1.0 - np.minimum.accumulate(yz[:, 1])
        area = float(widths @ heights)
        volume += (right - left) * area
    return float(volume)


def igd(front, ref):
    # Direct reference-point loop avoids using the analysis distance helper.
    a = np.asarray(front)
    return float(np.mean([np.min(np.linalg.norm(np.clip(a - p, 0, None), axis=1)) for p in ref]))


def signed_rank(d):
    v = np.asarray(d, float)
    v = v[v != 0]
    if not len(v):
        return 1., 0
    order = sorted(range(len(v)), key=lambda i: abs(v[i]))
    twice_ranks = np.zeros(len(v), dtype=int)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and abs(v[order[end]]) == abs(v[order[start]]):
            end += 1
        twice_ranks[order[start:end]] = start + 1 + end
        start = end
    observed = abs(int(np.dot(np.sign(v).astype(int), twice_ranks)))
    distribution = {0: 1}
    for rank in twice_ranks:
        updated = defaultdict(int)
        for value, count in distribution.items():
            updated[value - int(rank)] += count
            updated[value + int(rank)] += count
        distribution = updated
    p = sum(count for value, count in distribution.items() if abs(value) >= observed) / 2 ** len(v)
    return p, len(v)


def effect(values, family, metric, control, budget):
    d = np.array(values, float, copy=True)
    if metric != "hv":
        d *= -1
    d[abs(d) < 1e-12] = 0
    # The frozen RNG and percentile estimator are part of the protocol.
    rng = np.random.default_rng(20261008)
    samples = rng.integers(0, len(d), (20000, len(d)))
    lo, hi = np.quantile(np.mean(d[samples], axis=1), [.025, .975])
    p, n_nonzero = signed_rank(d)
    return dict(family=family, metric=metric, control=control, budget=str(budget), n=len(d),
                delta=float(np.mean(d)), median_delta=float(np.median(d)), lo=float(lo), hi=float(hi),
                wins=int(np.sum(d > 0)), ties=int(np.sum(d == 0)), p=p,
                nonzero_pairs=n_nonzero, minimum_two_sided_p=2 / 2**n_nonzero if n_nonzero else 1)


def adjust(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    raw = [min(1, rows[i]["p"] * (len(rows) - j)) for j, i in enumerate(order)]
    adjusted = np.maximum.accumulate(raw)
    for i, p in zip(order, adjusted):
        rows[i]["p_holm"] = float(p)


def sign_sensitivity(effects):
    """Post hoc diagnostic of equal sign probabilities; not the locked test.

    Conditioning on nonzero differences removes ties. Unlike signed ranks,
    this test ignores magnitudes and does not require symmetry of magnitudes
    around zero. It still requires exchangeable/independent paired units and
    equal probabilities of positive and negative signs under its null.
    """
    rows = []
    for effect_row in effects:
        row = {k: effect_row[k] for k in ("family", "metric", "control", "budget", "n", "wins", "ties", "delta")}
        row["losses"] = row["n"] - row["wins"] - row["ties"]
        nonzero = row["n"] - row["ties"]
        extreme = min(row["wins"], row["losses"])
        row["p"] = min(1., 2 * sum(math.comb(nonzero,k) for k in range(extreme+1)) / 2**nonzero) if nonzero else 1.
        row["signed_rank_p"] = effect_row["p"]
        row["signed_rank_p_holm"] = effect_row["p_holm"]
        row["analysis_status"] = "EXPLORATORY POST HOC ASSUMPTION SENSITIVITY"
        rows.append(row)
    for family in ("F1-Mk", "F1-E0", "F2", "F3"):
        subset = [r for r in rows if r["family"] == family]
        adjust(subset)
    for row in rows:
        row["signed_rank_rejects_005"] = row["signed_rank_p_holm"] < .05
        row["sign_test_rejects_005"] = row["p_holm"] < .05
        row["decision_changed"] = row["signed_rank_rejects_005"] != row["sign_test_rejects_005"]
    return rows


def dataset(instance):
    path = ROOT / "replication/data" / ("v42_dataset.json" if instance == "E0" else f"benchmarks/{instance}.json")
    raw = read(path)
    if "operations" in raw:
        by_job = defaultdict(list)
        by_op = defaultdict(list)
        for op in raw["operations"]:
            by_op[(op["job_id"], op["op_index"])].append(op)
        for (job, _), opts in by_op.items():
            by_job[job].append(opts)
        jobs = [(j, by_job[j["job_id"]]) for j in raw["jobs"]]
    else:
        jobs = [(j, [op["candidates"] for op in j["operations"]]) for j in raw["jobs"]]
    return raw, jobs


def energy(option):
    return option.get("total_energy", option.get("running_energy", 0) + option.get("auxiliary_energy", 0))


def analytic_contract():
    frozen = read(ROOT / "frozen/replication/results_v3/analytic-bounds.json")["bounds"]
    sizes = []
    for instance in INSTANCES:
        raw, jobs = dataset(instance)
        operations = [opts for _, ops in jobs for opts in ops]
        minima = [min(o["processing_time"] for o in opts) for opts in operations]
        low = [max(sum(minima) / len(raw["machines"]),
                   max(j.get("release_time", 0) + sum(min(o["processing_time"] for o in opts) for opts in ops) for j, ops in jobs)),
               sum(min(energy(o) for o in opts) for opts in operations),
               max(sum(minima) / len(raw["machines"]), max(minima))]
        occupy = sum(max(o["processing_time"] + o.get("setup_time", 0) for o in opts) for opts in operations)
        maintenance = max([0] + [b for m in raw["machines"] for a, b in m.get("maintenance_windows", [])])
        high = [max(j.get("release_time", 0) for j, _ in jobs) + maintenance + occupy,
                sum(max(energy(o) for o in opts) for opts in operations), occupy]
        reference = np.array(high) + np.maximum(1, .1 * (np.array(high) - low))
        for label, values in (("lower", low), ("upper", high), ("reference", reference)):
            assert np.max(np.abs(np.array(values) - frozen[instance][label])) < 1e-9, (instance, label)
        sizes.append([instance, len(jobs), f'{len(raw["machines"])}/{len({o["machine_id"] for opts in operations for o in opts})}',
                      len(operations), sum(map(len, operations))])
    return frozen, sizes


def metrics(source):
    original = ROOT / "frozen/replication/results_v3"
    output = ROOT / ("frozen/replication/results_v3" if source == "frozen" else "replication/results_v3")
    rows = [read(p) for p in sorted((output / "completed").glob("*.json"))]
    expected = {(i, m, s, b) for i in INSTANCES for m in METHODS[:3] for s in range(450001, 450007) for b in (2000, 4000, 8000)}
    expected |= {("E0", m, s, 4000) for m in METHODS[:3] for s in range(450007, 450013)}
    expected |= {(i, "RP-NSGA", s, 4000) for i in INSTANCES for s in range(450001, 450007)}
    assert len(rows) == 678 and {(r["instance"], r["method"], r["seed"], r["budget"]) for r in rows} == expected
    bounds, sizes = analytic_contract()
    records = []
    references = {}
    for instance in INSTANCES:
        group = [r for r in rows if r["instance"] == instance]
        lower = np.asarray(bounds[instance]["lower"])
        scale = np.asarray(bounds[instance]["reference"]) - lower
        normalize = lambda a: (np.asarray(a, float) - lower) / scale
        primary_sources = [r for r in group if r["budget"] == 4000 and r["seed"] <= 450006]
        assert len(primary_sources) == 24
        sampled = []
        for r in primary_sources:
            a = normalize(r["front"])
            sampled.append(np.unique(np.r_[fps(a, 50), a[np.argmin(a, axis=0)]], axis=0))
        primary = fps(nd(np.concatenate(sampled)), 500)
        all_points = nd(np.concatenate([normalize(r["front"]) for r in group]))
        references[instance] = {"primary_points": len(primary), "all_points": len(all_points)}
        for r in group:
            a = normalize(r["front"])
            c = r["decoder_counts"]
            assert r["evaluations"] == r["budget"] and r["invalid_schedules"] == 0
            assert c["full_calls"] + c["prefix_calls"] == r["evaluations"]
            count_ops = next(x[3] for x in sizes if x[0] == instance)
            assert c["reused_operations"] + c["reinserted_operations"] == r["evaluations"] * count_ops
            cfg = r["config"]
            assert cfg["population_size"] == 40 and cfg["restart_ratio"] == 0 and cfg["initialization_mode"] == "hybrid"
            assert cfg["local_search_trials"] == (2 if r["method"] == "H-MVNS-2" else 6)
            if r["method"] in ("H-NSGA", "RP-NSGA"):
                assert cfg["vns_probability"] == cfg["elite_local_search_ratio"] == 0
            row = {k: r[k] for k in ("instance", "method", "seed", "budget", "generations", "audited_schedules")}
            row.update(hv=hv(a), igdplus=igd(a, primary), igdplus_all=igd(a, all_points),
                       front_size=len(r["front"]), local_fresh=sum(v["fresh"] for v in r["local"].values()), **c)
            records.append(row)
        print(f"Recomputed {instance}: {len(group)} runs", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(ROOT / f"reports/statistics-{source}-metrics.csv", index=False)
    old = pd.read_csv(original / "metrics.csv")
    joined = frame.merge(old, on=["instance", "method", "seed", "budget"], suffixes=("_new", "_old"), validate="one_to_one")
    differences = {k: float(np.max(abs(joined[k + "_new"] - joined[k + "_old"]))) for k in ("hv", "igdplus", "igdplus_all")}
    if source == "frozen":
        assert max(differences.values()) < 1e-11, differences
    return frame, rows, sizes, references, differences


def make_effects(frame):
    means = frame.groupby(["instance", "method", "budget"]).mean(numeric_only=True).reset_index()
    effects = []
    for population in ("Mk", "E0"):
        part = means[(means.instance != "E0") & (means.budget == 4000)] if population == "Mk" else frame[(frame.instance == "E0") & (frame.budget == 4000)]
        family = []
        for metric in ("hv", "igdplus"):
            p = part.pivot(index="instance" if population == "Mk" else "seed", columns="method", values=metric)
            for control in METHODS[:2]:
                values = (p["H-MVNS-2"] - p[control]).dropna().to_numpy()
                assert len(values) == (10 if population == "Mk" else 12)
                family.append(effect(values, "F1-" + population, metric, control, 4000))
        adjust(family)
        effects.extend(family)
    mk = means[means.instance != "E0"]
    family = []
    for budget in (2000, 8000, "trapezoidal mean"):
        for metric in ("hv", "igdplus"):
            if isinstance(budget, int):
                part = mk[mk.budget == budget]
            else:
                part = mk[mk.method != "RP-NSGA"].copy()
                part[metric] *= part.budget.map({2000: 1/6, 4000: 1/2, 8000: 1/3})
                part = part.groupby(["instance", "method"])[metric].sum().reset_index()
            p = part.pivot(index="instance", columns="method", values=metric)
            for control in METHODS[:2]:
                family.append(effect(p["H-MVNS-2"] - p[control], "F2", metric, control, budget))
    adjust(family)
    effects.extend(family)
    family = []
    for metric in ("hv", "igdplus"):
        p = mk[mk.budget == 4000].pivot(index="instance", columns="method", values=metric)
        family.append(effect(p["H-MVNS-2"] - p["RP-NSGA"], "F3", metric, "RP-NSGA", 4000))
    adjust(family)
    return effects + family


def text_cell(cell):
    from lxml import etree
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main", "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
    return "".join(cell._tc.xpath(".//w:t/text()|.//m:t/text()")).replace("\u200b", "")


def table_contract(frame, rows, sizes, effects, source="frozen"):
    manuscript = ROOT.parent / "01_当前稿/三目标柔性作业车间调度_理论与叙事增强稿_20261029.docx"
    snapshot = ROOT / "reports/table-contract.json"
    if manuscript.exists():
        doc = Document(manuscript)
        tables = [[[text_cell(c) for c in r.cells] for r in t.rows] for t in doc.tables]
        snapshot.write_text(json.dumps(tables, ensure_ascii=False, indent=2), "utf8")
    else:
        tables = read(snapshot)
    result = []
    assert len(tables) == 13
    assert len(tables[0]) == 7
    result.append({"table": 1, "status": "notation structure checked; mathematical meaning reviewed by theory audit"})
    assert [[str(v) for v in r] for r in sizes] == tables[1][1:]
    result.append({"table": 2, "cells": 55, "status": "independent JSON counts PASS"})
    assert [r[:2] for r in tables[2][1:]] == [["H-NSGA", "0 / 0"], ["H-MVNS-6", "6 / 24"], ["H-MVNS-2", "2 / 8"], ["RP-NSGA", "0 / 0"]]
    result.append({"table": 3, "status": "atomic configs and no-restart/local-intensity settings PASS"})
    raw = ROOT / ("qa/theory-upgrade-evidence/single-move-raw.jsonl" if source == "fresh"
                  else "frozen/qa/theory-upgrade-evidence/single-move-raw.jsonl")
    moves = [json.loads(line) for line in raw.read_text("utf8").splitlines()]
    expected = []
    for scenario in ("E0", "E1", "E2"):
        g = [r for r in moves if r["scenario"] == scenario]
        e = np.array([r["proxy_error"] for r in g])
        rho = []
        for seed in sorted({r["seed"] for r in g}):
            a = [r for r in g if r["seed"] == seed]
            rho.append(spearmanr([round(max(r["predicted_loads"].values()), 9) for r in a], [round(r["new_objectives"][2], 9) for r in a]).statistic)
        expected.append([scenario, str(len(g)), f'{sum(e < -1e-8)}/{sum(abs(e) <= 1e-8)}/{sum(e > 1e-8)}',
                         f'{max(abs(e)):.2f}', f'{np.mean([max(r["upper"].values()) - max(r["lower"].values()) for r in g]):.2f}',
                         str(sum(max(r["upper"].values()) < r["old_objectives"][2] - 1e-8 for r in g)),
                         str(sum(max(r["lower"].values()) >= r["old_objectives"][2] - 1e-8 for r in g)), f'{np.mean(rho):.4f}'])
    assert expected == tables[3][1:], (expected, tables[3])
    result.append({"table": 4, "status": "6936 raw moves independently aggregated PASS"})
    oracle_root = ROOT / ("replication/results_v3/oracle" if source == "fresh" else "frozen/replication/results_v3/oracle")
    oracle = read(oracle_root / "summary.json")
    runs = read(oracle_root / "runs.json")
    expected = []
    for r in oracle:
        case = r["instance"]
        ratios = [np.mean([x["hv_ratio"] for x in runs if x["instance"] == case and x["method"] == m]) for m in METHODS[:3]]
        expected.append([case, "3×2" if case in ("T1", "T2", "T3") else "2×3", str(r["orientations"]), str(r["acyclic"]), str(r["encodings"]),
                         f'{len(r["exact_front"])}/{len(r["decoder_front"])}', *[f'{x:.5f}' for x in ratios]])
    assert expected == tables[4][1:]
    result.append({"table": 5, "status": "144 raw oracle run aggregates PASS; fresh enumeration separately executed by reproduction audit"})
    expected = []
    for population in ("E0", "Mk"):
        p = frame[(frame.budget == 4000) & ((frame.instance == "E0") if population == "E0" else (frame.instance != "E0"))]
        if population == "Mk":
            p = p.groupby(["instance", "method"]).mean(numeric_only=True).reset_index()
        for method in METHODS:
            g = p[p.method == method]
            expected.append([population, method, str(len(g)), f'{g.hv.mean():.5f} ± {g.hv.std():.5f}', f'{g.igdplus.mean():.5f} ± {g.igdplus.std():.5f}',
                             f'{g.front_size.mean():.1f}', f'{(100*g.local_fresh/g.budget).mean():.1f}'])
    # OMML plus/minus appears in text extraction; tolerate only whitespace, not values.
    normalize = lambda s: "".join(s.split())
    assert [[normalize(x) for x in r] for r in expected] == [[normalize(x) for x in r] for r in tables[5][1:]], (expected, tables[5])
    result.append({"table": 6, "status": "independent per-seed/per-instance mean and sample SD PASS"})
    expected = []
    for r in effects:
        if r["family"] == "F2":
            continue
        expected.append([r["family"], r["control"], "HV" if r["metric"] == "hv" else "IGD+", str(r["n"]), f'{r["delta"]:+.6f}',
                         f'[{r["lo"]:+.6f}, {r["hi"]:+.6f}]', f'{r["p_holm"]:.5f}', f'{r["wins"]}/{r["ties"]}/{r["n"]-r["wins"]-r["ties"]}'])
    assert expected == tables[6][1:], (expected, tables[6])
    result.append({"table": 7, "status": "independent signed-rank subset-sum, bootstrap and Holm PASS"})
    expected = []
    means = frame.groupby(["instance", "method", "budget"]).mean(numeric_only=True).reset_index()
    for instance in INSTANCES[1:]:
        cells = [instance]
        for method in METHODS:
            g = means[(means.instance == instance) & (means.method == method) & (means.budget == 4000)].iloc[0]
            cells.append(f'{g.hv:.5f} / {g.igdplus:.5f}')
        expected.append(cells)
    assert expected == tables[7][1:]
    result.append({"table": 8, "status": "all 40 independently recomputed metric-pair cells PASS"})
    expected = []
    for r in effects:
        if r["family"] != "F2":
            continue
        expected.append(["梯形均值" if r["budget"] == "trapezoidal mean" else r["budget"], r["control"], "HV" if r["metric"] == "hv" else "IGD+",
                         f'{r["delta"]:+.6f}', f'[{r["lo"]:+.6f},{r["hi"]:+.6f}]', f'{r["p_holm"]:.5f}', str(r["wins"])])
    assert expected == tables[8][1:]
    result.append({"table": 9, "status": "all 12 F2 comparisons incl. trapezoid weights PASS"})
    expected = []
    for budget in (2000, 4000, 8000):
        for method in METHODS[:3]:
            g = frame[(frame.instance == "E0") & (frame.method == method) & (frame.budget == budget) & (frame.seed <= 450006)]
            expected.append([str(budget), method, f'{g.hv.mean():.5f}', f'{g.igdplus.mean():.5f}', f'{(100*g.local_fresh/g.budget).mean():.1f}',
                             f'{g.generations.mean():.1f}', f'{(100*g.reused_operations/(g.budget*58)).mean():.1f}'])
    assert expected == tables[9][1:]
    result.append({"table": 10, "status": "common-six-seed evaluation, generation and reuse counts PASS"})
    step = pd.read_csv(ROOT / ("replication/results_support/single_step.csv" if source == "fresh"
                               else "frozen/replication/results_v2/single_step.csv"))
    expected = []
    for focus in ("Cmax", "TEC", "Lmax"):
        for version in ("original", "matched"):
            g = step[(step.focus == focus) & (step.version == version)]
            assert len(g) == 200 and len(g.parent_seed.unique()) == 10
            expected.append([focus, "原邻域" if version == "original" else "目标匹配", f'{100*g.groupby("parent_seed").relative_target.mean().mean():+.3f}',
                             f'{g.accepted.mean():.3f}', f'{g.duplicate.mean():.3f}'])
    assert expected == tables[10][1:]
    result.append({"table": 11, "status": "1200 actual exploratory proposals, aggregate within 10 parents PASS"})
    if source == "fresh":
        factorial = read(ROOT / "replication/results_support/factorial_runs.json")
    else:
        factorial = read(ROOT / "frozen/replication/results_v2/constraint_factorial_runs.json")
        factorial += [r for r in read(ROOT / "frozen/replication/results_v2/matched_runs.json")
                      if r["method"] == "H-MVNS" and r["instance"] in ("E1", "E2")]
    expected = []
    by_scenario = {}
    for instance, name in (("CF0", "无附加约束"), ("E1", "换型"), ("CFM", "维护"), ("E2", "换型与维护")):
        g = sorted([r for r in factorial if r["instance"] == instance], key=lambda r:r["seed"])
        assert len(g) == 12 and [r["seed"] for r in g] == list(range(430001,430013))
        endpoints = np.array([np.min(r["front"], axis=0) for r in g])
        by_scenario[instance] = endpoints
        expected.append([instance, name, *[f'{endpoints[:,i].mean():.2f} ± {endpoints[:,i].std(ddof=1):.2f}' for i in range(3)]])
    assert [[normalize(x) for x in r] for r in expected] == [[normalize(x) for x in r] for r in tables[11][1:]], (expected, tables[11])
    interaction = by_scenario["E2"] - by_scenario["E1"] - by_scenario["CFM"] + by_scenario["CF0"]
    rng = np.random.default_rng(20261007)
    estimates = interaction[:,0][rng.integers(0,12,(20000,12))].mean(axis=1)
    interaction_result = dict(mean=float(interaction[:,0].mean()), ci=list(np.quantile(estimates,[.025,.975])))
    result.append({"table": 12, "status": "48 raw factorial endpoint aggregates PASS", "cmax_interaction": interaction_result})
    raw, jobs = dataset("E0")
    opts = [a for _, ops in jobs for a in ops]
    machines = sorted({o["machine_id"] for a in opts for o in a})
    arcs = [(i,o) for i,a in enumerate(opts) for o in a]
    n = len(arcs)
    eq = np.zeros((len(opts),n+1))
    ub = np.zeros((len(machines)+1,n+1))
    for k,(i,o) in enumerate(arcs):
        eq[i,k] = 1
        ub[machines.index(o["machine_id"]),k] = o["processing_time"]
        ub[-1,k] = energy(o)
    ub[:-1,-1] = -1
    objective = np.r_[np.zeros(n),1]
    expected = []
    for row in tables[12][1:]:
        cap = float(row[0])
        bound = linprog(objective, A_ub=ub, b_ub=np.r_[np.zeros(len(machines)),cap], A_eq=eq, b_eq=np.ones(len(opts)), bounds=(0,None),method="highs")
        assert bound.success
        expected.append([f'{cap:.2f}', f'{bound.fun:.2f}'])
    assert expected == tables[12][1:]
    result.append({"table": 13, "status": "8 independently assembled LPs without redundant arc upper bounds PASS"})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("frozen","fresh"), default="frozen")
    args = parser.parse_args()
    frame, rows, sizes, references, differences = metrics(args.source)
    effects = make_effects(frame)
    pd.DataFrame(effects).to_csv(ROOT / f"reports/statistics-{args.source}-effects.csv", index=False)
    sensitivity = sign_sensitivity(effects)
    pd.DataFrame(sensitivity).to_csv(ROOT / f"reports/statistics-{args.source}-sign-sensitivity.csv", index=False)
    old = pd.read_csv(ROOT / "frozen/replication/results_v3/effects.csv",dtype={"budget":str})
    actual = pd.DataFrame(effects)
    joined = actual.merge(old,on=["family","metric","control","budget"],suffixes=("_new","_old"),validate="one_to_one")
    effect_differences = {k:float(max(abs(joined[k+"_new"] - joined[k+"_old"]))) for k in ("delta","lo","hi","p","p_holm")}
    if args.source == "frozen":
        assert max(effect_differences.values()) < 1e-10, effect_differences
    tables = table_contract(frame, rows, sizes, effects, args.source)
    report = dict(status="PASS", source=args.source, raw_runs=len(rows), evaluations=sum(r["evaluations"] for r in rows),
                  audited_schedules=sum(r["audited_schedules"] for r in rows),
                  decoder_totals={k:sum(r["decoder_counts"][k] for r in rows) for k in rows[0]["decoder_counts"]},
                  references=references, metric_max_abs_difference_from_archive=differences,
                  effect_max_abs_difference_from_archive=effect_differences, statistical_families={"F1-Mk":4,"F1-E0":4,"F2":12,"F3":2},
                  effects=effects, tables=tables, exploratory_sign_test_sensitivity=sensitivity,
                  limitations=[("No fresh search run is implied by frozen-data recomputation." if args.source == "frozen"
                                else "Fresh same-seed reproduction does not increase the independent confirmatory sample size."),
                               "Exact signed-rank probabilities condition on nonzero absolute differences and assume sign symmetry under the null.",
                               "Bootstrap intervals concern mean paired differences and are pointwise, not simultaneous or inverted signed-rank tests.",
                               "Fixed benchmark means do not constitute random samples of all industrial shops.",
                               "Single-step and factorial records are exploratory; parent/run remains the unit."])
    target = ROOT / f"reports/statistics-{args.source}-audit.json"
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),"utf8")
    print(json.dumps({"status":report["status"], "raw_runs":report["raw_runs"],"tables":len(tables),
                      "metric_max_difference":differences,"effect_max_difference":effect_differences},ensure_ascii=True))


if __name__ == "__main__":
    main()
