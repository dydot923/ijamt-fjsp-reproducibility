"""Independent theory/code audit. Writes reports only; frozen evidence is read-only."""
from collections import Counter, defaultdict
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT.parent / "04_修订与核查/工具/python_deps"
sys.path[:0] = [str(DEPS), str(ROOT / "replication")]
import numpy as np
from scipy.optimize import linprog
from fjsp.data_model import ProblemData, JobSpec, MachineSpec, OperationSpec, OperationOption, Schedule, ScheduleEntry
from fjsp.dataset_io import load_any_dataset
from fjsp.decoder import decode
from fjsp.nsga_vns import _build_order_sequence, _build_machine_assignment, Individual, _ObjectivePolicy, _score_pair
from fjsp.objectives import static_objectives
from prefix_evaluator import make_evaluator
from run_budget_study import make_data

OUT = ROOT / "reports"
FROZEN = ROOT / "frozen"
TOL = 1e-7


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf8")


def mapped(data, os):
    """Reconstruct occurrences without calling decoder's mapping helper."""
    seen = Counter()
    result = []
    for job in os:
        seen[job] += 1
        result.append(data.jobs[job].operations[seen[job] - 1].operation_id)
    assert seen == {j: len(spec.operations) for j, spec in data.jobs.items()}
    return result


def audit_schedule(data, schedule, claimed=None):
    """Use input arcs and final machine order, not solver's checker/objectives."""
    assert schedule.feasible, schedule.violations
    assert set(schedule.entries) == set(data.operations)
    per_machine = defaultdict(list)
    energies, processing, setups = {}, defaultdict(float), defaultdict(float)
    for oid, entry in schedule.entries.items():
        operation = data.operations[oid]
        assert entry.job_id == operation.job_id and entry.op_index == operation.op_index
        opts = {opt.machine_id: opt for opt in operation.candidates}
        assert entry.machine_id in opts
        opt = opts[entry.machine_id]
        assert min(entry.setup_start, entry.start, entry.end) >= -TOL
        assert abs(entry.end - entry.start - opt.processing_time) < TOL
        assert abs(entry.processing_time - opt.processing_time) < TOL
        assert abs(entry.start - entry.setup_start - entry.setup_time) < TOL
        assert abs(entry.energy - opt.total_energy) < TOL
        for left, right in data.machines[entry.machine_id].maintenance_windows:
            assert entry.end <= left + TOL or entry.setup_start >= right - TOL
        energies[oid] = opt.total_energy
        processing[entry.machine_id] += opt.processing_time
        per_machine[entry.machine_id].append(entry)
    for job in data.jobs.values():
        ready = job.release_time
        for operation in job.operations:
            e = schedule.entries[operation.operation_id]
            assert e.setup_start >= ready - TOL
            ready = e.end
    for machine, entries in per_machine.items():
        entries.sort(key=lambda e: (e.setup_start, e.start, e.operation_id))
        previous = None
        for e in entries:
            if previous:
                assert e.setup_start >= previous.end - TOL
            opt = next(o for o in data.operations[e.operation_id].candidates if o.machine_id == machine)
            q = 0 if previous and data.jobs[previous.job_id].family == data.jobs[e.job_id].family else opt.setup_time
            assert abs(e.setup_time - q) < TOL
            setups[machine] += q
            previous = e
    loads = {m: processing[m] + setups[m] for m in data.machines}
    f = (max(e.end for e in schedule.entries.values()), math.fsum(energies.values()), max(loads.values()))
    assert f[2] <= f[0] + TOL
    if claimed is not None:
        assert all(abs(a-b) < 1e-6 for a,b in zip(f,claimed)), (f, claimed)
    return dict(objectives=f, loads=loads, processing=dict(processing), setups=dict(setups))


def independent_cut(data, old_os, old_ma, new_os, new_ma):
    old_ids, new_ids = mapped(data, old_os), mapped(data, new_os)
    cut = 0
    for a, b in zip(old_ids, new_ids):
        if a != b or old_ma[a] != new_ma[b]:
            break
        cut += 1
    return cut, new_ids


def interval(data, base, ids, ma, cut):
    prefix = set(ids[:cut])
    lower, upper = {}, {}
    for m in data.machines:
        retained = [e for oid,e in base.entries.items() if oid in prefix and e.machine_id == m]
        retained_setup = math.fsum(e.setup_time for e in retained)
        present = {data.jobs[e.job_id].family for e in retained}
        new_families = {}
        max_extra = processing = 0.0
        for v in ids:
            if ma[v] != m:
                continue
            opt = next(o for o in data.operations[v].candidates if o.machine_id == m)
            processing += opt.processing_time
            if v in prefix:
                continue
            max_extra += opt.setup_time
            family = data.jobs[data.operations[v].job_id].family
            if family not in present:
                new_families[family] = min(new_families.get(family, math.inf), opt.setup_time)
        lower[m] = processing + retained_setup + sum(new_families.values())
        upper[m] = processing + retained_setup + max_extra
    return lower, upper


def moves():
    counts = Counter()
    maxima = dict(energy_increment=0.0, zero_setup_increment=0.0, setup_error_identity=0.0)
    mismatch_saved = 0
    frozen_raw = (FROZEN / "qa/theory-upgrade-evidence/single-move-raw.jsonl").open(encoding="utf8")
    raw_path = OUT / "theory-single-moves.jsonl"
    with raw_path.open("w", encoding="utf8") as stream:
        for scenario in ("E0", "E1", "E2"):
            data = make_data(scenario)
            for seed in range(551001,551009):
                rng = random.Random(seed)
                os = _build_order_sequence(data,"random",rng)
                ma = _build_machine_assignment(data,"random",rng)
                base = decode(data,os,ma)
                old = audit_schedule(data,base,static_objectives(data,base))
                for v in data.operation_ids:
                    a = ma[v]
                    old_opt = data.option(v,a)
                    for opt in data.operations[v].candidates:
                        b = opt.machine_id
                        if b == a:
                            continue
                        candidate_ma = dict(ma);candidate_ma[v] = b
                        candidate = decode(data,os,candidate_ma)
                        new = audit_schedule(data,candidate,static_objectives(data,candidate))
                        err = abs(new["objectives"][1]-old["objectives"][1]-(opt.total_energy-old_opt.total_energy))
                        assert err < TOL
                        maxima["energy_increment"] = max(maxima["energy_increment"],err)
                        proxy = dict(old["loads"])
                        proxy[a] -= old_opt.processing_time + base.entries[v].setup_time
                        proxy[b] += opt.processing_time + opt.setup_time
                        eps = {m: new["setups"].get(m,0)-old["setups"].get(m,0)+
                               (base.entries[v].setup_time if m == a else 0)-
                               (opt.setup_time if m == b else 0) for m in data.machines}
                        identity = max(abs(new["loads"][m]-proxy[m]-eps[m]) for m in data.machines)
                        assert identity < TOL
                        maxima["setup_error_identity"] = max(maxima["setup_error_identity"],identity)
                        max_error = new["objectives"][2]-max(proxy.values())
                        assert min(eps.values())-TOL <= max_error <= max(eps.values())+TOL
                        if scenario == "E0":
                            maxima["zero_setup_increment"] = max(maxima["zero_setup_increment"],abs(max_error))
                            assert abs(max_error) < TOL
                        cut,ids = independent_cut(data,os,ma,os,candidate_ma)
                        low,high = interval(data,base,ids,candidate_ma,cut)
                        assert all(low[m]-TOL <= new["loads"][m] <= high[m]+TOL for m in data.machines)
                        assert all(vars(base.entries[oid]) == vars(candidate.entries[oid]) for oid in ids[:cut])
                        old_record = json.loads(next(frozen_raw))
                        assert (old_record["scenario"],old_record["seed"],old_record["operation"],old_record["target"]) == (scenario,seed,v,b)
                        if any(abs(x-y)>1e-7 for x,y in zip(new["objectives"],old_record["new_objectives"])):
                            mismatch_saved += 1
                        stream.write(json.dumps(dict(scenario=scenario,seed=seed,operation=v,source=a,target=b,
                            cut=cut,objectives=new["objectives"],energy_increment_error=err,
                            setup_error_identity=identity,interval_lower=max(low.values()),interval_upper=max(high.values())),ensure_ascii=False)+"\n")
                        counts[scenario] += 1
    assert sum(counts.values()) == 6936 and mismatch_saved == 0
    assert next(frozen_raw,None) is None
    frozen_raw.close()
    return dict(cases=dict(counts),total=6936,max_absolute_error=maxima,saved_objective_mismatches=mismatch_saved)


def prefixes():
    counters = dict(full_calls=0,prefix_calls=0,reused_operations=0,reinserted_operations=0)
    evaluator = make_evaluator(counters)
    counts = Counter()
    output = []
    def check(data,os,ma,new_os,new_ma,mode):
        base_schedule = decode(data,os,ma)
        base = SimpleNamespace(order_sequence=os,machine_assignment=ma,schedule=base_schedule)
        candidate = Individual(list(new_os),dict(new_ma));candidate._base = base
        evaluator(candidate,data)
        full = decode(data,new_os,new_ma)
        assert {k:vars(v) for k,v in full.entries.items()} == {k:vars(v) for k,v in candidate.schedule.entries.items()}
        assert static_objectives(data,full) == candidate.objectives
        new = audit_schedule(data,candidate.schedule,candidate.objectives)
        cut,ids = independent_cut(data,os,ma,new_os,new_ma)
        low,high = interval(data,base_schedule,ids,new_ma,cut)
        assert all(low[m]-TOL <= new["loads"][m] <= high[m]+TOL for m in data.machines)
        counts[mode] += 1;counts["total"] += 1
        counts["cut_zero"] += int(cut == 0);counts["cut_full"] += int(cut == len(ids))
        output.append(dict(mode=mode,cut=cut,objectives=candidate.objectives,entries_equal=True))
    for scenario in ("E0","E1","E2"):
        data = make_data(scenario)
        for seed in range(771001,771004):
            rng = random.Random(seed)
            os = _build_order_sequence(data,"random",rng);ma = _build_machine_assignment(data,"random",rng)
            moves = [(v,z.machine_id) for v in mapped(data,os) for z in data.operations[v].candidates if z.machine_id != ma[v]]
            for v,target in rng.sample(moves,50):
                new_ma = dict(ma);new_ma[v] = target
                check(data,os,ma,os,new_ma,"ma_only")
        for seed in range(991001,991004):
            rng = random.Random(seed)
            os = _build_order_sequence(data,"random",rng);ma = _build_machine_assignment(data,"random",rng)
            ids = mapped(data,os)
            for trial in range(84):
                new_os,new_ma = list(os),dict(ma)
                mode = ["unchanged","first_ma","last_ma","os_swap","os_insert","mixed","multiple_ma"][trial%7]
                if mode in ("os_swap","mixed"):
                    i,j = rng.sample(range(len(new_os)),2);new_os[i],new_os[j] = new_os[j],new_os[i]
                if mode == "os_insert":
                    i,j = rng.sample(range(len(new_os)),2);new_os.insert(j,new_os.pop(i))
                targets = [ids[0]] if mode == "first_ma" else [ids[-1]] if mode == "last_ma" else rng.sample(data.operation_ids,3) if mode in ("mixed","multiple_ma") else []
                for v in targets:
                    alternatives = [z.machine_id for z in data.operations[v].candidates if z.machine_id != new_ma[v]]
                    if alternatives:new_ma[v] = rng.choice(alternatives)
                check(data,os,ma,new_os,new_ma,mode)
    assert counts["total"] == 1206 and counts["ma_only"] == 450
    dump("theory-prefix-cases.json",output)
    return dict(cases=dict(counts),decoder_counts=counters,entry_and_objective_equality=True)


def independent_lp(data,cap=None):
    """Lambda is the first variable, no redundant y<=1 bounds, eligible machines only."""
    ops = list(data.operations)
    eligible = sorted({o.machine_id for spec in data.operations.values() for o in spec.candidates})
    arcs = [(v,o) for v in ops for o in data.operations[v].candidates]
    eq = np.zeros((len(ops),len(arcs)+1))
    ub = np.zeros((len(eligible)+(cap is not None),len(arcs)+1))
    c = np.zeros(len(arcs)+1);c[0] = 1
    ub[:len(eligible),0] = -1
    oi,mi = {v:i for i,v in enumerate(ops)},{m:i for i,m in enumerate(eligible)}
    for k,(v,opt) in enumerate(arcs,1):
        eq[oi[v],k] = 1
        ub[mi[opt.machine_id],k] = opt.processing_time
        if cap is not None:ub[-1,k] = opt.total_energy
    b = np.zeros(len(ub))
    if cap is not None:b[-1] = cap
    res = linprog(c,A_ub=ub,b_ub=b,A_eq=eq,b_eq=np.ones(len(ops)),bounds=(0,None),method="highs")
    if res.status == 2:return dict(status=2,cap=cap)
    assert res.success, res.message
    d = float(b@res.ineqlin.marginals+np.ones(len(ops))@res.eqlin.marginals)
    stationarity = c-ub.T@res.ineqlin.marginals-eq.T@res.eqlin.marginals-res.lower.marginals
    residuals = dict(equality=float(np.max(abs(eq@res.x-1))),
        inequality=float(max(0,np.max(ub@res.x-b))),duality_gap=abs(float(res.fun)-d),
        stationarity=float(np.max(abs(stationarity))),
        complementarity=float(max(np.max(abs(res.ineqlin.marginals*(b-ub@res.x))),np.max(abs(res.lower.marginals*res.x)))),
        dual_sign=float(max(0,np.max(res.ineqlin.marginals),-np.min(res.lower.marginals))))
    assert max(residuals.values()) < 1e-6, residuals
    return dict(status=0,cap=cap,value=float(res.fun),residuals=residuals,eligible_machines=len(eligible),arcs=len(arcs))


def capacities():
    baseline = json.loads((FROZEN / "replication/results_20261029_capacity/capacity-envelope.json").read_text("utf8"))
    checks = [];max_difference = 0.0
    e0 = make_data("E0")
    for row in baseline["instances"]:
        label = row["instance"];data = make_data(label)
        for key,original in [("unconditioned",row["unconditioned"]),("minimum_energy",row["minimum_energy"])]+[(f"envelope_{i}",x) for i,x in enumerate(row.get("envelope",[]))]:
            cap = original["energy_cap"]
            result = independent_lp(data,cap)
            difference = abs(result["value"]-original["lambda_star"])
            max_difference = max(max_difference,difference)
            assert difference < 1e-6
            checks.append(dict(instance=label,source_record=key,saved_value=original["lambda_star"],difference=difference,**result))
    assert len(checks) == 30
    minimum_energy = sum(min(Decimal(str(o.total_energy)) for o in s.candidates) for s in e0.operations.values())
    assert minimum_energy == Decimal("728.06")
    below = independent_lp(e0,float(minimum_energy)-.001)
    assert below["status"] == 2
    assignment = {};loads = defaultdict(Decimal);separations = []
    for v,s in e0.operations.items():
        candidates = sorted(s.candidates,key=lambda o: Decimal(str(o.total_energy)))
        assert len(candidates)<2 or candidates[0].total_energy != candidates[1].total_energy
        assignment[v] = candidates[0].machine_id
        loads[candidates[0].machine_id] += Decimal(str(candidates[0].processing_time))
        if len(candidates)>1:separations.append(Decimal(str(candidates[1].total_energy))-Decimal(str(candidates[0].total_energy)))
    assert max(loads,key=loads.get) == "M07" and loads["M07"] == Decimal("1304.0")
    row = json.loads((FROZEN / "replication/results_v3/completed/E0_H-MVNS-2_450001_4000.json").read_text("utf8"))
    schedule = decode(e0,row["representative"]["order_sequence"],assignment)
    audit_schedule(e0,schedule,(1304,728.06,1304))
    convex = []
    for a,b in [(728.06,738.06),(738.06,762.59),(762.59,863.07)]:
        fa,fb,fm = [independent_lp(e0,x)["value"] for x in (a,b,(a+b)/2)]
        assert fm <= (fa+fb)/2+TOL
        convex.append(dict(a=a,b=b,value_at_midpoint=fm,endpoint_average=(fa+fb)/2))
    result = dict(lp_records=checks,matched_saved_lps=len(checks),max_saved_value_difference=max_difference,
                  e0_unconditioned=next(x["value"] for x in checks if x["instance"] == "E0" and x["source_record"] == "unconditioned"),
                  e0_at_762_59=next(x["value"] for x in checks if x["instance"] == "E0" and x["cap"] == 762.59),
                  below_minimum_energy=below,minimum_energy=str(minimum_energy),
                  forced_minimum_energy_loads={m:str(x) for m,x in loads.items()},
                  minimum_energy_gap=str(min(separations)),endpoint=(1304,728.06,1304),convexity_checks=convex)
    dump("theory-capacity.json",result)
    return {k:v for k,v in result.items() if k != "lp_records"}


def representatives():
    counts = Counter()
    for file in sorted((FROZEN / "replication/results_v3/completed").glob("*.json")):
        row = json.loads(file.read_text("utf8"));data = make_data(row["instance"])
        entries = {e["operation_id"]:ScheduleEntry(**e) for e in row["representative"]["entries"]}
        audit_schedule(data,Schedule(entries),row["representative"]["objectives"])
        fresh = decode(data,row["representative"]["order_sequence"],row["representative"]["machine_assignment"])
        assert all(vars(fresh.entries[k]) == vars(e) for k,e in entries.items())
        counts[row["instance"]] += 1
    assert sum(counts.values()) == 678
    return dict(total=678,by_instance=dict(counts),feasibility_and_redecoding="PASS")


def small_problem(jobs,windows=None):
    machines = {m:MachineSpec(m,maintenance_windows=w) for m,w in (windows or {"M":[]}).items()}
    data = ProblemData({},machines,{})
    for jid,family,release,p,q,energy in jobs:
        v = jid+"-O1"
        opts = [OperationOption(v,jid,1,m,p,q,0,0,energy) for m in machines]
        op = OperationSpec(v,jid,1,family,opts)
        data.operations[v] = op;data.jobs[jid] = JobSpec(jid,jid,family,release,1000,"",1,[op])
    return data


def boundaries():
    data = small_problem([("U","A",0,1,2,1),("W","A",10,1,2,1),("V","B",3,1,1,1),("T","A",3,2,1,1)])
    # Build two complete legal examples, rather than injecting partial fixed state.
    incompatible = decode(data,["U","W","V","T"],{v:"M" for v in data.operations})
    audit_schedule(data,incompatible)
    assert incompatible.entries["V-O1"].setup_start >= 11
    compatible = decode(data,["U","W","T","V"],{v:"M" for v in data.operations})
    audit_schedule(data,compatible)
    assert compatible.entries["T-O1"].setup_start == 3
    windows = small_problem([("J","A",0,1,1,1)],{"M":[(5,7),(1,3),(2,6),(8,9)]})
    shifted = decode(windows,["J"],{"J-O1":"M"});audit_schedule(windows,shifted)
    assert shifted.entries["J-O1"].setup_start == 9
    # Explicitly demonstrate why the manuscript's floating-point qualification is needed.
    policy = _ObjectivePolicy("multi",(0.999,0.0005,0.0005),(0,0,0),(1,1,1))
    right = Individual([],{});right.objectives=(1,1,1)
    left = Individual([],{});left.objectives=(1+0.9e-9,1-1.1e-9,1)
    assert _score_pair(left,right,policy,policy.weights)
    score_increase = policy.score(left)-policy.score(right)
    assert score_increase > 0
    def restore(payload):
        restored = ProblemData({}, {m:MachineSpec(**attrs) for m,attrs in payload["machines"].items()}, {})
        for oid,attrs in payload["operations"].items():
            spec = dict(attrs);spec["candidates"] = [OperationOption(**o) for o in spec["candidates"]]
            restored.operations[oid] = OperationSpec(**spec)
        for jid,attrs in payload["jobs"].items():
            spec = dict(attrs);spec["operations"] = [restored.operations[o["operation_id"]] for o in spec["operations"]]
            restored.jobs[jid] = JobSpec(**spec)
        return restored
    path = FROZEN / "qa/theory-upgrade-evidence/counterexample-setup-instance.json"
    saved = json.loads(path.read_text("utf8"));example = restore(saved["data"])
    original = decode(example,saved["os"],saved["ma"])
    old = audit_schedule(example,original)
    actual = []
    for move in saved["moves"]:
        changed = decode(example,saved["os"],move["new_ma"])
        new = audit_schedule(example,changed)
        v,a,b = move["operation"],move["source"],move["target"]
        proxy = dict(old["loads"])
        proxy[a] -= original.entries[v].processing_time + original.entries[v].setup_time
        proxy[b] += example.option(v,b).processing_time + example.option(v,b).setup_time
        actual.append(dict(old=old["objectives"][2],predicted=max(proxy.values()),actual=new["objectives"][2],loads=new["loads"]))
    assert (actual[0]["old"],actual[0]["predicted"],actual[0]["actual"]) == (41,39,43)
    assert actual[1]["loads"]["M1"] == 42
    return dict(successor_setup_guard="PASS",touching_and_overlapping_unsorted_windows="PASS",
                setup_proxy_counterexamples=actual,
                floating_tolerance_accepts_scalar_increase=score_increase,
                floating_claim_qualification_required=True)


def horizon_regressions():
    from upgrade_bounds import analytic_bounds
    past = small_problem([("J","A",0,1,0,1)],{"M":[(-10,-5)]})
    past_bounds = analytic_bounds(past)
    assert past_bounds["upper"][0] == 1, past_bounds
    schedule = decode(past,["J"],{"J-O1":"M"})
    audit_schedule(past,schedule,(1,1,1))
    empty = small_problem([("J","A",0,1,0,1)])
    delayed = Schedule({"J-O1":ScheduleEntry("J-O1","J",1,"M",100,100,101,1,0,1)})
    audit_schedule(empty,delayed,(101,1,1))
    upper = analytic_bounds(empty)["upper"][0]
    assert upper == 1 and delayed.makespan > upper
    return dict(negative_window=dict(window=[-10,-5],decoder_makespan=1,clamped_upper=1),
                arbitrary_delay=dict(constructed_diagnostic=True,valid_model_schedule=True,decoder_upper=upper,
                                     feasible_delayed_makespan=delayed.makespan),
                required_scope="Upper horizon bounds decoder outputs, not arbitrary feasible schedules")


def main():
    started = time.perf_counter()
    result = dict(status="RUNNING",python=sys.version,platform=platform.platform())
    result["boundaries"] = boundaries()
    result["horizon_regressions"] = horizon_regressions()
    result["single_moves"] = moves()
    result["prefix_equivalence"] = prefixes()
    result["capacity"] = capacities()
    result["archived_representatives"] = representatives()
    result["elapsed_s"] = time.perf_counter()-started
    result["status"] = "PASS"
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump("theory-summary.json",result)
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__ == "__main__":
    main()
