"""Executable mathematical audit; never modifies solver, manuscript, or results.
Run: python qa/theory-upgrade-evidence/verify_theory.py
"""
import hashlib, json, math, random, sys, time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'replication'))
from run_budget_study import make_data
from run_stress import check_schedule
from fjsp.nsga_vns import _build_order_sequence, _build_machine_assignment
from fjsp.decoder import decode, DecodeContext, _sequence_to_operation_ids, repair_machine_assignment, _find_earliest_slot
from fjsp.objectives import static_objectives
from fjsp.data_model import ProblemData, MachineSpec, JobSpec, OperationSpec, OperationOption, ScheduleEntry


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def loads(data, schedule):
    return {m: sum(e.duration for e in schedule.entries_for_machine(m)) for m in data.machine_ids}


def prefix_cut(data, old_os, new_os, old_ma, new_ma):
    """Static valid complete encodings. Compare effective repaired assignments."""
    old_ids = _sequence_to_operation_ids(data, old_os)
    new_ids = _sequence_to_operation_ids(data, new_os)
    old_assignment = repair_machine_assignment(data, old_ma)
    new_assignment = repair_machine_assignment(data, new_ma)
    cut = 0
    for old_id, new_id in zip(old_ids, new_ids):
        if old_id != new_id or old_assignment[old_id] != new_assignment[new_id]:
            break
        cut += 1
    return cut, new_ids


def prefix_decode(data, base, old_os, new_os, old_ma, new_ma):
    cut, new_ids = prefix_cut(data, old_os, new_os, old_ma, new_ma)
    context = DecodeContext(
        fixed_entries={v: base.entries[v] for v in new_ids[:cut]},
        active_operation_ids=set(new_ids[cut:]),
    )
    return decode(data, new_os, new_ma, context), cut


def prefix_bounds(data, base, new_ids, new_ma, cut):
    prefix = set(new_ids[:cut])
    lower, upper = {}, {}
    for machine in data.machine_ids:
        existing = [base.entries[v] for v in prefix if base.entries[v].machine_id == machine]
        q_prefix = sum(e.setup_time for e in existing)
        old_families = {data.jobs[e.job_id].family for e in existing}
        suffix = [v for v in new_ids[cut:] if new_ma[v] == machine]
        family_min = {}
        for v in suffix:
            family = data.jobs[data.operations[v].job_id].family
            setup = data.option(v, machine).setup_time
            if family not in old_families:
                family_min[family] = min(family_min.get(family, math.inf), setup)
        processing = sum(data.option(v, machine).processing_time for v in new_ids if new_ma[v] == machine)
        lower[machine] = processing + q_prefix + sum(family_min.values())
        upper[machine] = processing + q_prefix + sum(data.option(v, machine).setup_time for v in suffix)
    return lower, upper


def entry_equal(left, right):
    return {v: vars(e) for v, e in left.entries.items()} == {v: vars(e) for v, e in right.entries.items()}


def run_single_moves():
    count = Counter()
    max_energy_error = max_zero_error = max_proxy_error = 0.0
    parents = []
    false_examples = []
    with (OUT / 'single-move-raw.jsonl').open('w', encoding='utf-8') as stream:
        for scenario in ['E0', 'E1', 'E2']:
            data = make_data(scenario)
            for index in range(8):
                seed = 551001 + index
                rng = random.Random(seed)
                os = _build_order_sequence(data, 'random', rng)
                ma = _build_machine_assignment(data, 'random', rng)
                base = decode(data, os, ma)
                old_f = static_objectives(data, base)
                old_loads = loads(data, base)
                ids = _sequence_to_operation_ids(data, os)
                positions = {v: k for k, v in enumerate(ids)}
                assert not check_schedule(data, base, old_f)
                parents.append(dict(scenario=scenario, seed=seed, os=os, ma=ma,
                                    objectives=old_f, entries=[asdict(e) for e in base.entries.values()]))
                for v in data.operation_ids:
                    entry = base.entries[v]
                    source = entry.machine_id
                    for option in data.operations[v].candidates:
                        target = option.machine_id
                        if target == source:
                            continue
                        new_ma = dict(ma)
                        new_ma[v] = target
                        new = decode(data, os, new_ma)
                        new_f = static_objectives(data, new)
                        new_loads = loads(data, new)
                        count[scenario + '_moves'] += 1
                        issues = check_schedule(data, new, new_f)
                        assert not issues
                        count['feasible_moves'] += 1
                        energy_error = abs((new_f[1] - old_f[1]) - (option.total_energy - entry.energy))
                        assert energy_error < 1e-8
                        max_energy_error = max(max_energy_error, energy_error)
                        predicted = dict(old_loads)
                        predicted[source] -= entry.duration
                        predicted[target] += option.processing_time + option.setup_time
                        error = max(predicted.values()) - new_f[2]
                        max_proxy_error = max(max_proxy_error, abs(error))
                        if scenario == 'E0':
                            max_zero_error = max(max_zero_error, abs(error))
                            assert abs(error) < 1e-8
                        else:
                            count['setup_proxy_nonexact'] += int(abs(error) > 1e-8)
                            if max(predicted.values()) < old_f[2] - 1e-8 and new_f[2] >= old_f[2] - 1e-8:
                                count['false_predicted_load_improvement'] += 1
                                false_examples.append(dict(scenario=scenario, seed=seed, operation=v, source=source,
                                                           target=target, old=old_f[2], predicted=max(predicted.values()), true=new_f[2]))
                        cut = positions[v]
                        for u in ids[:cut]:
                            assert vars(base.entries[u]) == vars(new.entries[u])
                        lower, upper = prefix_bounds(data, base, ids, new_ma, cut)
                        for machine in data.machine_ids:
                            assert lower[machine] - 1e-8 <= new_loads[machine] <= upper[machine] + 1e-8
                        assert max(lower.values()) - 1e-8 <= new_f[2] <= max(upper.values()) + 1e-8
                        count['prefix_interval_passes'] += 1
                        stream.write(json.dumps(dict(scenario=scenario, seed=seed, operation=v, source=source,
                            target=target, cut=cut, old_objectives=old_f, new_objectives=new_f, old_loads=old_loads,
                            new_loads=new_loads, predicted_loads=predicted, lower=lower, upper=upper,
                            energy_increment_error=energy_error, proxy_error=error, independent_violations=issues), ensure_ascii=False) + '\n')
    dump(OUT / 'parents.json', parents)
    return dict(counts=dict(count), max_energy_error=max_energy_error, max_zero_error=max_zero_error,
                max_proxy_error=max_proxy_error, false_improvement_examples=false_examples)


def run_prefix_equivalence():
    # Reproduce original 450 MA-only pairs exactly.
    n = suffix_operations = 0
    records = []
    for scenario in ['E0', 'E1', 'E2']:
        data = make_data(scenario)
        for index in range(3):
            seed = 771001 + index
            rng = random.Random(seed)
            os = _build_order_sequence(data, 'random', rng)
            ma = _build_machine_assignment(data, 'random', rng)
            base = decode(data, os, ma)
            ids = _sequence_to_operation_ids(data, os)
            moves = [(v, z.machine_id) for v in ids for z in data.operations[v].candidates if z.machine_id != ma[v]]
            for v, target in rng.sample(moves, 50):
                new_ma = dict(ma)
                new_ma[v] = target
                full = decode(data, os, new_ma)
                reused, cut = prefix_decode(data, base, os, os, ma, new_ma)
                assert full.feasible == reused.feasible and entry_equal(full, reused)
                assert static_objectives(data, full) == static_objectives(data, reused)
                assert not check_schedule(data, reused, static_objectives(data, reused))
                n += 1
                suffix_operations += len(ids) - cut
                records.append(dict(scenario=scenario, seed=seed, operation=v, target=target, cut=cut,
                                    suffix_operations=len(ids)-cut, equal=True, objectives=static_objectives(data, full)))
    with (OUT / 'prefix-ma-only-raw.jsonl').open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return dict(prefix_reuse_equal_pairs=n, suffix_operations_total=suffix_operations,
                full_operations_total=n * 58, independent_violations=0)


def run_general_cut():
    count = Counter()
    with (OUT / 'prefix-general-raw.jsonl').open('w', encoding='utf-8') as stream:
        for scenario in ['E0', 'E1', 'E2']:
            data = make_data(scenario)
            for index in range(3):
                seed = 991001 + index
                rng = random.Random(seed)
                old_os = _build_order_sequence(data, 'random', rng)
                old_ma = _build_machine_assignment(data, 'random', rng)
                base = decode(data, old_os, old_ma)
                old_ids = _sequence_to_operation_ids(data, old_os)
                for trial in range(84):
                    new_os, new_ma = list(old_os), dict(old_ma)
                    mode = ['unchanged', 'first_ma', 'last_ma', 'os_swap', 'os_insert', 'mixed', 'multiple_ma'][trial % 7]
                    if mode in ['os_swap', 'mixed']:
                        i, j = rng.sample(range(len(new_os)), 2)
                        new_os[i], new_os[j] = new_os[j], new_os[i]
                    if mode == 'os_insert':
                        i, j = rng.sample(range(len(new_os)), 2)
                        value = new_os.pop(i)
                        new_os.insert(j, value)
                    if mode == 'first_ma':
                        targets = [old_ids[0]]
                    elif mode == 'last_ma':
                        targets = [old_ids[-1]]
                    elif mode in ['mixed', 'multiple_ma']:
                        targets = rng.sample(data.operation_ids, 3)
                    else:
                        targets = []
                    for v in targets:
                        alternatives = [z.machine_id for z in data.operations[v].candidates if z.machine_id != new_ma[v]]
                        if alternatives:
                            new_ma[v] = rng.choice(alternatives)
                    full = decode(data, new_os, new_ma)
                    reused, cut = prefix_decode(data, base, old_os, new_os, old_ma, new_ma)
                    assert full.feasible == reused.feasible and entry_equal(full, reused)
                    assert static_objectives(data, full) == static_objectives(data, reused)
                    issues = check_schedule(data, reused, static_objectives(data, reused))
                    assert not issues
                    new_ids = _sequence_to_operation_ids(data, new_os)
                    lower, upper = prefix_bounds(data, base, new_ids, new_ma, cut)
                    new_loads = loads(data, full)
                    for m in data.machine_ids:
                        assert lower[m] - 1e-8 <= new_loads[m] <= upper[m] + 1e-8
                    count[mode] += 1
                    count['all_pairs'] += 1
                    count['cut_zero'] += int(cut == 0)
                    count['cut_full'] += int(cut == len(new_ids))
                    stream.write(json.dumps(dict(scenario=scenario, seed=seed, trial=trial, mode=mode, old_os=old_os,
                        new_os=new_os, old_ma=old_ma, new_ma=new_ma, cut=cut, equal=True,
                        objectives=static_objectives(data, full), lower=lower, upper=upper,
                        independent_violations=issues), ensure_ascii=False) + '\n')
    return dict(count)


def construct_counterexamples():
    rng = random.Random(881008)
    data = ProblemData({}, {m: MachineSpec(m) for m in ['M1', 'M2', 'M3']}, {})
    for j in range(4):
        jid = 'J' + str(j)
        family = ['A', 'B', 'A', 'C'][j]
        operations = []
        for k in range(1, 4):
            oid = jid + '-O' + str(k)
            options = [OperationOption(oid, jid, k, m, rng.randint(1,10), rng.randint(1,8), 0, 0,
                                       rng.randint(1,10)) for m in data.machine_ids]
            operation = OperationSpec(oid, jid, k, family, options)
            operations.append(operation)
            data.operations[oid] = operation
        data.jobs[jid] = JobSpec(jid, jid, family, 0, 1000, '', 1, operations)
    os = [j for j in data.job_ids for _ in range(3)]
    rng.shuffle(os)
    ma = {v: rng.choice(data.machine_ids) for v in data.operation_ids}
    base = decode(data, os, ma)
    L = loads(data, base)
    examples = []
    for v, target in [('J2-O2', 'M3'), ('J0-O1', 'M3')]:
        entry = base.entries[v]
        option = data.option(v, target)
        new_ma = dict(ma)
        new_ma[v] = target
        new = decode(data, os, new_ma)
        predicted = dict(L)
        predicted[entry.machine_id] -= entry.duration
        predicted[target] += option.processing_time + option.setup_time
        cut, ids = prefix_cut(data, os, os, ma, new_ma)
        lower, upper = prefix_bounds(data, base, ids, new_ma, cut)
        assert not check_schedule(data, new, static_objectives(data, new))
        examples.append(dict(operation=v, source=entry.machine_id, target=target, new_ma=new_ma,
            cut=cut, old_loads=L, predicted_loads=predicted, true_loads=loads(data,new), lower=lower, upper=upper,
            old_objectives=static_objectives(data,base), new_objectives=static_objectives(data,new),
            new_entries=[asdict(e) for e in new.entries.values()]))
    assert max(examples[0]['predicted_loads'].values()) == 39
    assert max(examples[0]['true_loads'].values()) == 43
    assert examples[1]['true_loads']['M1'] == 42
    dump(OUT / 'counterexample-setup-instance.json', dict(seed=881008, data=asdict(data), os=os, ma=ma,
         base_entries=[asdict(e) for e in base.entries.values()], moves=examples))
    one = ProblemData({}, {'A': MachineSpec('A'), 'B': MachineSpec('B',maintenance_windows=[(0,100)])}, {})
    options = [OperationOption('J-O1','J',1,'A',10,0,0,0,10), OperationOption('J-O1','J',1,'B',1,0,0,0,1)]
    operation = OperationSpec('J-O1','J',1,'F',options)
    one.operations[operation.operation_id] = operation
    one.jobs['J'] = JobSpec('J','J','F',0,999,'',1,[operation])
    old = decode(one,['J'],{'J-O1':'A'})
    new = decode(one,['J'],{'J-O1':'B'})
    assert (old.makespan,new.makespan)==(10,101)
    dump(OUT / 'counterexample-cmax-instance.json', dict(data=asdict(one), os=['J'], old_ma={'J-O1':'A'},
         new_ma={'J-O1':'B'}, old_entries=[asdict(e) for e in old.entries.values()],
         new_entries=[asdict(e) for e in new.entries.values()], old_cmax=10, new_cmax=101))
    gap = ProblemData({}, {'M':MachineSpec('M')}, {})
    for jid, family, setup in [('U','A',2),('W','A',2),('V','B',1),('T','A',1)]:
        option = OperationOption(jid+'-O1',jid,1,'M',1 if jid!='T' else 2,setup,0,0,1)
        operation = OperationSpec(option.operation_id,jid,1,family,[option])
        gap.operations[operation.operation_id] = operation
        gap.jobs[jid] = JobSpec(jid,jid,family,0,999,'',1,[operation])
    u = ScheduleEntry('U-O1','U',1,'M',0,2,3,1,2,1)
    w = ScheduleEntry('W-O1','W',1,'M',10,10,11,1,0,1)
    blocked = _find_earliest_slot(gap,gap.option('V-O1','M'),3,[u,w],[])
    accepted = _find_earliest_slot(gap,gap.option('T-O1','M'),3,[u,w],[])
    assert blocked==(11,12,13,1) and accepted==(3,3,5,0)
    dump(OUT / 'successor-setup-guard.json', dict(data=asdict(gap),existing=[asdict(u),asdict(w)],ready=3,
         incompatible_family_slot=blocked,compatible_family_slot=accepted))
    return dict(setup_false_improvement=dict(old=41,predicted=39,true=43), third_machine_change=dict(old=41,true=42),
                cmax_counterexample=dict(old=10,new=101),successor_guard=dict(rejected_gap=blocked,accepted_gap=accepted))


def main():
    started = time.perf_counter()
    summary = dict(single_moves=run_single_moves(),ma_only_prefix=run_prefix_equivalence(),
                   general_prefix=run_general_cut(),counterexamples=construct_counterexamples())
    summary['elapsed_s'] = time.perf_counter()-started
    summary['python'] = sys.version
    summary['source_sha256'] = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
        'replication/fjsp/decoder.py','replication/fjsp/data_model.py','replication/fjsp/objectives.py',
        'replication/fjsp/nsga_vns.py','replication/run_matched_study.py','replication/run_budget_study.py',
        'replication/run_stress.py','replication/data/v42_dataset.json']}
    summary['verification_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['formal_performance_experiment'] = False
    summary['status'] = 'all assertions passed'
    dump(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
