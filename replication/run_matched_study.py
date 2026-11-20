"""Exploratory revision: match machine neighborhood to local objective weights.

Only the machine-neighborhood generator differs from frozen H-VNS. All budgets,
initialization, acceptance, operators, and monitoring remain shared.
"""
import argparse,json,random
from concurrent.futures import ProcessPoolExecutor,as_completed
import run_budget_study as study
import fjsp.nsga_vns as module

def matched_machine(current,data,weights,rng):
    candidate=current.clone(); candidate.schedule=None;candidate.objectives=()
    focus=max(range(3),key=lambda k:weights[k])
    entries=list(current.schedule.entries.values())
    loads={k:sum(e.duration for e in current.schedule.entries_for_machine(k)) for k in data.machine_ids}
    moves=[]
    if focus==1:
        for entry in entries:
            for opt in data.operations[entry.operation_id].candidates:
                if opt.machine_id!=entry.machine_id:
                    gain=entry.energy-opt.total_energy
                    if gain>1e-9:
                        moves.append((gain,entry.operation_id,opt.machine_id))
        if moves:
            moves=sorted(moves,reverse=True)[:5]
            _,op,ma=rng.choice(moves)
        else:
            op=rng.choice(data.operation_ids)
            alternatives=[o for o in data.operations[op].candidates if o.machine_id!=current.machine_assignment[op]]
            if not alternatives:return candidate
            ma=min(alternatives,key=lambda o:(o.total_energy,o.processing_time)).machine_id
    elif focus==2:
        bottleneck=max(loads,key=loads.get)
        for entry in current.schedule.entries_for_machine(bottleneck):
            for opt in data.operations[entry.operation_id].candidates:
                if opt.machine_id==bottleneck:continue
                predicted=dict(loads)
                predicted[bottleneck]-=entry.duration
                predicted[opt.machine_id]+=opt.processing_time+opt.setup_time
                moves.append((max(predicted.values()),opt.total_energy,entry.operation_id,opt.machine_id))
        if not moves:return candidate
        moves=sorted(moves)[:5]
        _,_,op,ma=rng.choice(moves)
    else:
        entry=max(entries,key=lambda e:e.end);op=entry.operation_id
        alternatives=[o for o in data.operations[op].candidates if o.machine_id!=entry.machine_id]
        if not alternatives:return candidate
        ma=min(alternatives,key=lambda o:(o.processing_time+o.setup_time,o.total_energy)).machine_id
    candidate.machine_assignment[op]=ma
    return candidate

def instrument_matched():
    original=module._propose_neighbor
    def wrapped(current,neighborhood,data,policy,weights,rng):
        if neighborhood=='machine':candidate=matched_machine(current,data,weights,rng)
        else:candidate=original(current,neighborhood,data,policy,weights,rng)
        candidate._source=neighborhood;candidate._base=current;candidate._acceptance=(policy,weights)
        return candidate
    module._propose_neighbor=wrapped
    return original

def run_one(task):
    if task[1]=='H-MVNS':study.instrument_neighbor=instrument_matched
    return study.run_one(task)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',default='matched');parser.add_argument('--workers',type=int,default=3);a=parser.parse_args()
    if a.group=='matched':
        tasks=[('E0','H-MVNS',s,4000) for s in range(410001,410031)]
        tasks +=[(f'Mk{i:02d}','H-MVNS',s,4000) for i in range(1,11) for s in range(420001,420011)]
        tasks +=[(i,'H-MVNS',s,4000) for i in ['E1','E2'] for s in range(430001,430013)]
    else:
        tasks=[(f'Mk{i:02d}','H-NSGA',s,4000) for i in range(1,11) for s in range(420001,420011)]
    name='matched' if a.group=='matched' else 'benchmark_ablation'
    path=study.OUT/(name+'_runs.json');done=json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    keys={(r['instance'],r['method'],r['seed'],r['budget']) for r in done};tasks=[t for t in tasks if t not in keys]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(run_one,t) for t in tasks]
        for fut in as_completed(fs):
            r=fut.result();done.append(r);path.write_text(json.dumps(done,ensure_ascii=False),encoding='utf-8')
            print(f'{name} {len(done)} {r["instance"]} {r["method"]} seed={r["seed"]} eval={r["evaluations"]} sec={r["runtime_s"]:.2f} invalid={r["invalid_schedules"]}',flush=True)
if __name__=='__main__':main()
