"""Prospective independent budget/local-intensity confirmation; frozen grid.

Writes one completed-run JSON per task, atomically, and a resumable aggregate.
Old v2 logs are read-only. Parallel runtimes are not algorithm speed evidence.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from dataclasses import asdict
import argparse,json,time,hashlib
import numpy as np
import run_budget_study as study
import fjsp.nsga_vns as module
from fjsp.config import AlgorithmConfig
from run_matched_study import instrument_matched
from reference_survival import make_reference_survival
from prefix_evaluator import make_evaluator

OUT=Path(__file__).resolve().parent/'results_v3'
INSTANCES=['E0']+[f'Mk{i:02d}' for i in range(1,11)]
METHODS=['H-NSGA','H-MVNS-6','H-MVNS-2']
SEEDS=list(range(450001,450007))
BUDGETS=[2000,4000,8000]

def one(task):
    instance,method,seed,budget=task
    data=study.make_data(instance)
    cfg=AlgorithmConfig(seed=seed,verbose=False,generations=10000,population_size=40,
        restart_ratio=0,evaluation_budget=budget,initialization_mode='hybrid')
    if method in ['H-NSGA','RP-NSGA']:cfg.vns_probability=0;cfg.elite_local_search_ratio=0
    if method=='H-MVNS-2':cfg.local_search_trials=2
    original_neighbor=(instrument_matched() if method.startswith('H-MVNS') else study.instrument_neighbor())
    original_survival=module.environmental_selection
    original_evaluate=study._evaluate
    decode_counts=dict(full_calls=0,prefix_calls=0,reused_operations=0,reinserted_operations=0)
    study._evaluate=make_evaluator(decode_counts)
    if method=='RP-NSGA':module.environmental_selection=make_reference_survival(seed)
    study.CHECKPOINTS=[p for p in [40,500,1000,2000,3000,4000,6000,8000] if p<=budget]
    monitor=study.Monitor(data,budget)
    solver=study.BudgetSolver(data,cfg,monitor)
    started=time.perf_counter()
    try:solver.run()
    except study.BudgetEnd:pass
    finally:
        module._propose_neighbor=original_neighbor
        module.environmental_selection=original_survival
        study._evaluate=original_evaluate
    if monitor.evaluations!=budget:raise RuntimeError((task,'budget not exhausted',monitor.evaluations))
    issues=[study.check_schedule(data,p.schedule,p.objectives) for p in monitor.points]
    assert not any(issues),(task,issues)
    array=np.array([p.objectives for p in monitor.points]);minimum=array.min(axis=0);maximum=array.max(axis=0)
    idx=int(np.argmin(np.linalg.norm((array-minimum)/np.maximum(1,maximum-minimum),axis=1)))
    representative=monitor.points[idx]
    result=dict(instance=instance,method=method,seed=seed,budget=budget,evaluations=monitor.evaluations,
        runtime_s=time.perf_counter()-started,cache_hits=monitor.hits,generations=len(solver.history),
        config=asdict(cfg),front=array.tolist(),snapshots=monitor.snapshots,local=monitor.local,
        decoder_counts=decode_counts,
        audited_schedules=len(issues),invalid_schedules=0,representative=dict(
            objectives=list(representative.objectives),order_sequence=representative.order_sequence,
            machine_assignment=representative.machine_assignment,
            entries=[asdict(e) for e in representative.schedule.entries.values()]))
    target=OUT/'completed'/(f'{instance}_{method}_{seed}_{budget}.json')
    temp=target.with_suffix('.tmp');temp.write_text(json.dumps(result,ensure_ascii=False),encoding='utf8');temp.replace(target)
    return dict(instance=instance,method=method,seed=seed,budget=budget,runtime_s=result['runtime_s'],
                invalid=0)

def grid(group):
    if group=='rp':return [(i,'RP-NSGA',s,4000) for i in INSTANCES for s in SEEDS]
    tasks=[(i,m,s,b) for i in INSTANCES for m in METHODS for s in SEEDS for b in BUDGETS]
    tasks += [('E0',m,s,4000) for m in METHODS for s in range(450007,450013)]
    return tasks

def aggregate():
    allrows=[json.loads(p.read_text(encoding='utf8')) for p in sorted((OUT/'completed').glob('*.json'))]
    for name,rows in [('confirm',[r for r in allrows if r['method']!='RP-NSGA']),
                      ('reference_point',[r for r in allrows if r['method']=='RP-NSGA'])]:
        (OUT/(name+'_runs.json')).write_text(json.dumps(rows,ensure_ascii=False),encoding='utf8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',choices=['confirm','rp','all'],default='all')
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    OUT.mkdir(exist_ok=True);(OUT/'completed').mkdir(exist_ok=True)
    tasks=grid('confirm')+grid('rp') if args.group=='all' else grid(args.group)
    lock=Path(__file__).resolve().parents[1]/'plan/upgrade-experiment-lock.md'
    manifest=dict(protocol_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),tasks=tasks,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_survival_sha256=hashlib.sha256((Path(__file__).parent/'reference_survival.py').read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256((Path(__file__).parent/'prefix_evaluator.py').read_bytes()).hexdigest(),
        bounds_sha256=hashlib.sha256((OUT/'analytic-bounds.json').read_bytes()).hexdigest(),
        workers=args.workers,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    path=OUT/'run-design.json'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf8'))
        assert previous['tasks']==[list(t) for t in tasks] and previous['protocol_sha256']==manifest['protocol_sha256']
        for key in ['script_sha256','reference_survival_sha256','evaluator_sha256','bounds_sha256']:
            assert previous[key]==manifest[key],('frozen implementation changed',key)
    else:path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    missing=[t for t in tasks if not (OUT/'completed'/f'{t[0]}_{t[1]}_{t[2]}_{t[3]}.json').exists()]
    # Small/large tasks alternate so monitoring is not dominated by a single size.
    missing.sort(key=lambda t:(t[3],t[2],t[0],t[1]))
    done=len(tasks)-len(missing)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(one,t):t for t in missing}
        for future in as_completed(futures):
            row=future.result();done+=1
            if done%6==0 or done==len(tasks):aggregate()
            print(f'{done}/{len(tasks)} {row["instance"]} {row["method"]} seed={row["seed"]} budget={row["budget"]} sec={row["runtime_s"]:.2f} invalid={row["invalid"]}',flush=True)
    aggregate()

if __name__=='__main__':main()
