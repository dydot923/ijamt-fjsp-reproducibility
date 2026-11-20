"""Independent machine-order enumeration, decoder reachability, real small runs."""
import itertools,json,random,time,hashlib
from pathlib import Path
import numpy as np
from fjsp.dataset_io import load_any_dataset
from fjsp.decoder import decode
from fjsp.objectives import static_objectives
from fjsp.config import AlgorithmConfig
import run_budget_study as study
import fjsp.nsga_vns as module
from run_matched_study import instrument_matched
from prefix_evaluator import make_evaluator
from upgrade_bounds import analytic_bounds
from analyze_budget_study import nondominated,hv3,igdplus

OUT=Path(__file__).parent/'results_v3/oracle'

def construct(index):
    rng=random.Random(520001+index)
    nj,no=(3,2) if index<3 else (2,3)
    jobs=[dict(job_id=f'J{j+1}',family=f'F{j+1}',release_time=0,due_date=1000) for j in range(nj)]
    arcs=[dict(job_id=j['job_id'],op_index=k+1,machine_id=f'M{m+1}',processing_time=rng.randint(1,9),
               setup_time=0,running_energy=rng.randint(1,12),auxiliary_energy=0)
          for j in jobs for k in range(no) for m in range(2)]
    return dict(jobs=jobs,machines=[dict(machine_id='M1'),dict(machine_id='M2')],operations=arcs,
                metadata=dict(seed=520001+index,constructed_validation_not_measured=True))

def orientations(data):
    """No decoder used: all assignment/machine sequence DAG earliest schedules."""
    ids=data.operation_ids;n=len(ids);index={v:i for i,v in enumerate(ids)}
    job_edges=[(index[a.operation_id],index[b.operation_id]) for j in data.jobs.values()
               for a,b in zip(j.operations,j.operations[1:])]
    objectives=set();total=acyclic=0
    for machines in itertools.product(data.machine_ids,repeat=n):
        processing=[data.option(v,m).processing_time for v,m in zip(ids,machines)]
        energy=sum(data.option(v,m).total_energy for v,m in zip(ids,machines))
        load=max(sum(p for p,m in zip(processing,machines) if m==machine) for machine in data.machine_ids)
        groups=[[i for i,m in enumerate(machines) if m==machine] for machine in data.machine_ids]
        for orders in itertools.product(*(itertools.permutations(g) for g in groups)):
            total+=1
            edges=set(job_edges+[(a,b) for order in orders for a,b in zip(order,order[1:])])
            successors=[[] for _ in ids];degree=[0]*n;start=[0.]*n
            for a,b in edges:successors[a].append(b);degree[b]+=1
            ready=[i for i,d in enumerate(degree) if not d];visited=0;maximum=0.
            while ready:
                a=ready.pop();visited+=1;end=start[a]+processing[a];maximum=max(maximum,end)
                for b in successors[a]:
                    start[b]=max(start[b],end);degree[b]-=1
                    if degree[b]==0:ready.append(b)
            if visited==n:acyclic+=1;objectives.add((maximum,energy,load))
    return nondominated(list(objectives)),dict(orientations=total,acyclic=acyclic,unique_objectives=len(objectives))

def sequences(data):
    remaining={j:len(data.jobs[j].operations) for j in data.job_ids}
    def visit(prefix):
        if len(prefix)==len(data.operations):yield prefix;return
        for j in data.job_ids:
            if remaining[j]:
                remaining[j]-=1
                yield from visit(prefix+[j])
                remaining[j]+=1
    yield from visit([])

def reachable(data):
    points=set();count=0
    assignments=[dict(zip(data.operation_ids,ma)) for ma in itertools.product(data.machine_ids,repeat=len(data.operations))]
    for os in sequences(data):
        for ma in assignments:
            schedule=decode(data,os,ma);f=static_objectives(data,schedule)
            assert not study.check_schedule(data,schedule,f)
            count+=1;points.add(f)
    return nondominated(list(points)),count

def optimize(data,method,seed):
    cfg=AlgorithmConfig(seed=seed,verbose=False,generations=10000,population_size=40,
        restart_ratio=0,evaluation_budget=200,initialization_mode='hybrid')
    if method=='H-NSGA':cfg.vns_probability=0;cfg.elite_local_search_ratio=0
    if method=='H-MVNS-2':cfg.local_search_trials=2
    old_neighbor=instrument_matched() if method!='H-NSGA' else study.instrument_neighbor()
    old_evaluate=study._evaluate
    count=dict(full_calls=0,prefix_calls=0,reused_operations=0,reinserted_operations=0)
    study._evaluate=make_evaluator(count)
    monitor=study.Monitor(data,200);solver=study.BudgetSolver(data,cfg,monitor)
    started=time.perf_counter()
    try:solver.run()
    except study.BudgetEnd:pass
    finally:module._propose_neighbor=old_neighbor;study._evaluate=old_evaluate
    assert monitor.evaluations==200
    assert not any(study.check_schedule(data,p.schedule,p.objectives) for p in monitor.points)
    return dict(method=method,seed=seed,evaluations=monitor.evaluations,
                front=[p.objectives for p in monitor.points],audited_schedules=len(monitor.points),
                invalid_schedules=0,runtime_s=time.perf_counter()-started,decoder_counts=count)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[]
    for i in range(6):
        path=OUT/f'T{i+1}-input.json';payload=construct(i)
        if path.exists():assert json.loads(path.read_text(encoding='utf8'))==payload
        else:path.write_text(json.dumps(payload,indent=2),encoding='utf8')
        inputs.append(path)
    design=dict(cases=6,operations=6,machines=2,input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                methods=['H-NSGA','H-MVNS-6','H-MVNS-2'],seeds=list(range(530001,530009)),budget=200)
    (OUT/'design.json').write_text(json.dumps(design,indent=2),encoding='utf8')
    summary=[];runs=[]
    for path in inputs:
        data=load_any_dataset(path);exact,counts=orientations(data);decoded,encodings=reachable(data)
        b=analytic_bounds(data);lo=np.array(b['lower']);scale=np.array(b['reference'])-lo
        reference=(exact-lo)/scale;decoder_hv=hv3((decoded-lo)/scale);optimal_hv=hv3(reference)
        assert decoder_hv<=optimal_hv+1e-12
        item=dict(instance=path.stem.replace('-input',''),bounds=b,exact_front=exact.tolist(),
                  decoder_front=decoded.tolist(),encodings=encodings,exact_hv=optimal_hv,
                  decoder_hv=decoder_hv,fronts_equal=np.array_equal(exact,decoded),**counts)
        summary.append(item)
        for method in design['methods']:
            for seed in design['seeds']:
                row=optimize(data,method,seed);array=np.asarray(row['front']);norm=(array-lo)/scale
                row.update(instance=item['instance'],hv=hv3(norm),igdplus=igdplus(norm,reference),
                    hv_ratio=hv3(norm)/optimal_hv,
                    exact_coverage=sum(any(np.allclose(p,q,atol=1e-8,rtol=0) for p in array) for q in exact)/len(exact))
                runs.append(row)
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
        (OUT/'runs.json').write_text(json.dumps(runs,indent=2),encoding='utf8')
        print(item['instance'],counts,'exactPF',len(exact),'decoderEqual',item['fronts_equal'],'runs',len(runs),flush=True)
    assert len(runs)==144

if __name__=='__main__':main()
