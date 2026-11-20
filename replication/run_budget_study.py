"""Controlled 2026 revision: exact unique decode budget, all-visited Pareto archive.

The legacy source is preserved outside this directory. Checkpoints contain actual
solver outputs; no performance values are fabricated. Extended benchmark energy
is an explicit dimensionless coefficient, not measured plant energy.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import numpy as np
from fjsp.config import AlgorithmConfig
from fjsp.dataset_io import load_any_dataset
from fjsp.decoder import repair_machine_assignment
from fjsp.nsga_vns import (IGA_NSGA_VNS, Individual, initialize_population,
    _individual_signature, _evaluate, crossover, mutate)
import fjsp.nsga_vns as module
from fjsp.objectives import dominates
from run_stress import make_scenario, check_schedule

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_v2'
DATA = HERE / 'data'
VARIANTS = ['R-NSGA', 'H-NSGA', 'R-VNS', 'H-VNS', 'H-VNS-R', 'MOEAD']
BUDGET = 4000
CHECKPOINTS = [40, 500, 1000, 2000, 3000, 4000]

class BudgetEnd(Exception):
    pass

class Monitor:
    def __init__(self, data, budget):
        self.data, self.budget = data, budget
        self.evaluations = 0
        self.points = []
        self.snapshots = []
        self.sources = {}
        self.cache = {}
        self.hits = 0
        self.local = {name: dict(proposals=0, fresh=0, accepted=0, dominates=0)
                      for name in ['insert', 'critical_block', 'machine', 'swap']}
        self.started = time.perf_counter()

    def evaluate(self, ind):
        ind.machine_assignment = repair_machine_assignment(self.data, ind.machine_assignment)
        sig = _individual_signature(ind, self.data)
        cached = self.cache.get(sig)
        if cached is None and self.evaluations >= self.budget:
            raise BudgetEnd()
        if cached:
            ind.schedule = cached.schedule.copy()
            ind.objectives = cached.objectives
            self.hits += 1
        else:
            _evaluate(ind, self.data, None, None)
            self.evaluations += 1
            self.cache[sig] = ind.clone()
        name = getattr(ind, '_source', '')
        if name in self.local:
            row = self.local[name]
            row['proposals'] += 1
            row['fresh'] += int(cached is None)
            base = getattr(ind, '_base', None)
            policy, weights = getattr(ind, '_acceptance', (None, None))
            if base:
                row['dominates'] += int(dominates(ind.objectives, base.objectives))
                row['accepted'] += int(module._score_pair(ind, base, policy, weights))
        if cached is None:
            if not any(p.objectives == ind.objectives or dominates(p.objectives, ind.objectives) for p in self.points):
                self.points = [p for p in self.points if not dominates(ind.objectives, p.objectives)]
                self.points.append(ind.clone())
                self.sources[ind.objectives] = name or 'genetic/initial'
            if self.evaluations in CHECKPOINTS:
                self.snapshots.append(dict(evaluations=self.evaluations,
                    elapsed=time.perf_counter()-self.started,
                    front=[list(p.objectives) for p in self.points]))
        return ind

class BudgetSolver(IGA_NSGA_VNS):
    def __init__(self, data, config, monitor):
        super().__init__(data, config)
        self.monitor = monitor

    def _evaluate_cached(self, individual, context, dynamic_baseline):
        out = self.monitor.evaluate(individual)
        self.evaluation_count = self.monitor.evaluations
        self.cache_hits = self.monitor.hits
        return out

def instrument_neighbor():
    original = module._propose_neighbor
    def wrapped(current, neighborhood, data, policy, weights, rng):
        candidate = original(current, neighborhood, data, policy, weights, rng)
        candidate._source = neighborhood
        candidate._base = current
        candidate._acceptance = (policy, weights)
        return candidate
    module._propose_neighbor = wrapped
    return original

def run_moead(data, cfg, mon):
    """Discrete MOEA/D: simplex weights, Tchebycheff, 10 neighbors, nr=2.
    OS/MA operators are shared to isolate selection strategy. No ALNS additions.
    """
    rng = random.Random(cfg.seed)
    weights = np.array([(i/7,j/7,(7-i-j)/7) for i in range(8) for j in range(8-i)]
                       + [(1/3,1/3,1/3),(.5,.25,.25),(.25,.5,.25),(.25,.25,.5)])
    size = len(weights)  # 40, same population size as NSGA variants
    cfg.population_size = size
    pop = initialize_population(data,cfg,rng)
    for ind in pop: mon.evaluate(ind)
    scales = np.maximum(1., np.max([p.objectives for p in pop],axis=0))
    ideal = np.min([p.objectives for p in pop],axis=0)
    neighbors = np.argsort(((weights[:,None]-weights[None,:])**2).sum(axis=2),axis=1)[:,:10]
    def score(ind,w):
        return np.max(np.maximum(.0001,w)*np.abs(np.array(ind.objectives)-ideal)/scales)
    while True:
        order = list(range(size)); rng.shuffle(order)
        for i in order:
            pool = list(map(int,neighbors[i])) if rng.random()<.9 else list(range(size))
            a,b = rng.sample(pool,2)
            child,_ = crossover(pop[a],pop[b],data,rng)
            mutate(child,data,.20,.20,rng)
            mon.evaluate(child)
            ideal = np.minimum(ideal,np.array(child.objectives))
            rng.shuffle(pool)
            changes=0
            for j in pool:
                if score(child,weights[j]) <= score(pop[j],weights[j]):
                    pop[j]=child.clone(); changes+=1
                if changes>=2: break

def make_data(instance):
    if instance in ['E0','E1','E2']:
        return make_scenario(load_any_dataset(DATA/'v42_dataset.json'), {'E0':'S0','E1':'S1','E2':'S2'}[instance])
    return load_any_dataset(DATA/'benchmarks'/(instance+'.json'))

def run_one(task):
    instance, method, seed, budget = task
    data = make_data(instance)
    cfg = AlgorithmConfig(seed=seed, verbose=False, generations=10000,
        population_size=40, restart_ratio=0, evaluation_budget=budget,
        initialization_mode='hybrid' if method.startswith('H') else 'random')
    if method in ['R-NSGA','H-NSGA']:
        cfg.vns_probability=0; cfg.elite_local_search_ratio=0
    if method=='H-VNS-R': cfg.restart_ratio=.15
    mon = Monitor(data,budget)
    original = instrument_neighbor()
    started = time.perf_counter()
    solver = None
    try:
        if method=='MOEAD': run_moead(data,cfg,mon)
        else:
            solver = BudgetSolver(data,cfg,mon)
            solver.run()
    except BudgetEnd:
        pass
    finally:
        module._propose_neighbor = original
    elapsed = time.perf_counter()-started
    if mon.evaluations != budget: raise RuntimeError('budget not exhausted')
    issues = [check_schedule(data,p.schedule,p.objectives) for p in mon.points]
    mins = np.array([p.objectives for p in mon.points]).min(axis=0)
    maxs = np.array([p.objectives for p in mon.points]).max(axis=0)
    # Ideal-distance knee selection from same actual Pareto archive; deterministic.
    distances = [np.linalg.norm((np.array(p.objectives)-mins)/np.maximum(1.,maxs-mins)) for p in mon.points]
    representative = mon.points[int(np.argmin(distances))]
    row = dict(instance=instance,method=method,seed=seed,budget=budget,
        evaluations=mon.evaluations,cache_hits=mon.hits,runtime_s=elapsed,
        generations=len(solver.history) if solver else None,
        restarts=solver.restart_count if solver else 0,
        config=asdict(cfg),front=[list(p.objectives) for p in mon.points],
        snapshots=mon.snapshots,local=mon.local,
        front_sources={k:sum(mon.sources.get(p.objectives)==k for p in mon.points) for k in ['genetic/initial','insert','critical_block','machine','swap']},
        audited_schedules=len(issues),invalid_schedules=sum(bool(p) for p in issues),
        representative=dict(objectives=list(representative.objectives),
            order_sequence=representative.order_sequence, machine_assignment=representative.machine_assignment,
            entries=[asdict(e) for e in representative.schedule.entries.values()]))
    return row

def convert_benchmarks(source=None):
    # The distributed JSON inputs are already frozen. Never clear their
    # provenance manifest merely because the original author's E: drive is absent.
    target=DATA/'benchmarks'
    if source is None:
        required=[target/f'Mk{i:02d}.json' for i in range(1,11)]
        if all(path.is_file() for path in required):
            return
        raise FileNotFoundError('Packaged Mk01-Mk10 JSON inputs are missing; supply --benchmark-source with the original .fjs directory')
    source=Path(source)
    if not source.is_dir() or not list(source.glob('Mk*.fjs')):
        raise FileNotFoundError(f'No Mk*.fjs inputs found in {source}')
    target.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for src in sorted(source.glob('Mk*.fjs')):
        tokens=src.read_text().split(); n,m=int(tokens[0]),int(tokens[1])
        # Third header field is average flexibility; rest are one line per job.
        lines=src.read_text().splitlines()[1:]
        jobs=[]; operations=[]
        for i,line in enumerate([l for l in lines if l.strip()]):
            nums=list(map(int,line.split())); count=nums[0]; pos=1
            job=f'J{i+1:02d}'
            jobs.append(dict(job_id=job,family=job,release_time=0,due_date=10000))
            for j in range(count):
                nc=nums[pos]; pos+=1
                for _ in range(nc):
                    k,p=nums[pos:pos+2]; pos+=2
                    operations.append(dict(job_id=job,op_index=j+1,operation_id=str(j+1),
                        machine_id=f'M{k:02d}',processing_time=p,setup_time=0,
                        running_energy=p*(.85+.03*k),auxiliary_energy=0))
            if pos!=len(nums): raise ValueError('parser mismatch '+src.name)
        if len(jobs)!=n: raise ValueError('job count mismatch')
        payload=dict(jobs=jobs,operations=operations,
            machines=[dict(machine_id=f'M{k:02d}') for k in range(1,m+1)],
            metadata=dict(source=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
            time_unit='benchmark unit',energy_extension='dimensionless e=p*(0.85+0.03*k)'))
        (target/(src.stem+'.json')).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        manifest.append(dict(instance=src.stem,jobs=n,machines=m,operations=sum(1 for o in {(r['job_id'],r['op_index']) for r in operations}),arcs=len(operations),**payload['metadata']))
    (DATA/'benchmark_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

def task_list(group):
    if group=='main': return [('E0',m,s,BUDGET) for s in range(410001,410031) for m in VARIANTS]
    if group=='benchmark': return [(f'Mk{i:02d}',m,s,BUDGET) for i in range(1,11) for s in range(420001,420011) for m in ['R-NSGA','H-VNS','MOEAD']]
    if group=='stress': return [(i,m,s,BUDGET) for i in ['E1','E2'] for s in range(430001,430013) for m in ['H-NSGA','H-VNS']]
    if group=='pilot': return [('E0',m,400001,4000) for m in VARIANTS]
    raise ValueError(group)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--group',default='pilot')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--benchmark-source',type=Path,default=None)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True); convert_benchmarks(args.benchmark_source)
    tasks=task_list(args.group)
    done=[]
    checkpoint=OUT/(args.group+'_runs.json')
    if checkpoint.exists(): done=json.loads(checkpoint.read_text(encoding='utf-8'))
    keys={(r['instance'],r['method'],r['seed'],r['budget']) for r in done}
    tasks=[t for t in tasks if t not in keys]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_one,t):t for t in tasks}
        for fut in as_completed(futures):
            row=fut.result(); done.append(row)
            checkpoint.write_text(json.dumps(done,ensure_ascii=False),encoding='utf-8')
            print(f'{args.group} {len(done)} {row["instance"]} {row["method"]} seed={row["seed"]} decode={row["evaluations"]} sec={row["runtime_s"]:.2f} front={len(row["front"])} invalid={row["invalid_schedules"]}',flush=True)

if __name__=='__main__': main()
