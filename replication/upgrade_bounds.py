"""Input-only analytic bounds, fixed before prospective optimizer results."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from run_budget_study import make_data

def analytic_bounds(data):
    minimum = {v:min(o.processing_time for o in s.candidates) for v,s in data.operations.items()}
    total_min = sum(minimum.values())
    m = len(data.machine_ids)
    c_lower = max(total_min/m, max(j.release_time+sum(minimum[o.operation_id] for o in j.operations)
                                  for j in data.jobs.values()))
    longest = max(0, max((end for spec in data.machines.values() for _,end in spec.maintenance_windows),default=0))
    occupation_upper = sum(max(o.processing_time+o.setup_time for o in s.candidates)
                           for s in data.operations.values())
    lo = np.array([c_lower,sum(min(o.total_energy for o in s.candidates) for s in data.operations.values()),
                   max(total_min/m,max(minimum.values()))])
    hi = np.array([max(j.release_time for j in data.jobs.values())+longest+occupation_upper,
                   sum(max(o.total_energy for o in s.candidates) for s in data.operations.values()),occupation_upper])
    assert np.all(np.isfinite(lo)) and np.all(hi>=lo)
    reference = hi+np.maximum(1,.1*(hi-lo))
    return dict(lower=lo.tolist(),upper=hi.tolist(),reference=reference.tolist(),
                eligible_machines=len({o.machine_id for s in data.operations.values() for o in s.candidates}),
                defined_machines=m)

def main():
    here = Path(__file__).parent
    instances=['E0']+[f'Mk{i:02d}' for i in range(1,11)]
    rows={i:analytic_bounds(make_data(i)) for i in instances}
    sources={str(p.relative_to(here)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [here/'data/v42_dataset.json']+sorted((here/'data/benchmarks').glob('Mk*.json'))}
    out=here/'results_v3';out.mkdir(exist_ok=True)
    (out/'analytic-bounds.json').write_text(json.dumps(dict(bounds=rows,data_sha256=sources),indent=2),encoding='utf8')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
