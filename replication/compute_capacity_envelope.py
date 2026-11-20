"""Input-only LP diagnostics added on 2026-10-29; does not alter frozen HV bounds."""
from pathlib import Path
import sys, json, csv, hashlib, platform
ROOT=Path(__file__).resolve().parents[2]
LOCAL=ROOT/'04_修订与核查'/'工具'/'python_deps'
if LOCAL.exists(): sys.path.insert(0,str(LOCAL))
import numpy as np
import scipy
from scipy.optimize import linprog
from fjsp.dataset_io import load_any_dataset

HERE=Path(__file__).resolve().parent
OUT=HERE/'results_20261029_capacity'

def solve(data, cap=None, allow_infeasible=False):
    ops=list(data.operations.values())
    arcs=[(i,o) for i,s in enumerate(ops) for o in s.candidates]
    machines=list(data.machine_ids)
    mi={m:i for i,m in enumerate(machines)}
    n=len(arcs)
    eq=np.zeros((len(ops),n+1)); ub=np.zeros((len(machines)+(cap is not None),n+1))
    energy=np.zeros(n+1)
    for k,(i,o) in enumerate(arcs):
        eq[i,k]=1; ub[mi[o.machine_id],k]=o.processing_time; energy[k]=o.total_energy
    ub[:len(machines),-1]=-1
    b=np.zeros(ub.shape[0])
    if cap is not None: ub[-1]=energy; b[-1]=cap
    c=np.zeros(n+1);c[-1]=1
    res=linprog(c,A_ub=ub,b_ub=b,A_eq=eq,b_eq=np.ones(len(ops)),bounds=[(0,1)]*n+[(0,None)],method='highs')
    if not res.success:
        if allow_infeasible and res.status==2:return dict(energy_cap=cap,status=2,message=res.message)
        raise RuntimeError(res.message)
    eqerr=float(np.max(np.abs(eq@res.x-1)))
    violation=float(max(0,np.max(ub@res.x-b),-np.min(res.x),np.max(res.x[:-1]-1)))
    dual=float(b@res.ineqlin.marginals+np.ones(len(ops))@res.eqlin.marginals+np.ones(n)@res.upper.marginals[:-1])
    gap=abs(float(res.fun)-dual)
    assert eqerr<1e-7 and violation<1e-7 and gap<1e-6,(eqerr,violation,gap)
    return dict(energy_cap=cap,lambda_star=float(res.fun),fractional_energy=float(energy@res.x),
                primal_eq_residual=eqerr,primal_max_violation=violation,dual_objective=dual,duality_gap=gap,
                assignment=[dict(operation=ops[i].operation_id,machine=o.machine_id,y=float(res.x[k])) for k,(i,o) in enumerate(arcs) if res.x[k]>1e-10])

def main():
    OUT.mkdir(exist_ok=True)
    records=[]; sources={}
    for label in ['E0']+[f'Mk{i:02d}' for i in range(1,11)]:
        p=HERE/'data'/('v42_dataset.json' if label=='E0' else f'benchmarks/{label}.json')
        data=load_any_dataset(p)
        mins={v:min(o.processing_time for o in s.candidates) for v,s in data.operations.items()}
        total=sum(mins.values());q=len({o.machine_id for s in data.operations.values() for o in s.candidates})
        chain=max(j.release_time+sum(mins[o.operation_id] for o in j.operations) for j in data.jobs.values())
        e_min=sum(min(o.total_energy for o in s.candidates) for s in data.operations.values())
        row=dict(instance=label,defined_machines=len(data.machine_ids),eligible_machines=q,operations=len(mins),
                 arcs=sum(len(s.candidates) for s in data.operations.values()),energy_min=e_min,
                 average_defined=total/len(data.machine_ids),average_eligible=total/q,chain_lower=chain,
                 unconditioned=solve(data),minimum_energy=solve(data,e_min))
        if label=='E0':
            forced={};ties=[]
            for v,s in data.operations.items():
                mine=min(o.total_energy for o in s.candidates)
                opts=[o for o in s.candidates if abs(o.total_energy-mine)<1e-9]
                if len(opts)!=1: ties.append(v)
                else:
                    o=opts[0];forced[o.machine_id]=forced.get(o.machine_id,0.0)+o.processing_time
            row['minimum_energy_certificate']=dict(tied_operations=ties,forced_processing_by_machine=forced,
                max_forced_processing=max(forced.values()),bottleneck=max(forced,key=forced.get))
            assert not ties and abs(max(forced.values())-1304)<1e-7
            caps=[e_min,e_min+5,e_min+10,e_min+20,762.59,800,863.07,1015.82]
            row['envelope']=[solve(data,v) for v in caps]
            vals=[r['lambda_star'] for r in row['envelope']]
            assert all(b<=a+1e-7 for a,b in zip(vals,vals[1:]))
            # Jensen check at three budget pairs, independently solved at their midpoints.
            row['convexity_checks']=[]
            for a,b in [(caps[0],caps[2]),(caps[2],caps[4]),(caps[4],caps[6])]:
                fa=solve(data,a)['lambda_star'];fb=solve(data,b)['lambda_star'];fm=solve(data,(a+b)/2)['lambda_star']
                assert fm<=(fa+fb)/2+1e-7
                row['convexity_checks'].append(dict(a=a,b=b,midpoint_value=fm,endpoint_mean=(fa+fb)/2))
            row['below_minimum_energy_check']=solve(data,e_min-1e-3,allow_infeasible=True)
            assert row['below_minimum_energy_check']['status']==2
        records.append(row);sources[str(p.relative_to(HERE))]=hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(date='2026-10-29',purpose='post hoc structural diagnostics, no change to confirmation protocol',
                python=sys.version,scipy=scipy.__version__,numpy=np.__version__,platform=platform.platform(),
                sources=sources,instances=records)
    (OUT/'capacity-envelope.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    with (OUT/'bounds-summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=['instance','defined_machines','eligible_machines','average_defined','average_eligible','chain_lower','energy_min','lambda_unconstrained','lambda_minimum_energy']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in records:w.writerow({**{k:r[k] for k in keys if k in r},'lambda_unconstrained':r['unconditioned']['lambda_star'],'lambda_minimum_energy':r['minimum_energy']['lambda_star']})
    e0=records[0]
    print(json.dumps({k:v for k,v in e0.items() if k not in ['envelope','unconditioned','minimum_energy']},ensure_ascii=False,indent=2))
    print(json.dumps({'unconstrained':e0['unconditioned']['lambda_star'],'minimum_energy':e0['minimum_energy']['lambda_star'],
                      'envelope':[{k:r[k] for k in ['energy_cap','lambda_star','duality_gap']} for r in e0['envelope']]},indent=2))

if __name__=='__main__':main()
