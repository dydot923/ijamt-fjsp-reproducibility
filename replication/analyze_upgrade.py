"""Locked analytic normalization, balanced reference set and paired families."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
from analyze_budget_study import nondominated,hv3,igdplus
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'replication/results_v3'

def fps(points,limit):
    a=np.unique(np.round(np.asarray(points,float),9),axis=0)
    if len(a)<=limit:return a
    chosen=list(dict.fromkeys(int(np.argmin(a[:,i])) for i in range(3)))
    nearest=np.min(np.sum((a[:,None,:]-a[chosen][None,:,:])**2,axis=2),axis=1)
    nearest[chosen]=-1
    while len(chosen)<limit:
        k=int(np.argmax(nearest));chosen.append(k)
        nearest=np.minimum(nearest,((a-a[k])**2).sum(axis=1));nearest[chosen]=-1
    return a[chosen]

def effect(d,family,metric,control,budget,unit):
    d=np.array(d,dtype=float,copy=True);d[np.abs(d)<1e-12]=0
    raw_delta=float(d.mean())
    if metric!='hv':d=-d
    rng=np.random.default_rng(20261008)
    means=d[rng.integers(0,len(d),size=(20000,len(d)))].mean(axis=1)
    lo,hi=np.quantile(means,[.025,.975])
    nonzero=d[d!=0]
    if len(nonzero):
        ranks=rankdata(np.abs(nonzero),method='average')
        observed=abs(np.dot(np.sign(nonzero),ranks))
        signs=2*((np.arange(2**len(nonzero))[:,None]>>np.arange(len(nonzero)))&1)-1
        p=float(np.mean(np.abs(signs@ranks)>=observed-1e-10))
    else:p=1.
    return dict(family=family,metric=metric,control=control,candidate='H-MVNS-2',budget=budget,
                unit=unit,n=len(d),delta=float(d.mean()),median_delta=float(np.median(d)),
                lo=float(lo),hi=float(hi),wins=int((d>0).sum()),ties=int((d==0).sum()),raw_delta=raw_delta,
                p=p)

def holm(rows):
    floor=0
    for rank,i in enumerate(sorted(range(len(rows)),key=lambda i:rows[i]['p'])):
        floor=max(floor,min(1,rows[i]['p']*(len(rows)-rank)));rows[i]['p_holm']=floor

def theory_summary():
    path=ROOT/'qa/theory-upgrade-evidence/single-move-raw.jsonl'
    rows=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
    summary=[]
    for scenario in ['E0','E1','E2']:
        group=[r for r in rows if r['scenario']==scenario]
        err=np.array([r['proxy_error'] for r in group]);width=np.array([max(r['upper'].values())-max(r['lower'].values()) for r in group])
        correlations=[]
        for seed in sorted({r['seed'] for r in group}):
            father=[r for r in group if r['seed']==seed]
            # Identical mathematical loads may differ at1e-13 due to sum order.
            # Round only this descriptive rank diagnostic, never objective/HV data.
            x=[round(max(r['predicted_loads'].values()),9) for r in father];y=[round(r['new_objectives'][2],9) for r in father]
            correlations.append(float(spearmanr(x,y).statistic))
        summary.append(dict(scenario=scenario,moves=len(group),parents=8,
            over=int((err>1e-8).sum()),equal=int((np.abs(err)<=1e-8).sum()),under=int((err<-1e-8).sum()),
            max_abs_error=float(np.abs(err).max()),mean_abs_error=float(np.abs(err).mean()),
            interval_width_mean=float(width.mean()),interval_width_max=float(width.max()),
            guaranteed_improve=sum(max(r['upper'].values())<r['old_objectives'][2]-1e-8 for r in group),
            guaranteed_no_strict_improve=sum(max(r['lower'].values())>=r['old_objectives'][2]-1e-8 for r in group),
            false_predicted_improve=sum(max(r['predicted_loads'].values())<r['old_objectives'][2]-1e-8 and r['new_objectives'][2]>=r['old_objectives'][2]-1e-8 for r in group),
            parent_spearman_mean=float(np.mean(correlations))))
    pd.DataFrame(summary).to_csv(OUT/'theory-diagnostics.csv',index=False,encoding='utf-8-sig')
    return summary

def main():
    rows=[json.loads(p.read_text(encoding='utf8')) for p in sorted((OUT/'completed').glob('*.json'))]
    design=json.loads((OUT/'run-design.json').read_text(encoding='utf8'))
    expected={tuple(t) for t in design['tasks']}
    actual={(r['instance'],r['method'],r['seed'],r['budget']) for r in rows}
    assert len(rows)==678 and actual==expected
    bounds=json.loads((OUT/'analytic-bounds.json').read_text(encoding='utf8'))['bounds']
    records=[];curves=[];fronts={}
    for instance in bounds:
        subset=[r for r in rows if r['instance']==instance]
        b=bounds[instance];lo=np.array(b['lower']);hi=np.array(b['upper']);scale=np.array(b['reference'])-lo
        normalize=lambda a:(np.asarray(a)-lo)/scale
        balanced=[r for r in subset if r['budget']==4000 and 450001<=r['seed']<=450006]
        assert len(balanced)==24
        samples=[]
        for r in balanced:
            a=normalize(r['front'])
            endpoints=a[[int(np.argmin(a[:,i])) for i in range(3)]]
            samples.append(np.unique(np.concatenate([fps(a,50),endpoints]),axis=0))
        primary= fps(nondominated(np.concatenate(samples)),500)
        allfront=nondominated(np.concatenate([normalize(r['front']) for r in subset]))
        fronts[instance]=dict(primary_normalized=primary.tolist(),all_normalized=allfront.tolist(),
                             primary_raw=(primary*scale+lo).tolist(),all_raw=(allfront*scale+lo).tolist(),
                             balanced_runs=24,primary_points=len(primary),all_points=len(allfront))
        for r in subset:
            a=np.asarray(r['front']);assert np.all(a>=lo-1e-6) and np.all(a<=hi+1e-6),(instance,a.min(0),a.max(0),b)
            assert r['evaluations']==r['budget'] and r['invalid_schedules']==0
            c=r['decoder_counts'];assert c['full_calls']+c['prefix_calls']==r['evaluations']
            assert c['reinserted_operations']+c['reused_operations']==r['evaluations']*len(__import__('run_budget_study').make_data(instance).operations)
            norm=normalize(a)
            record={k:r[k] for k in ['instance','method','seed','budget','evaluations','cache_hits','runtime_s','generations','audited_schedules','invalid_schedules']}
            record.update(hv=hv3(norm),igdplus=igdplus(norm,primary),igdplus_all=igdplus(norm,allfront),
                front_size=len(a),cmax_min=a[:,0].min(),tec_min=a[:,1].min(),lmax_min=a[:,2].min(),
                local_fresh=sum(v['fresh'] for v in r['local'].values()),**c)
            records.append(record)
            for s in r['snapshots']:
                na=normalize(s['front'])
                curves.append(dict(instance=instance,method=r['method'],seed=r['seed'],budget=r['budget'],
                                  evaluations=s['evaluations'],hv=hv3(na),igdplus=igdplus(na,primary)))
    df=pd.DataFrame(records);pd.DataFrame(curves).to_csv(OUT/'curves.csv',index=False,encoding='utf-8-sig')
    df.to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
    (OUT/'reference-fronts.json').write_text(json.dumps(fronts,indent=2),encoding='utf8')
    means=df.groupby(['instance','method','budget']).mean(numeric_only=True).reset_index()
    means.to_csv(OUT/'instance-means.csv',index=False,encoding='utf-8-sig')
    effects=[]
    controls=['H-NSGA','H-MVNS-6']
    for population,unit,n in [('Mk','10 instance means',10),('E0','12 paired seeds',12)]:
        data=means[(means.instance!='E0')&(means.budget==4000)] if population=='Mk' else df[(df.instance=='E0')&(df.budget==4000)]
        family=[]
        for metric in ['hv','igdplus']:
            p=data.pivot(index='instance' if population=='Mk' else 'seed',columns='method',values=metric)
            for ctrl in controls:
                d=(p['H-MVNS-2']-p[ctrl]).dropna();assert len(d)==n
                family.append(effect(d,f'F1-{population}',metric,ctrl,4000,unit))
        holm(family);effects+=family
    family=[]
    mk=means[means.instance!='E0']
    for budget in [2000,8000,'trapezoidal mean']:
        for metric in ['hv','igdplus']:
            data=mk[mk.budget==budget] if isinstance(budget,int) else mk[(mk.method!='RP-NSGA')].copy()
            if not isinstance(budget,int):
                data['weighted']=data[metric]*data.budget.map({2000:1/6,4000:1/2,8000:1/3})
                data=data.groupby(['instance','method'])['weighted'].sum().reset_index().rename(columns={'weighted':metric})
            p=data.pivot(index='instance',columns='method',values=metric)
            for ctrl in controls:family.append(effect(p['H-MVNS-2']-p[ctrl],'F2',metric,ctrl,budget,'10 instance means'))
    holm(family);effects+=family
    family=[]
    for metric in ['hv','igdplus']:
        p=mk[mk.budget==4000].pivot(index='instance',columns='method',values=metric)
        family.append(effect(p['H-MVNS-2']-p['RP-NSGA'],'F3',metric,'RP-NSGA',4000,'10 instance means'))
    holm(family);effects+=family
    pd.DataFrame(effects).to_csv(OUT/'effects.csv',index=False,encoding='utf-8-sig')
    oracle=pd.DataFrame(json.loads((OUT/'oracle/runs.json').read_text(encoding='utf8')))
    oracle.groupby(['instance','method'])[['hv_ratio','igdplus','exact_coverage']].agg(['mean','std']).to_csv(OUT/'oracle-metrics.csv',encoding='utf-8-sig')
    theory=theory_summary()
    summary=dict(runs=len(rows),evaluations=sum(r['evaluations'] for r in rows),audited_schedules=int(df.audited_schedules.sum()),
                 invalid_schedules=int(df.invalid_schedules.sum()),families=effects,theory=theory,
                 decoder_totals={k:int(df[k].sum()) for k in ['full_calls','prefix_calls','reused_operations','reinserted_operations']},
                 oracle_runs=len(oracle),oracle_evaluations=int(oracle.evaluations.sum()),oracle_audited=int(oracle.audited_schedules.sum()),
                 file_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'run-design.json',OUT/'analytic-bounds.json',Path(__file__)]})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
