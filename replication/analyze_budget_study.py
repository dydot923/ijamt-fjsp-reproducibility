"""Recompute all metrics with fixed instance-wise scales and paired statistics."""
from pathlib import Path
import json,csv,itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,friedmanchisquare,rankdata
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'replication'/'results_v2'

def nondominated(points):
    a=np.unique(np.round(np.asarray(points,dtype=float),9),axis=0)
    a=a[np.argsort(a[:,0],kind='stable')]
    keep=[]
    for p in a:
        if keep and np.any(np.all(np.asarray(keep)<=p+1e-9,axis=1)): continue
        keep=[q for q in keep if not (np.all(p<=q+1e-9) and np.any(p<q-1e-9))]
        keep.append(p)
    return np.asarray(keep)

def hv3(points):
    a=np.unique(np.round(np.asarray(points),12),axis=0)
    a=a[np.all(a<1,axis=1)]; a=a[np.argsort(a[:,0])]
    total=0.
    for i,p in enumerate(a):
        right=a[i+1,0] if i+1<len(a) else 1.
        yz=a[:i+1,1:]; yz=yz[np.argsort(yz[:,0])]
        z=np.minimum.accumulate(yz[:,1]); y=np.r_[yz[:,0],1.]
        area=np.sum(np.diff(y)*(1-z))
        total+=(right-p[0])*area
    return float(total)

def igdplus(front, ref):
    a=np.asarray(front); r=np.asarray(ref)
    delta=np.maximum(a[None,:,:]-r[:,None,:],0)
    return float(np.sqrt((delta**2).sum(axis=2)).min(axis=1).mean())

def interval(d):
    d=np.asarray(d,float); rng=np.random.default_rng(20261007)
    v=d[rng.integers(0,len(d),size=(20000,len(d)))].mean(axis=1)
    return list(map(float,np.quantile(v,[.025,.975])))

def paired(d,label):
    d=np.asarray(d,float)
    return dict(contrast=label,n=len(d),delta=float(d.mean()),lo=interval(d)[0],hi=interval(d)[1],
        wins=int((d>1e-9).sum()),ties=int((np.abs(d)<=1e-9).sum()),
        p=float(wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto').pvalue) if np.any(np.abs(d)>1e-9) else 1.)

def holm(rows):
    order=sorted(range(len(rows)),key=lambda i:rows[i]['p']); last=0.
    for j,i in enumerate(order):
        last=max(last,min(1.,rows[i]['p']*(len(rows)-j)));rows[i]['p_holm']=last

def main():
    rows=[]
    for name in ['main','benchmark','benchmark_ablation','matched','stress','constraint_factorial','local_budget']:
        p=OUT/(name+'_runs.json')
        if p.exists(): rows.extend(json.loads(p.read_text(encoding='utf-8')))
    summaries=[]; records=[]; curves=[]; bounds={}
    for instance in sorted({r['instance'] for r in rows}):
        subset=[r for r in rows if r['instance']==instance]
        raw=[p for r in subset for s in r['snapshots'] for p in s['front']]+[p for r in subset for p in r['front']]
        a=np.asarray(raw); low=a.min(axis=0); high=a.max(axis=0)
        ref=high+np.maximum(1.,.1*(high-low))
        scale=ref-low
        empirical=nondominated([p for r in subset for p in r['front']]); refnorm=(empirical-low)/scale
        bounds[instance]=dict(ideal=low.tolist(),reference=ref.tolist(),reference_front=empirical.tolist())
        for r in subset:
            assert r['evaluations']==r['budget']==4000
            assert r['config'].get('evaluation_budget')==4000
            assert r['invalid_schedules']==0
            front=np.unique(np.round(np.asarray(r['front']),9),axis=0)
            normalized=(front-low)/scale
            result={k:r[k] for k in ['instance','method','seed','evaluations','cache_hits','runtime_s','generations','restarts','audited_schedules','invalid_schedules']}
            result.update(hv=hv3(normalized),igdplus=igdplus(normalized,refnorm),front_size=len(front),
                cmax_min=front[:,0].min(),tec_min=front[:,1].min(),lmax_min=front[:,2].min())
            records.append(result)
            for s in r['snapshots']:
                norms=(np.asarray(s['front'])-low)/scale
                curves.append(dict(instance=instance,method=r['method'],seed=r['seed'],evaluations=s['evaluations'],
                    hv=hv3(norms),igdplus=igdplus(norms,refnorm)))
    df=pd.DataFrame(records); df.to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(curves).to_csv(OUT/'curves.csv',index=False,encoding='utf-8-sig')
    (OUT/'bounds.json').write_text(json.dumps(bounds,indent=2),encoding='utf-8')
    agg=df.groupby(['instance','method']).agg(n=('hv','count'),hv_mean=('hv','mean'),hv_sd=('hv','std'),
        igd_mean=('igdplus','mean'),igd_sd=('igdplus','std'),cmax_mean=('cmax_min','mean'),tec_mean=('tec_min','mean'),lmax_mean=('lmax_min','mean'),
        decode_mean=('evaluations','mean'),generations_mean=('generations','mean'),restart_mean=('restarts','mean'))
    agg.to_csv(OUT/'summary.csv',encoding='utf-8-sig')
    base=df[df.instance=='E0'].pivot(index='seed',columns='method',values='hv')
    effects=[]
    for label,a,b in [('H effect, no VNS','H-NSGA','R-NSGA'),('H effect, with VNS','H-VNS','R-VNS'),
                       ('VNS effect, random','R-VNS','R-NSGA'),('VNS effect, hybrid','H-VNS','H-NSGA'),('Restart','H-VNS-R','H-VNS')]:
        if a in base and b in base: effects.append(paired((base[a]-base[b]).dropna(),label))
    if all(c in base for c in ['H-VNS','H-NSGA','R-VNS','R-NSGA']):
        effects.append(paired(base['H-VNS']-base['H-NSGA']-base['R-VNS']+base['R-NSGA'],'H x VNS interaction'))
    if 'H-MVNS' in base:
        for b in ['H-VNS','H-NSGA','R-NSGA','MOEAD']:
            effects.append(paired(base['H-MVNS']-base[b],'H-MVNS minus '+b))
    holm(effects);pd.DataFrame(effects).to_csv(OUT/'effects.csv',index=False,encoding='utf-8-sig')
    bm=df[df.instance.str.startswith('Mk')]
    contrasts=[]; cross={}
    if len(bm):
        for ins,g in bm.groupby('instance'):
            pp=g.pivot(index='seed',columns='method',values='hv')
            for b in ['R-NSGA','H-NSGA','MOEAD','H-VNS']:
                for a in ['H-VNS','H-MVNS']:
                    if a!=b and a in pp and b in pp:
                        d=(pp[a]-pp[b]).dropna(); row=paired(d,a+' minus '+b);row['instance']=ins;contrasts.append(row)
        holm(contrasts);pd.DataFrame(contrasts).to_csv(OUT/'benchmark_effects.csv',index=False,encoding='utf-8-sig')
        pp=bm.groupby(['instance','method']).hv.mean().unstack().dropna()
        if len(pp)>1 and pp.shape[1]>=3:
            stat,p=friedmanchisquare(*[pp[c].values for c in pp]);cross['friedman']={'stat':float(stat),'p':float(p),'n_instances':len(pp)}
            ranks=pp.apply(lambda v:rankdata(-v),axis=1,result_type='expand'); ranks.columns=pp.columns
            cross['mean_ranks']=ranks.mean().to_dict()
            comp=[]
            main='H-MVNS' if 'H-MVNS' in pp else 'H-VNS'
            for b in pp:
                if b!=main: comp.append(paired(pp[main]-pp[b],main+' minus '+b))
            holm(comp);cross['instance_level_comparisons']=comp
    stress=[]
    for ins in ['E1','E2']:
        pp=df[df.instance==ins].pivot(index='seed',columns='method',values='hv')
        for a in ['H-VNS','H-MVNS']:
            if a in pp and 'H-NSGA' in pp:
                row=paired((pp[a]-pp['H-NSGA']).dropna(),a+' minus H-NSGA');row['instance']=ins;stress.append(row)
    holm(stress);pd.DataFrame(stress).to_csv(OUT/'stress_effects.csv',index=False,encoding='utf-8-sig')
    audit=dict(runs=len(rows),decode_total=sum(r['evaluations'] for r in rows),audited_schedules=sum(r['audited_schedules'] for r in rows),
        invalid_schedules=sum(r['invalid_schedules'] for r in rows),cross_instance=cross,
        representative_method='H-MVNS',
        representative_seed=int(df[(df.instance=='E0')&(df.method=='H-MVNS')].iloc[(df[(df.instance=='E0')&(df.method=='H-MVNS')].hv-df[(df.instance=='E0')&(df.method=='H-MVNS')].hv.median()).abs().argmin()].seed))
    (OUT/'analysis.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(agg.round(5).to_string());print(pd.DataFrame(effects).round(6).to_string(index=False));print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__': main()
