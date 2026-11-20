import json
from pathlib import Path
import numpy as np,pandas as pd
from analyze_budget_study import OUT,hv3,nondominated,igdplus,paired,holm
rows=json.loads((OUT/'main_runs.json').read_text(encoding='utf8'))+json.loads((OUT/'matched_runs.json').read_text(encoding='utf8'))
rows=[r for r in rows if r['instance']=='E0'];assert len(rows)==210
a=np.array([p for r in rows for s in r['snapshots'] for p in s['front']]);lo=a.min(axis=0);hi=a.max(axis=0);scale=hi+np.maximum(1.,.1*(hi-lo))-lo
reference=(nondominated([p for r in rows for p in r['front']])-lo)/scale
records=[]
for r in rows:
    p=np.array(r['front']);records.append(dict(method=r['method'],seed=r['seed'],hv=hv3((p-lo)/scale),igdplus=igdplus((p-lo)/scale,reference),
        cmax=p[:,0].min(),tec=p[:,1].min(),lmax=p[:,2].min(),generations=r['generations'],restarts=r['restarts']))
df=pd.DataFrame(records);df.to_csv(OUT/'engineering_metrics.csv',index=False,encoding='utf8')
df.groupby('method').agg(['mean','std']).to_csv(OUT/'engineering_summary.csv',encoding='utf8')
pp=df.pivot(index='seed',columns='method',values='hv');effects=[]
for label,a,b in [('H effect, no VNS','H-NSGA','R-NSGA'),('H effect, with VNS','H-VNS','R-VNS'),('VNS effect, random','R-VNS','R-NSGA'),('VNS effect, hybrid','H-VNS','H-NSGA'),('Restart','H-VNS-R','H-VNS'),('H-MVNS minus H-VNS','H-MVNS','H-VNS'),('H-MVNS minus H-NSGA','H-MVNS','H-NSGA'),('H-MVNS minus R-NSGA','H-MVNS','R-NSGA'),('H-MVNS minus MOEAD','H-MVNS','MOEAD')]:effects.append(paired(pp[a]-pp[b],label))
effects.append(paired(pp['H-VNS']-pp['H-NSGA']-pp['R-VNS']+pp['R-NSGA'],'H x VNS interaction'));holm(effects)
pd.DataFrame(effects).to_csv(OUT/'engineering_effects.csv',index=False,encoding='utf8')
print(df.groupby('method').mean().round(6).to_string());print(pd.DataFrame(effects).round(6).to_string(index=False))
