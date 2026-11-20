"""Paired physical constraint factorial and independent intensity diagnostics."""
from pathlib import Path
import json
import pandas as pd
from analyze_budget_study import paired,holm
OUT=Path(__file__).resolve().parent/'results_v2'
def main():
    df=pd.read_csv(OUT/'metrics.csv'); effects=[]
    part=df[(df.method=='H-MVNS')&df.instance.isin(['CF0','E1','CFM','E2'])]
    assert len(part)==48
    for col in ['cmax_min','tec_min','lmax_min']:
        p=part.pivot(index='seed',columns='instance',values=col)
        assert p.shape==(12,4) and not p.isna().any().any()
        for title,d in [('setup only',p.E1-p.CF0),('maintenance only',p.CFM-p.CF0),
                        ('joint',p.E2-p.CF0),('interaction',p.E2-p.E1-p.CFM+p.CF0)]:
            row=paired(d,title);row['objective']=col;effects.append(row)
    holm(effects);pd.DataFrame(effects).to_csv(OUT/'constraint_effects.csv',index=False,encoding='utf-8-sig')
    part.groupby('instance')[['cmax_min','tec_min','lmax_min']].agg(['mean','std']).to_csv(OUT/'constraint_summary.csv',encoding='utf-8-sig')
    raw=json.loads((OUT/'local_budget_runs.json').read_text(encoding='utf8'))
    rows=[]
    for r in raw:
        fresh=sum(v['fresh'] for v in r['local'].values())
        rows.append(dict(seed=r['seed'],method=r['method'],trials=r['config']['local_search_trials'],
                         local_fresh=fresh,local_fraction=fresh/r['evaluations']))
    tuning=df[df.instance=='E0T'].merge(pd.DataFrame(rows),on=['method','seed'],validate='one_to_one')
    assert len(tuning)==24
    tuning.to_csv(OUT/'intensity_metrics.csv',index=False,encoding='utf-8-sig')
    tuning.groupby('trials')[['hv','igdplus','generations','local_fraction','cmax_min','lmax_min']].agg(['mean','std']).to_csv(OUT/'intensity_summary.csv',encoding='utf-8-sig')
    # Count repeated proposals and summarize within parents before aggregating.
    step=pd.read_csv(OUT/'single_step.csv')
    print('Constraint effects\n',pd.DataFrame(effects).to_string(index=False))
    print('Intensity\n',tuning.groupby('trials')[['hv','igdplus','generations','local_fraction']].mean().to_string())
if __name__=='__main__':main()
