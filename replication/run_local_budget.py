"""Exploratory local-search intensity, separate seeds from main comparison."""
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import run_budget_study as study
from run_matched_study import instrument_matched
from fjsp.config import AlgorithmConfig as BaseConfig
def run_alias(task):
    trials=int(task[1].split('-')[-1][1:])
    def cfg(**kwargs):return BaseConfig(local_search_trials=trials,**kwargs)
    original=study.make_data;study.make_data=lambda ins:original('E0')
    study.AlgorithmConfig=cfg;study.instrument_neighbor=instrument_matched
    return study.run_one(task)
def main():
    tasks=[('E0T',f'H-MVNS-L{t}',s,4000) for t in [1,2,4,6] for s in range(440001,440007)]
    path=study.OUT/'local_budget_runs.json';rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(run_alias,t) for t in tasks]):
            r=f.result();rows.append(r);path.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf8');print('local budget',len(rows),r['method'],r['seed'],flush=True)
if __name__=='__main__':main()
