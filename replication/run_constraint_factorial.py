"""Complete same-seed physical constraint factorial for the matched method."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import run_budget_study as study
import run_matched_study as matched
from fjsp.dataset_io import load_any_dataset
def data(instance):
    d=load_any_dataset(study.DATA/'v42_dataset.json')
    if instance=='CFM':
        for k in ['M01','M04','M08']:d.machines[k].maintenance_windows=[(100.,135.),(210.,245.)]
    return d
def run(task):
    study.make_data=data
    return matched.run_one(task)
def main():
    tasks=[(ins,'H-MVNS',seed,4000) for ins in ['CF0','CFM'] for seed in range(430001,430013)]
    path=study.OUT/'constraint_factorial_runs.json';rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(run,t) for t in tasks]):
            r=f.result();rows.append(r);path.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf8')
            print('constraint factorial',len(rows),r['instance'],r['seed'],r['invalid_schedules'],flush=True)
if __name__=='__main__':main()
