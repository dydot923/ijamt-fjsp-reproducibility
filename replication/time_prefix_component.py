"""Serial paired component timing, run after parallel confirmation finishes."""
import json,random,time,platform,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
import run_budget_study as study
from fjsp.nsga_vns import _build_order_sequence,_build_machine_assignment
from fjsp.decoder import decode,_sequence_to_operation_ids
from prefix_evaluator import make_evaluator

OUT=Path(__file__).parent/'results_v3'

def main():
    assert len(list((OUT/'completed').glob('*.json')))==678,'Wait for parallel experiment to finish'
    results=[]
    counters=dict(full_calls=0,prefix_calls=0,reused_operations=0,reinserted_operations=0)
    evaluator=make_evaluator(counters)
    for instance in ['E0','E1','E2']:
        data=study.make_data(instance)
        for seed in range(771001,771004):
            rng=random.Random(seed);os=_build_order_sequence(data,'random',rng);ma=_build_machine_assignment(data,'random',rng)
            base_schedule=decode(data,os,ma);base=SimpleNamespace(order_sequence=os,machine_assignment=ma,schedule=base_schedule)
            ids=_sequence_to_operation_ids(data,os)
            moves=[(v,z.machine_id) for v in ids for z in data.operations[v].candidates if z.machine_id!=ma[v]]
            sample=rng.sample(moves,50)
            # Two ways include objective reconstruction and repair; full API exactly matches solver evaluation.
            from fjsp.nsga_vns import Individual,_evaluate
            def evaluate(v,target,reuse):
                a=dict(ma);a[v]=target;candidate=Individual(list(os),a);candidate._base=base
                start=time.perf_counter_ns()
                if reuse:evaluator(candidate,data,None,None)
                else:_evaluate(candidate,data,None,None)
                return candidate,time.perf_counter_ns()-start
            for v,target in sample[:10]:evaluate(v,target,False);evaluate(v,target,True)
            for repeat in range(3):
                for trial,(v,target) in enumerate(sample):
                    timings={};solutions={}
                    for reuse in ([False,True] if (trial+repeat)%2==0 else [True,False]):
                        candidate,ns=evaluate(v,target,reuse);timings[reuse]=ns;solutions[reuse]=candidate
                    assert solutions[False].objectives==solutions[True].objectives
                    assert {k:vars(e) for k,e in solutions[False].schedule.entries.items()}=={k:vars(e) for k,e in solutions[True].schedule.entries.items()}
                    cut=ids.index(v)
                    results.append(dict(instance=instance,seed=seed,repeat=repeat,trial=trial,cut=cut,
                                        prefix_fraction=cut/len(ids),full_ns=timings[False],prefix_ns=timings[True],equal=True))
    df=pd.DataFrame(results);df.to_csv(OUT/'prefix-timing.csv',index=False,encoding='utf-8-sig')
    # Descriptive parent batches; the same3 seeds are paired across scenarios.
    # No timing hypothesis test pools these9 batches as independent instances.
    batches=df.groupby(['instance','seed'])[['full_ns','prefix_ns']].sum().reset_index()
    batches['ratio']=batches.prefix_ns/batches.full_ns
    batches.to_csv(OUT/'prefix-timing-batches.csv',index=False,encoding='utf-8-sig')
    summary=dict(candidate_pairs=len(df),parent_batches=9,paired_seeds_across_scenarios=3,repeats=3,warmup_candidates_per_parent=10,
        full_evaluations=len(df),prefix_evaluations=len(df),strictly_equal=int(df.equal.sum()),
        mean_batch_ratio=float(batches.ratio.mean()),min_batch_ratio=float(batches.ratio.min()),max_batch_ratio=float(batches.ratio.max()),
        platform=platform.platform(),processor=platform.processor(),python=sys.version,numpy=np.__version__,
        scope='Serial evaluator component only; parallel main experiment finished before timing; not whole algorithm acceleration')
    (OUT/'prefix-timing-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
