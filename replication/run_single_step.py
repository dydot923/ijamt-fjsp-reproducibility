"""Same-parent single-step diagnostic, isolated from complete search trajectories."""
import csv,json,random
from pathlib import Path
from fjsp.data_model import Schedule,ScheduleEntry
from fjsp.nsga_vns import Individual,_ObjectivePolicy,_propose_neighbor,_evaluate,_score_pair,_individual_signature
from fjsp.objectives import dominates
from run_budget_study import make_data,OUT
from run_matched_study import matched_machine
from run_stress import check_schedule

def main():
    rows=json.loads((OUT/'main_runs.json').read_text(encoding='utf-8'))
    parents=sorted([r for r in rows if r['method']=='H-NSGA' and r['seed']<=410010],key=lambda r:r['seed'])
    assert len(parents)==10
    data=make_data('E0');output=[]
    for i,row in enumerate(parents):
        rep=row['representative'];schedule=Schedule(entries={e['operation_id']:ScheduleEntry(**e) for e in rep['entries']})
        parent=Individual(rep['order_sequence'],rep['machine_assignment'],schedule,tuple(rep['objectives']))
        policy=_ObjectivePolicy('multi',(1/3,)*3,(0.,)*3,tuple(parent.objectives))
        for focus in range(3):
            weights=tuple(1. if k==focus else .05 for k in range(3))
            for version in ['original','matched']:
                seen=set()
                for trial in range(20):
                    rng=random.Random(20261007+i*1000+focus*100+trial)
                    if version=='matched':candidate=matched_machine(parent,data,weights,rng)
                    else:candidate=_propose_neighbor(parent,'machine',data,policy,weights,rng)
                    sig=_individual_signature(candidate,data);duplicate=sig in seen;seen.add(sig)
                    _evaluate(candidate,data,None,None)
                    issues=check_schedule(data,candidate.schedule,candidate.objectives)
                    output.append(dict(parent_seed=row['seed'],focus=['Cmax','TEC','Lmax'][focus],version=version,trial=trial,
                        duplicate=int(duplicate),accepted=int(_score_pair(candidate,parent,policy,weights)),
                        dominates=int(dominates(candidate.objectives,parent.objectives)),invalid=int(bool(issues)),
                        delta_cmax=candidate.objectives[0]-parent.objectives[0],delta_tec=candidate.objectives[1]-parent.objectives[1],
                        delta_lmax=candidate.objectives[2]-parent.objectives[2],
                        relative_target=(candidate.objectives[focus]-parent.objectives[focus])/parent.objectives[focus]))
    with (OUT/'single_step.csv').open('w',newline='',encoding='utf-8-sig') as fh:
        w=csv.DictWriter(fh,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    print('same parents',len(parents),'actual decoded candidates',len(output),'invalid',sum(r['invalid'] for r in output))
if __name__=='__main__':main()
