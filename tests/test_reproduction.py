"""Portable checks on constructed inputs, with no engineering/benchmark data."""
from pathlib import Path
from dataclasses import asdict
import json, random, sys, unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'replication'))
from fjsp.dataset_io import load_any_dataset
from fjsp.decoder import decode
from fjsp.objectives import static_objectives
from fjsp.nsga_vns import Individual, _evaluate
from prefix_evaluator import make_evaluator
from run_exact_oracle import construct, orientations, reachable, optimize
from run_stress import check_schedule


class ReproductionTests(unittest.TestCase):
    def test_constructed_inputs_and_independent_front(self):
        reference=json.loads((ROOT/'reference/small_instances/summary.json').read_text('utf8'))
        for i in range(6):
            path=ROOT/'examples'/f'T{i+1}.json'
            self.assertEqual(json.loads(path.read_text('utf8')),construct(i))
            data=load_any_dataset(path)
            exact,counts=orientations(data)
            np.testing.assert_array_equal(exact,np.asarray(reference[i]['exact_front']))
            self.assertEqual(counts['orientations'],5040)
        data=load_any_dataset(ROOT/'examples/T1.json')
        exact,_=orientations(data)
        decoded,_=reachable(data)
        np.testing.assert_array_equal(exact,decoded)

    def test_prefix_reuse_preserves_full_schedule(self):
        data=load_any_dataset(ROOT/'examples/T1.json')
        rng=random.Random(20261120)
        for _ in range(25):
            os=[j for j in data.job_ids for _ in data.jobs[j].operations]
            rng.shuffle(os)
            ma={v:rng.choice(data.operations[v].candidates).machine_id for v in data.operation_ids}
            parent=Individual(os,ma)
            _evaluate(parent,data,None,None)
            child=parent.clone()
            operation=rng.choice(data.operation_ids)
            alternatives=[o.machine_id for o in data.operations[operation].candidates if o.machine_id!=ma[operation]]
            child.machine_assignment[operation]=alternatives[0]
            child._base=parent
            full=child.clone()
            _evaluate(full,data,None,None)
            counters=dict(full_calls=0,prefix_calls=0,reused_operations=0,reinserted_operations=0)
            make_evaluator(counters)(child,data)
            self.assertEqual(child.objectives,full.objectives)
            self.assertEqual({k:asdict(v) for k,v in child.schedule.entries.items()},
                             {k:asdict(v) for k,v in full.schedule.entries.items()})
            self.assertFalse(check_schedule(data,child.schedule,child.objectives))
            self.assertEqual(counters['prefix_calls'],1)

    def test_budget_controlled_solver_matches_saved_reference(self):
        data=load_any_dataset(ROOT/'examples/T1.json')
        references=json.loads((ROOT/'reference/small_instances/runs.json').read_text('utf8'))
        for method in ['H-NSGA','H-MVNS-6','H-MVNS-2']:
            row=optimize(data,method,530001)
            expected=next(r for r in references if r['instance']=='T1' and r['method']==method and r['seed']==530001)
            for name in ['front','evaluations','decoder_counts','audited_schedules','invalid_schedules']:
                self.assertEqual(json.loads(json.dumps(row[name])),expected[name],(method,name))
            self.assertEqual(row['evaluations'],200)


if __name__=='__main__':unittest.main()
