"""A real small-instance run, independent enumeration and archived-result check."""
from pathlib import Path
import argparse, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'replication'))
from fjsp.dataset_io import load_any_dataset
from run_exact_oracle import orientations, reachable, optimize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance',choices=[f'T{i}' for i in range(1,7)],default='T1')
    parser.add_argument('--method',choices=['H-NSGA','H-MVNS-6','H-MVNS-2'],default='H-MVNS-2')
    parser.add_argument('--seed',type=int,default=530001)
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    data=load_any_dataset(ROOT/'examples'/f'{args.instance}.json')
    exact,counts=orientations(data)
    decoded,encodings=reachable(data)
    assert np.array_equal(exact,decoded),'Independent and decoder-reachable fronts differ'
    result=optimize(data,args.method,args.seed)
    assert result['evaluations']==200 and result['invalid_schedules']==0
    expected=json.loads((ROOT/'reference/small_instances/runs.json').read_text('utf8'))
    expected=next((r for r in expected if r['instance']==args.instance and r['method']==args.method and r['seed']==args.seed),None)
    matched=None
    if expected is not None:
        for field in ['front','audited_schedules','invalid_schedules','decoder_counts','evaluations']:
            # The live objective vectors are tuples; JSON archives store lists.
            assert json.loads(json.dumps(result[field]))==expected[field],f'Archived deterministic field differs: {field}'
        matched=True
    result.update(instance=args.instance,input_kind='constructed validation; not production measurements',
                  exact_front=exact.tolist(),decoder_front=decoded.tolist(),
                  enumeration_counts=counts,reachable_encoding_count=encodings,
                  exact_and_reachable_fronts_equal=True,archived_run_matched=matched)
    output=args.output or ROOT/'outputs'/f'demo-{args.instance}-{args.method}-{args.seed}.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),'utf8')
    print(json.dumps({'status':'PASS','instance':args.instance,'method':args.method,'evaluations':200,
                      'invalid_schedules':0,'exact_front_size':len(exact),'archived_run_matched':matched,
                      'output':output.name},indent=2),flush=True)


if __name__=='__main__':main()
