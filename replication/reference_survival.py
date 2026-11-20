"""Reference-point survival control with the existing discrete mating operators.

This isolates NSGA-III association/niching, not a reproduction of a complete
recent reinforcement-learning or continuous-domain scheduling algorithm.
"""
import numpy as np
import fjsp.nsga_vns as module

REFERENCE_DIRECTIONS=np.array([(i/7,j/7,(7-i-j)/7) for i in range(8) for j in range(8-i)]
    +[(1/3,1/3,1/3),(.5,.25,.25),(.25,.5,.25),(.25,.25,.5)],float)

def make_reference_survival(seed):
    rng=np.random.default_rng(seed+49152)
    directions=REFERENCE_DIRECTIONS/np.linalg.norm(REFERENCE_DIRECTIONS,axis=1)[:,None]
    def survival(population,target_size,policy=None,data=None):
        candidates=module._deduplicate_genotypes(population,data)
        if not candidates or target_size<=0:return []
        fronts=module.fast_non_dominated_sort(candidates)
        selected=[];split=[]
        for front in fronts:
            module.assign_crowding(front)
            if len(selected)+len(front)<=target_size:selected.extend(front)
            else:split=front;break
        if not split:return [p.clone() for p in selected[:target_size]]
        pool=selected+split
        values=np.array([p.objectives for p in pool],float)
        shifted=values-values.min(axis=0)
        extreme=[]
        for k in range(3):
            weight=np.full(3,1e-6);weight[k]=1.
            extreme.append(shifted[np.argmin((shifted/weight).max(axis=1))])
        try:
            intercept=1/np.linalg.solve(np.array(extreme),np.ones(3))
            if np.any(~np.isfinite(intercept)) or np.any(intercept<=1e-12):raise ValueError('degenerate intercept')
        except (np.linalg.LinAlgError,ValueError):intercept=shifted.max(axis=0)
        normalized=shifted/np.maximum(intercept,1e-12)
        projection=normalized@directions.T
        perpendicular=np.maximum(0,(normalized**2).sum(axis=1)[:,None]-projection**2)**.5
        associated=perpendicular.argmin(axis=1)
        counts=np.bincount(associated[:len(selected)],minlength=len(directions))
        remaining=list(range(len(selected),len(pool)))
        while remaining and len(selected)<target_size:
            available=np.unique(associated[remaining])
            niche=int(rng.choice(available[counts[available]==counts[available].min()]))
            in_niche=[i for i in remaining if associated[i]==niche]
            index=(min(in_niche,key=lambda i:perpendicular[i,niche]) if counts[niche]==0
                   else int(rng.choice(in_niche)))
            selected.append(pool[index]);remaining.remove(index);counts[niche]+=1
        return [p.clone() for p in selected]
    return survival
