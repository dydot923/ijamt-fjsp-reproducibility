"""Exact static common-prefix reuse; no candidates are filtered or approximated."""
from fjsp.decoder import (decode, DecodeContext, _sequence_to_operation_ids,
                          repair_machine_assignment)
from fjsp.objectives import static_objectives
from fjsp.nsga_vns import _evaluate


def prefix_cut(data, base, candidate):
    old = _sequence_to_operation_ids(data, base.order_sequence)
    new = _sequence_to_operation_ids(data, candidate.order_sequence)
    cut = 0
    for a, b in zip(old, new):
        if a != b or base.machine_assignment[a] != candidate.machine_assignment[b]:
            break
        cut += 1
    return cut, new


def make_evaluator(counters):
    def evaluate(ind, data, context=None, dynamic_baseline=None):
        ind.machine_assignment = repair_machine_assignment(data, ind.machine_assignment)
        base = getattr(ind, '_base', None)
        n = len(data.operations)
        if (context is not None or dynamic_baseline is not None or base is None
                or base.schedule is None or not base.schedule.feasible):
            counters['full_calls'] += 1
            counters['reinserted_operations'] += n
            return _evaluate(ind, data, context, dynamic_baseline)
        cut, ids = prefix_cut(data, base, ind)
        state = DecodeContext(fixed_entries={v: base.schedule.entries[v] for v in ids[:cut]},
                              active_operation_ids=set(ids[cut:]))
        ind.schedule = decode(data, ind.order_sequence, ind.machine_assignment, state)
        ind.objectives = static_objectives(data, ind.schedule)
        counters['prefix_calls'] += 1
        counters['reused_operations'] += cut
        counters['reinserted_operations'] += n-cut
        return ind
    return evaluate
