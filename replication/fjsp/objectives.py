"""调度目标和 Pareto 指标。"""

import math

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .data_model import ProblemData, Schedule


def _job_completion_times(data: ProblemData, schedule: Schedule) -> Dict[str, float]:
    completion: Dict[str, float] = {}
    for entry in schedule.entries.values():
        completion[entry.job_id] = max(completion.get(entry.job_id, 0.0), entry.end)
    return completion


def weighted_tardiness(data: ProblemData, schedule: Schedule) -> float:
    completion = _job_completion_times(data, schedule)
    total = 0.0
    for job_id, job in data.jobs.items():
        finish = completion.get(job_id, float("inf"))
        total += job.tardiness_weight * max(0.0, finish - job.due_date)
    return total


def total_energy(schedule: Schedule) -> float:
    return sum(entry.energy for entry in schedule.entries.values())


def max_machine_load(data: ProblemData, schedule: Schedule) -> float:
    """最大机器负荷，以加工时间和准备时间之和衡量。"""

    loads = {
        machine_id: 0.0
        for machine_id in data.machine_ids
    }
    for entry in schedule.entries.values():
        loads[entry.machine_id] = loads.get(entry.machine_id, 0.0) + entry.duration
    return max(loads.values(), default=0.0)


def disturbance(data: ProblemData, baseline: Optional[Schedule], candidate: Schedule) -> float:
    """计算未执行工序的时间、机器和顺序变化。"""

    if baseline is None:
        return 0.0
    horizon = max(1.0, baseline.makespan)
    time_change = 0.0
    machine_changes = 0.0
    candidate_by_machine: Dict[str, List[str]] = {}
    baseline_by_machine: Dict[str, List[str]] = {}
    for machine_id in data.machine_ids:
        candidate_by_machine[machine_id] = [
            item.operation_id for item in candidate.entries_for_machine(machine_id)
        ]
        baseline_by_machine[machine_id] = [
            item.operation_id for item in baseline.entries_for_machine(machine_id)
        ]
    for operation_id, base_entry in baseline.entries.items():
        if base_entry.frozen:
            continue
        new_entry = candidate.entries.get(operation_id)
        if new_entry is None:
            continue
        time_change += abs(new_entry.start - base_entry.start) / horizon
        machine_changes += float(new_entry.machine_id != base_entry.machine_id)
    order_changes = 0.0
    for machine_id in data.machine_ids:
        base_order = baseline_by_machine[machine_id]
        new_order = candidate_by_machine[machine_id]
        common = [item for item in base_order if item in new_order]
        for left, right in zip(common, new_order):
            if left != right:
                order_changes += 1.0
    scale = max(1.0, len(baseline.entries))
    return (time_change + machine_changes + order_changes) / scale


def static_objectives(data: ProblemData, schedule: Schedule) -> Tuple[float, float, float]:
    """本章静态三目标：Cmax、总能耗 TEC 和最大机器负荷 Lmax。"""

    penalty = 1e9 if not schedule.feasible else 0.0
    return (
        schedule.makespan + penalty,
        total_energy(schedule) + penalty,
        max_machine_load(data, schedule) + penalty,
    )


def dynamic_objectives(data: ProblemData, schedule: Schedule, baseline: Schedule) -> Tuple[float, float, float, float]:
    """动态四目标：Cmax、加权延期、总能耗与重调度扰动度。"""

    penalty = 1e9 if not schedule.feasible else 0.0
    return (
        schedule.makespan + penalty,
        weighted_tardiness(data, schedule) + penalty,
        total_energy(schedule) + penalty,
        disturbance(data, baseline, schedule) + penalty,
    )


def dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b + 1e-9 for a, b in zip(left, right)) and any(a < b - 1e-9 for a, b in zip(left, right))


def pareto_front(vectors: Sequence[Sequence[float]]) -> List[int]:
    return [
        index for index, vector in enumerate(vectors)
        if not any(dominates(other, vector) for other in vectors)
    ]


def _normalize(values: List[float]) -> List[float]:
    low = min(values)
    high = max(values)
    if high - low < 1e-12:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _dominated_area_2d(points: Sequence[Sequence[float]], reference: Sequence[float]) -> float:
    """计算共享右上参考点的二维矩形并集面积。"""

    if not points:
        return 0.0
    y_breaks = sorted(set([point[0] for point in points] + [reference[0]]))
    area = 0.0
    for left, right in zip(y_breaks, y_breaks[1:]):
        active_z = [point[1] for point in points if point[0] <= left + 1e-12]
        if active_z:
            area += max(0.0, right - left) * max(0.0, reference[1] - min(active_z))
    return area


def _dominated_volume(points: Sequence[Sequence[float]], reference: Sequence[float]) -> float:
    """按第一维切片，精确计算低维最小化前沿的支配超体积。"""

    if not points:
        return 0.0
    if len(reference) == 1:
        return max(0.0, reference[0] - min(point[0] for point in points))
    breaks = sorted(set([point[0] for point in points] + [reference[0]]))
    volume = 0.0
    for left, right in zip(breaks, breaks[1:]):
        active = [point[1:] for point in points if point[0] <= left + 1e-12]
        if active:
            volume += max(0.0, right - left) * _dominated_volume(active, reference[1:])
    return volume


def approximate_hypervolume(
    vectors: Sequence[Sequence[float]],
    ideal_point: Optional[Sequence[float]] = None,
    reference_point: Optional[Sequence[float]] = None,
) -> float:
    """返回最小化问题的归一化 HV。

    旧实现对每一代前沿单独做 min-max 归一化，单点前沿和端点前沿经常得到 0，
    也无法跨代比较。本实现使用固定理想点/参考点，并精确计算三目标支配体积。
    """

    if not vectors:
        return 0.0
    points = [tuple(float(value) for value in vector) for vector in vectors]
    dimension_count = len(points[0])
    if any(len(point) != dimension_count for point in points):
        raise ValueError("目标向量维度不一致")
    dimensions = list(zip(*points))
    ideal = tuple(ideal_point) if ideal_point is not None else tuple(min(values) for values in dimensions)
    if reference_point is None:
        reference = []
        for index, values in enumerate(dimensions):
            high = max(values)
            span = max(1.0, high - ideal[index], abs(high) * 0.05)
            reference.append(high + 0.20 * span)
        reference = tuple(reference)
    else:
        reference = tuple(float(value) for value in reference_point)
    if len(ideal) != dimension_count or len(reference) != dimension_count:
        raise ValueError("理想点、参考点与目标向量维度不一致")

    clipped = [
        tuple(max(ideal[index], min(point[index], reference[index])) for index in range(dimension_count))
        for point in points
        if all(point[index] < reference[index] - 1e-12 for index in range(dimension_count))
    ]
    front = [clipped[index] for index in pareto_front(clipped)] if clipped else []
    box_volume = 1.0
    for low, high in zip(ideal, reference):
        box_volume *= max(1e-12, high - low)
    if dimension_count == 1:
        volume = max(0.0, reference[0] - min(point[0] for point in front)) if front else 0.0
    elif dimension_count <= 4:
        volume = _dominated_volume(front, reference)
    else:
        # 五维以上保留稳定、保守的近似，避免递归切片的组合开销。
        volume = max(
            math.prod(max(0.0, reference[index] - point[index]) for index in range(dimension_count))
            for point in front
        ) if front else 0.0
    return max(0.0, min(1.0, volume / box_volume))


def summarize_schedule(data: ProblemData, schedule: Schedule, baseline: Optional[Schedule] = None) -> Dict[str, float]:
    result = {
        "feasible": float(schedule.feasible),
        "makespan": schedule.makespan,
        "total_energy": total_energy(schedule),
        "max_machine_load": max_machine_load(data, schedule),
        "operation_count": float(len(schedule.entries)),
    }
    if baseline is not None:
        result["disturbance"] = disturbance(data, baseline, schedule)
    return result
