"""调度论文常用实验指标。

``objectives.py`` 保留算法评价函数；本模块面向实验报告，提供可比较的 HV、IGD、
Spacing、负荷均衡和计划保留率等指标，避免把报告统计逻辑混进优化器。
"""

import math
from typing import Iterable, List, Optional, Sequence, Tuple

from .data_model import ProblemData, Schedule
from .objectives import disturbance, pareto_front, summarize_schedule


def _as_points(vectors: Sequence[Sequence[float]]) -> List[Tuple[float, ...]]:
    return [tuple(float(value) for value in vector) for vector in vectors]


def _normalize_points(vectors: Sequence[Sequence[float]], reference: Optional[Sequence[Sequence[float]]] = None) -> List[Tuple[float, ...]]:
    points = _as_points(vectors)
    all_points = points + (_as_points(reference) if reference else [])
    if not all_points:
        return []
    dimension_count = len(all_points[0])
    lows = [min(point[index] for point in all_points) for index in range(dimension_count)]
    highs = [max(point[index] for point in all_points) for index in range(dimension_count)]
    return [
        tuple((point[index] - lows[index]) / (highs[index] - lows[index]) if highs[index] > lows[index] else 0.0 for index in range(dimension_count))
        for point in points
    ]


def generational_distance(front: Sequence[Sequence[float]], reference: Sequence[Sequence[float]]) -> float:
    """GD：当前前沿到参考前沿的平均最近距离。"""

    points = _normalize_points(front, reference)
    refs = _normalize_points(reference, front)
    if not points or not refs:
        return float("inf")
    distances = [min(math.sqrt(sum((a - b) ** 2 for a, b in zip(point, ref))) for ref in refs) for point in points]
    return sum(distances) / len(distances)


def inverted_generational_distance(front: Sequence[Sequence[float]], reference: Sequence[Sequence[float]]) -> float:
    """IGD：参考前沿到当前前沿的平均最近距离。"""

    return generational_distance(reference, front)


def spacing(front: Sequence[Sequence[float]]) -> float:
    """Spacing：相邻解距离的离散程度，越小表示分布越均匀。"""

    points = _normalize_points(front)
    if len(points) <= 2:
        return 0.0
    distances = []
    for index, point in enumerate(points):
        nearest = min(
            math.sqrt(sum((a - b) ** 2 for a, b in zip(point, other)))
            for other_index, other in enumerate(points) if index != other_index
        )
        distances.append(nearest)
    mean_distance = sum(distances) / len(distances)
    return math.sqrt(sum((distance - mean_distance) ** 2 for distance in distances) / (len(distances) - 1))


def hypervolume(front: Sequence[Sequence[float]], reference_point: Optional[Sequence[float]] = None) -> float:
    """归一化三目标 HV 近似值，适用于论文内同一批次算法比较。"""

    points = _normalize_points(front)
    if not points:
        return 0.0
    dimension_count = len(points[0])
    ref = tuple(reference_point) if reference_point is not None else tuple(1.0 for _ in range(dimension_count))
    # 三目标元启发式对比通常只需一个稳定的相对 HV；重叠体积采用保守求和近似。
    return sum(math.prod(max(0.0, ref[index] - point[index]) for index in range(dimension_count)) for point in points) / len(points)


def machine_load_ratios(data: ProblemData, schedule: Schedule) -> List[float]:
    horizon = max(1.0, schedule.makespan)
    return [sum(entry.end - entry.start for entry in schedule.entries_for_machine(machine_id)) / horizon for machine_id in data.machine_ids]


def load_balance_index(data: ProblemData, schedule: Schedule) -> float:
    """机器负荷标准差，越小表示负荷越均衡。"""

    values = machine_load_ratios(data, schedule)
    if len(values) <= 1:
        return 0.0
    mean_value = sum(values) / len(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def schedule_retention(data: ProblemData, baseline: Schedule, candidate: Schedule) -> float:
    """计划保留率：未冻结工序中开始时间和机器均未改变的比例。"""

    comparable = [entry for entry in baseline.entries.values() if not entry.frozen and entry.operation_id in candidate.entries]
    if not comparable:
        return 1.0
    retained = sum(
        1 for entry in comparable
        if abs(candidate.entries[entry.operation_id].start - entry.start) <= 1e-9
        and candidate.entries[entry.operation_id].machine_id == entry.machine_id
    )
    return retained / len(comparable)


def response_metrics(data: ProblemData, baseline: Schedule, candidate: Schedule) -> dict:
    result = summarize_schedule(data, candidate, baseline)
    result.update({
        "load_balance_index": load_balance_index(data, candidate),
        "schedule_retention": schedule_retention(data, baseline, candidate),
        "machine_change_count": sum(
            1 for operation_id, base_entry in baseline.entries.items()
            if operation_id in candidate.entries and candidate.entries[operation_id].machine_id != base_entry.machine_id
        ),
    })
    return result
