"""双层编码解码器。

解码器是整个算法的可行性核心：遗传算子只负责产生候选编码，所有工艺、机器、
换型、维护窗口和事件冻结约束都在这里统一检查和修复。
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .data_model import OperationOption, ProblemData, Schedule, ScheduleEntry


@dataclass
class DecodeContext:
    """动态重调度时传入的固定状态。"""

    fixed_entries: Dict[str, ScheduleEntry] = field(default_factory=dict)
    blocked_intervals: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    active_operation_ids: Optional[Set[str]] = None
    forced_machines: Dict[str, str] = field(default_factory=dict)


def _overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> bool:
    return left_start < right_end - 1e-9 and right_start < left_end - 1e-9


def _family_of_job(data: ProblemData, job_id: str) -> str:
    return data.jobs[job_id].family if job_id in data.jobs else "未知"


def _setup_for_option(data: ProblemData, option: OperationOption, predecessor: Optional[ScheduleEntry]) -> float:
    """同工件族连续加工不重复计换型；跨族时采用候选表换型时间。"""

    if predecessor is None:
        return option.setup_time
    previous_family = _family_of_job(data, predecessor.job_id)
    current_family = _family_of_job(data, option.job_id)
    return option.setup_time if previous_family != current_family else 0.0


def _conflicts_with_intervals(start: float, end: float, intervals: Iterable[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    for interval_start, interval_end in intervals:
        if _overlap(start, end, interval_start, interval_end):
            return interval_start, interval_end
    return None


def _move_after_unavailable(
    process_start: float,
    processing_time: float,
    setup_time: float,
    blocked: Sequence[Tuple[float, float]],
    maintenance: Sequence[Tuple[float, float]],
) -> float:
    """将工序整体右移到所有不可用区间之后。"""

    candidate = process_start
    for _ in range(len(blocked) + len(maintenance) + 2):
        setup_start = candidate - setup_time
        end = candidate + processing_time
        conflict = _conflicts_with_intervals(setup_start, end, list(blocked) + list(maintenance))
        if conflict is None:
            return candidate
        candidate = conflict[1] + setup_time
    return candidate


def _find_earliest_slot(
    data: ProblemData,
    option: OperationOption,
    ready_time: float,
    existing_entries: List[ScheduleEntry],
    blocked: Sequence[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    """在指定机器的已有工序之间寻找最早可行插入位置。"""

    machine = data.machines[option.machine_id]
    ordered = sorted(existing_entries, key=lambda item: (item.setup_start, item.start, item.operation_id))
    for gap_index in range(len(ordered) + 1):
        predecessor = ordered[gap_index - 1] if gap_index > 0 else None
        successor = ordered[gap_index] if gap_index < len(ordered) else None
        setup_time = _setup_for_option(data, option, predecessor)
        # 首道工序的准备阶段也不能早于工件释放时间；有前序工序时，
        # 换型从前序工序结束后开始，确保 CSV 和甘特图不存在负时间。
        earliest = max(ready_time + setup_time, predecessor.end + setup_time if predecessor else ready_time + setup_time)
        process_start = _move_after_unavailable(
            earliest,
            option.processing_time,
            setup_time,
            blocked,
            machine.maintenance_windows,
        )
        process_end = process_start + option.processing_time
        successor_setup_start = successor.setup_start if successor else float("inf")
        if successor is not None:
            # Inserting an operation changes the predecessor of the old successor.
            # The old successor is already fixed on the timeline. Retain this gap
            # only when its required setup duration remains exactly unchanged.
            successor_option = data.option(successor.operation_id, successor.machine_id)
            same_family = _family_of_job(data, option.job_id) == _family_of_job(data, successor.job_id)
            required_successor_setup = 0.0 if same_family else successor_option.setup_time
            if abs(required_successor_setup - successor.setup_time) > 1e-9:
                continue
        if process_end <= successor_setup_start + 1e-9:
            return process_start - setup_time, process_start, process_end, setup_time
    # 正常情况下最后一个间隙一定可行；保留兜底逻辑避免异常数据导致死循环。
    predecessor = ordered[-1] if ordered else None
    setup_time = _setup_for_option(data, option, predecessor)
    earliest = max(ready_time + setup_time, predecessor.end + setup_time if predecessor else ready_time + setup_time)
    process_start = _move_after_unavailable(
        earliest,
        option.processing_time,
        setup_time,
        blocked,
        machine.maintenance_windows,
    )
    return process_start - setup_time, process_start, process_start + option.processing_time, setup_time


def _sequence_to_operation_ids(data: ProblemData, order_sequence: Sequence[str]) -> List[str]:
    """将重复工件编号序列映射为工序编号。"""

    occurrence: Dict[str, int] = {job_id: 0 for job_id in data.job_ids}
    operation_ids: List[str] = []
    for job_id in order_sequence:
        if job_id not in data.jobs:
            continue
        operation_index = occurrence.get(job_id, 0)
        if operation_index >= len(data.jobs[job_id].operations):
            continue
        operation_ids.append(data.jobs[job_id].operations[operation_index].operation_id)
        occurrence[job_id] = operation_index + 1
    return operation_ids


def repair_machine_assignment(data: ProblemData, assignment: Dict[str, str]) -> Dict[str, str]:
    """将机器分配修复到合法候选集合。"""

    repaired: Dict[str, str] = {}
    for operation_id, operation in data.operations.items():
        choices = [option.machine_id for option in operation.candidates]
        selected = assignment.get(operation_id)
        repaired[operation_id] = selected if selected in choices else choices[0]
    return repaired


def decode(
    data: ProblemData,
    order_sequence: Sequence[str],
    machine_assignment: Dict[str, str],
    context: Optional[DecodeContext] = None,
) -> Schedule:
    """把 OS/MA 双层编码解码为调度方案。"""

    context = context or DecodeContext()
    active_ids = context.active_operation_ids
    entries: Dict[str, ScheduleEntry] = {
        operation_id: ScheduleEntry(**vars(entry))
        for operation_id, entry in context.fixed_entries.items()
    }
    machine_entries: Dict[str, List[ScheduleEntry]] = {machine_id: [] for machine_id in data.machine_ids}
    for entry in entries.values():
        machine_entries.setdefault(entry.machine_id, []).append(entry)

    assignment = repair_machine_assignment(data, machine_assignment)
    assignment.update(context.forced_machines)
    violations: List[str] = []
    occurrence: Dict[str, int] = {job_id: 0 for job_id in data.job_ids}
    sequence_operation_ids = _sequence_to_operation_ids(data, order_sequence)

    for operation_id in sequence_operation_ids:
        if active_ids is not None and operation_id not in active_ids:
            occurrence[data.operations[operation_id].job_id] = data.operations[operation_id].op_index
            continue
        if operation_id in entries:
            continue
        operation = data.operations.get(operation_id)
        if operation is None:
            violations.append("未知工序 {}".format(operation_id))
            continue
        job = data.jobs[operation.job_id]
        machine_id = assignment.get(operation_id)
        option = next((item for item in operation.candidates if item.machine_id == machine_id), None)
        if option is None:
            violations.append("工序 {} 的机器分配非法".format(operation_id))
            option = operation.candidates[0]
            machine_id = option.machine_id

        predecessor_end = job.release_time
        if operation.op_index > 1:
            predecessor_id = "{}-O{}".format(job.job_id, operation.op_index - 1)
            if predecessor_id in entries:
                predecessor_end = entries[predecessor_id].end
            elif active_ids is not None and predecessor_id in active_ids:
                violations.append("工序 {} 的前序工序 {} 尚未排入".format(operation_id, predecessor_id))
            else:
                violations.append("工序 {} 缺少前序工序 {}".format(operation_id, predecessor_id))

        setup_start, process_start, process_end, setup_time = _find_earliest_slot(
            data,
            option,
            predecessor_end,
            machine_entries.setdefault(machine_id, []),
            context.blocked_intervals.get(machine_id, []),
        )
        entry = ScheduleEntry(
            operation_id=operation_id,
            job_id=operation.job_id,
            op_index=operation.op_index,
            machine_id=machine_id,
            setup_start=setup_start,
            start=process_start,
            end=process_end,
            processing_time=option.processing_time,
            setup_time=setup_time,
            energy=option.total_energy,
        )
        entries[operation_id] = entry
        machine_entries[machine_id].append(entry)

    expected_ids = set(data.operation_ids) if active_ids is None else set(active_ids)
    missing = expected_ids.difference(entries.keys())
    if missing:
        violations.append("存在未排入工序: {}".format(",".join(sorted(missing))))

    # 对新增的调度条目再做一次重叠检查；固定条目之间的冲突保留为外部状态，不在这里移动。
    for machine_id, machine_schedule in machine_entries.items():
        ordered = sorted(machine_schedule, key=lambda item: (item.setup_start, item.start, item.operation_id))
        for previous, current in zip(ordered, ordered[1:]):
            if previous.end > current.setup_start + 1e-9:
                violations.append("机器 {} 工序 {} 与 {} 重叠".format(machine_id, previous.operation_id, current.operation_id))

    return Schedule(entries=entries, feasible=not violations, violations=violations)
