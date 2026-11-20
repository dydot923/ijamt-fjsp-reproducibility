"""机器故障与订单插入的事件驱动重调度。

动态层不另起一套编码，而是复用第 3 章的解码器和 IGA-NSGA-VNS：

事件检测 -> 状态冻结 -> 影响范围识别 -> 局部重调度 -> 阈值检查 -> 完全重调度。
"""

import copy
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .config import AlgorithmConfig, DynamicConfig
from .data_model import JobSpec, ProblemData, Schedule, ScheduleEntry
from .decoder import DecodeContext
from .nsga_vns import IGA_NSGA_VNS, Individual
from .objectives import disturbance, summarize_schedule, weighted_tardiness


@dataclass
class FailureEvent:
    machine_id: str
    start_time: float
    duration: float
    synthetic: bool = True

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


@dataclass
class OrderInsertionEvent:
    event_time: float
    template_job_id: str
    new_job_id: str
    due_window: float
    priority: str = "A"
    tardiness_weight: float = 1.50
    synthetic: bool = True


@dataclass
class RescheduleResult:
    schedule: Schedule
    strategy: str
    event_type: str
    event_time: float
    affected_operation_ids: Set[str] = field(default_factory=set)
    frozen_operation_ids: Set[str] = field(default_factory=set)
    local_schedule: Optional[Schedule] = None
    full_schedule: Optional[Schedule] = None
    convergence_history: List[Dict[str, float]] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


def _unfinished(entry: ScheduleEntry, event_time: float) -> bool:
    return entry.end > event_time + 1e-9


def _active(entry: ScheduleEntry, event_time: float) -> bool:
    return entry.setup_start < event_time < entry.end


def _copy_entry(entry: ScheduleEntry, frozen: bool) -> ScheduleEntry:
    copied = ScheduleEntry(**vars(entry))
    copied.frozen = frozen
    return copied


def _job_ids_on_machine_after(schedule: Schedule, machine_id: str, event_time: float) -> Set[str]:
    return {
        entry.job_id
        for entry in schedule.entries_for_machine(machine_id)
        if _unfinished(entry, event_time)
    }


def _unfinished_operations(schedule: Schedule, event_time: float, job_ids: Optional[Set[str]] = None) -> Set[str]:
    return {
        operation_id
        for operation_id, entry in schedule.entries.items()
        if _unfinished(entry, event_time) and (job_ids is None or entry.job_id in job_ids)
    }


def _tardiness_for_jobs(data: ProblemData, schedule: Schedule, job_ids: Set[str]) -> float:
    completion: Dict[str, float] = {}
    for entry in schedule.entries.values():
        if entry.job_id in job_ids:
            completion[entry.job_id] = max(completion.get(entry.job_id, 0.0), entry.end)
    return sum(
        data.jobs[job_id].tardiness_weight * max(0.0, completion.get(job_id, 0.0) - data.jobs[job_id].due_date)
        for job_id in job_ids
        if job_id in data.jobs
    )


def _choose_candidate(front: List[Individual], weights: Tuple[float, ...] = tuple()) -> Individual:
    """从非支配前沿选择无量纲增广 Tchebycheff 折中解。"""

    feasible = [item for item in front if item.schedule is not None and item.schedule.feasible]
    candidates = feasible or front
    dimensions = list(zip(*[item.objectives for item in candidates]))
    ideal = tuple(min(values) for values in dimensions)
    nadir = tuple(max(values) for values in dimensions)
    raw = tuple(max(0.0, float(value)) for value in weights)
    if len(raw) != len(dimensions) or sum(raw) <= 1e-12:
        active = tuple(1.0 / len(dimensions) for _ in dimensions)
    else:
        active = tuple(value / sum(raw) for value in raw)

    def score(item: Individual) -> float:
        normalized = [
            (value - low) / max(1.0, high - low)
            for value, low, high in zip(item.objectives, ideal, nadir)
        ]
        weighted = [weight * value for weight, value in zip(active, normalized)]
        return max(weighted) + 0.05 * sum(weighted)

    return min(candidates, key=lambda item: (score(item), tuple(item.objectives)))


class EventDrivenRescheduler:
    """事件驱动局部—完全重调度控制器。"""

    def __init__(
        self,
        data: ProblemData,
        algorithm_config: Optional[AlgorithmConfig] = None,
        dynamic_config: Optional[DynamicConfig] = None,
    ):
        self.data = data
        configured = algorithm_config or AlgorithmConfig()
        weights = tuple(configured.objective_weights)
        if len(weights) == 3:
            weights = weights + (1.0,)
        elif len(weights) != 4:
            weights = (1.0, 1.0, 1.0, 1.0)
        self.algorithm_config = replace(configured, objective_weights=weights)
        self.dynamic_config = dynamic_config or DynamicConfig()
        self.last_history: List[Dict[str, float]] = []

    def _add_inserted_job(self, event: Optional[OrderInsertionEvent]) -> Set[str]:
        if event is None:
            return set()
        if event.new_job_id in self.data.jobs:
            return set(self.data.jobs[event.new_job_id].operations[i].operation_id for i in range(len(self.data.jobs[event.new_job_id].operations)))
        new_job = self.data.clone_job(
            template_job_id=event.template_job_id,
            new_job_id=event.new_job_id,
            release_time=event.event_time,
            due_date=event.event_time + event.due_window,
            priority=event.priority,
            tardiness_weight=event.tardiness_weight,
        )
        self.data.add_job(new_job)
        return {operation.operation_id for operation in new_job.operations}

    def _prepare_context(
        self,
        baseline: Schedule,
        failure: Optional[FailureEvent],
        insertion: Optional[OrderInsertionEvent],
        local: bool,
    ) -> Tuple[DecodeContext, Set[str], Set[str], float, Set[str]]:
        event_times = [item for item in [failure.start_time if failure else None, insertion.event_time if insertion else None] if item is not None]
        event_time = min(event_times) if event_times else 0.0
        inserted_operations = self._add_inserted_job(insertion)
        affected_jobs: Set[str] = set()
        blocked: Dict[str, List[Tuple[float, float]]] = {}
        if failure is not None:
            blocked[failure.machine_id] = [(failure.start_time, failure.end_time)]
            affected_jobs.update(_job_ids_on_machine_after(baseline, failure.machine_id, failure.start_time))
        if insertion is not None:
            candidate_machines = {
                option.machine_id
                for operation_id in inserted_operations
                for option in self.data.operations[operation_id].candidates
            }
            # 插单局部范围包括同一候选设备上、事件之后尚未执行的工件。
            for entry in baseline.entries.values():
                if _unfinished(entry, insertion.event_time) and entry.machine_id in candidate_machines:
                    affected_jobs.add(entry.job_id)
        affected_operations = set(inserted_operations)
        affected_operations.update(_unfinished_operations(baseline, event_time, affected_jobs))

        frozen_ids: Set[str] = set()
        active_ids: Set[str] = set()
        for operation_id, entry in baseline.entries.items():
            if entry.end <= event_time + 1e-9:
                frozen_ids.add(operation_id)
            elif _active(entry, event_time):
                frozen_ids.add(operation_id)
                active_ids.add(operation_id)

        if local:
            # 冻结工序即使属于受影响集合，也必须留在固定表中。先从可移动集合
            # 剔除冻结项，再构造 fixed_entries，避免工序同时不属于两侧。
            movable_ids = set(affected_operations).difference(frozen_ids)
            fixed_entries = {
                operation_id: _copy_entry(entry, operation_id in frozen_ids)
                for operation_id, entry in baseline.entries.items()
                if operation_id not in movable_ids
            }
        else:
            movable_ids = set(inserted_operations)
            movable_ids.update(
                operation_id
                for operation_id, entry in baseline.entries.items()
                if operation_id not in frozen_ids
            )
            fixed_entries = {
                operation_id: _copy_entry(entry, True)
                for operation_id, entry in baseline.entries.items()
                if operation_id in frozen_ids
            }
        # 已完成/正在加工工序不能被动态算法改写。
        movable_ids.difference_update(frozen_ids)
        context = DecodeContext(
            fixed_entries=fixed_entries,
            blocked_intervals=blocked,
            active_operation_ids=movable_ids,
        )
        return context, affected_operations, frozen_ids, event_time, inserted_operations

    def _optimize_context(self, context: DecodeContext, baseline: Schedule) -> Schedule:
        solver = IGA_NSGA_VNS(self.data, self.algorithm_config)
        front = solver.run(context=context, dynamic_baseline=baseline)
        self.last_history = list(solver.history)
        return _choose_candidate(front, tuple(self.algorithm_config.objective_weights)).schedule or Schedule(feasible=False, violations=["优化器没有返回调度结果"])

    def reschedule(
        self,
        baseline: Schedule,
        failure: Optional[FailureEvent] = None,
        insertion: Optional[OrderInsertionEvent] = None,
    ) -> RescheduleResult:
        if failure is None and insertion is None:
            return RescheduleResult(baseline.copy(), "none", "none", 0.0)
        event_types = []
        if failure is not None:
            event_types.append("machine_failure")
        if insertion is not None:
            event_types.append("order_insertion")
        event_type = "+".join(event_types)
        original_job_ids = set(self.data.jobs.keys())
        event_time = min(item for item in [failure.start_time if failure else None, insertion.event_time if insertion else None] if item is not None)

        local_context, affected, frozen_ids, event_time, inserted_ids = self._prepare_context(baseline, failure, insertion, local=True)
        local_schedule = self._optimize_context(local_context, baseline)
        local_history = list(self.last_history)
        local_disturbance = disturbance(self.data, baseline, local_schedule)
        baseline_tardiness = _tardiness_for_jobs(self.data, baseline, original_job_ids)
        local_tardiness = _tardiness_for_jobs(self.data, local_schedule, original_job_ids)
        tardiness_increase = (local_tardiness - baseline_tardiness) / max(1.0, baseline_tardiness + 1.0)
        local_acceptable = (
            local_schedule.feasible
            and local_disturbance <= self.dynamic_config.local_disturbance_threshold
            and tardiness_increase <= self.dynamic_config.local_tardiness_increase_threshold
            and set(local_schedule.entries.keys()).issuperset(self.data.operation_ids)
        )
        if local_acceptable:
            return RescheduleResult(
                schedule=local_schedule,
                strategy="local",
                event_type=event_type,
                event_time=event_time,
                affected_operation_ids=affected,
                frozen_operation_ids=frozen_ids,
                local_schedule=local_schedule,
                convergence_history=local_history,
                metadata={
                    "synthetic_event": True,
                    "disturbance": local_disturbance,
                    "tardiness_increase_ratio": tardiness_increase,
                    "inserted_operations": sorted(inserted_ids),
                },
            )

        full_context, _, full_frozen_ids, _, _ = self._prepare_context(baseline, failure, insertion, local=False)
        full_schedule = self._optimize_context(full_context, baseline)
        full_history = list(self.last_history)
        return RescheduleResult(
            schedule=full_schedule,
            strategy="complete",
            event_type=event_type,
            event_time=event_time,
            affected_operation_ids=affected,
            frozen_operation_ids=full_frozen_ids,
            local_schedule=local_schedule,
            full_schedule=full_schedule,
            convergence_history=full_history,
            metadata={
                "synthetic_event": True,
                "local_disturbance": local_disturbance,
                "local_tardiness_increase_ratio": tardiness_increase,
                "local_acceptable": False,
                "inserted_operations": sorted(inserted_ids),
            },
        )


def choose_failure_machine(data: ProblemData, baseline: Schedule) -> str:
    """选择负荷较高机器作为可解释的故障场景机器。"""

    loads: Dict[str, float] = {machine_id: 0.0 for machine_id in data.machine_ids}
    for entry in baseline.entries.values():
        loads[entry.machine_id] += entry.duration
    return max(loads, key=loads.get)


def build_failure_event(
    data: ProblemData,
    baseline: Schedule,
    ratio: float,
    duration: float,
    machine_id: Optional[str] = None,
) -> FailureEvent:
    """构造故障事件；未指定机器时自动选择基准方案中的高负荷设备。"""

    machine_id = machine_id if machine_id in data.machines else choose_failure_machine(data, baseline)
    return FailureEvent(machine_id=machine_id, start_time=baseline.makespan * ratio, duration=duration)


def build_insertion_event(
    data: ProblemData,
    baseline: Schedule,
    ratio: float,
    new_job_id: str = "J21",
    due_window: float = 180.0,
    priority: str = "A",
    tardiness_weight: float = 1.50,
) -> OrderInsertionEvent:
    template_job_id = min(data.job_ids, key=lambda job_id: (data.jobs[job_id].due_date, job_id))
    return OrderInsertionEvent(
        event_time=baseline.makespan * ratio,
        template_job_id=template_job_id,
        new_job_id=new_job_id,
        due_window=due_window,
        priority=priority,
        tardiness_weight=tardiness_weight,
    )
