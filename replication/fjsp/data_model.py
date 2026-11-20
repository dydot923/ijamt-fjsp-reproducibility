"""柔性作业车间数据结构。

算法层只使用本文件中的类型，不直接依赖 Markdown 表头或中文字段。
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class OperationOption:
    """某道工序在一台候选机器上的加工选项。"""

    operation_id: str
    job_id: str
    op_index: int
    machine_id: str
    processing_time: float
    setup_time: float
    running_energy: float
    auxiliary_energy: float
    total_energy: float


@dataclass
class OperationSpec:
    """工序及其候选机器集合。"""

    operation_id: str
    job_id: str
    op_index: int
    family: str
    candidates: List[OperationOption] = field(default_factory=list)


@dataclass
class JobSpec:
    """工件/订单信息。"""

    job_id: str
    name: str
    family: str
    release_time: float
    due_date: float
    priority: str
    tardiness_weight: float
    operations: List[OperationSpec] = field(default_factory=list)


@dataclass
class MachineSpec:
    """机器属性和可用性信息。"""

    machine_id: str
    running_power: float = 0.0
    idle_power: float = 0.0
    oee: float = 1.0
    mtbf_hours: float = 0.0
    average_setup: float = 0.0
    maintenance_text: str = ""
    status: str = "可用"
    maintenance_windows: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class ScheduleEntry:
    """调度结果中的一道工序。"""

    operation_id: str
    job_id: str
    op_index: int
    machine_id: str
    setup_start: float
    start: float
    end: float
    processing_time: float
    setup_time: float
    energy: float
    frozen: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.setup_start


@dataclass
class Schedule:
    """完整调度方案。"""

    entries: Dict[str, ScheduleEntry] = field(default_factory=dict)
    feasible: bool = True
    violations: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def copy(self) -> "Schedule":
        return Schedule(
            entries={key: ScheduleEntry(**vars(value)) for key, value in self.entries.items()},
            feasible=self.feasible,
            violations=list(self.violations),
            metadata=dict(self.metadata),
        )

    def entries_for_machine(self, machine_id: str) -> List[ScheduleEntry]:
        return sorted(
            [entry for entry in self.entries.values() if entry.machine_id == machine_id],
            key=lambda item: (item.setup_start, item.start, item.operation_id),
        )

    def entries_for_job(self, job_id: str) -> List[ScheduleEntry]:
        return sorted(
            [entry for entry in self.entries.values() if entry.job_id == job_id],
            key=lambda item: item.op_index,
        )

    @property
    def makespan(self) -> float:
        return max((entry.end for entry in self.entries.values()), default=0.0)


@dataclass
class ProblemData:
    """论文第 3 组扩展 FJSP 数据。"""

    jobs: Dict[str, JobSpec]
    machines: Dict[str, MachineSpec]
    operations: Dict[str, OperationSpec]
    source_path: str = ""
    source_notes: str = ""

    @property
    def job_ids(self) -> List[str]:
        return sorted(self.jobs.keys())

    @property
    def operation_ids(self) -> List[str]:
        return [operation.operation_id for job_id in self.job_ids for operation in self.jobs[job_id].operations]

    @property
    def machine_ids(self) -> List[str]:
        # 同时兼容 M1/M2 和 Laser-A、机台-01 等任意机器命名，避免 int/str 混合排序异常。
        def sort_key(item: str) -> Tuple[int, str]:
            import re
            match = re.search(r"(\d+)$", str(item))
            return (int(match.group(1)), str(item)) if match else (10**9, str(item))
        return sorted(self.machines.keys(), key=sort_key)

    def operation(self, operation_id: str) -> OperationSpec:
        return self.operations[operation_id]

    def option(self, operation_id: str, machine_id: str) -> OperationOption:
        for option in self.operations[operation_id].candidates:
            if option.machine_id == machine_id:
                return option
        raise KeyError("No option for {} on {}".format(operation_id, machine_id))

    def validate(self) -> None:
        """检查关键数据约束，解析错误尽早暴露。"""

        if not self.jobs:
            raise ValueError("数据中没有工件")
        if not self.machines:
            raise ValueError("数据中没有机器")
        if not self.operations:
            raise ValueError("数据中没有工序")
        for job in self.jobs.values():
            if not job.operations:
                raise ValueError("工件 {} 没有工序".format(job.job_id))
            for operation in job.operations:
                if not operation.candidates:
                    raise ValueError("工序 {} 没有候选机器".format(operation.operation_id))
                for option in operation.candidates:
                    if option.processing_time <= 0:
                        raise ValueError("工序 {} 在 {} 上的加工时间必须大于0".format(operation.operation_id, option.machine_id))
                    if option.machine_id not in self.machines:
                        raise ValueError("{} 使用了未定义机器 {}".format(operation.operation_id, option.machine_id))
        for job in self.jobs.values():
            indices = [operation.op_index for operation in job.operations]
            if indices != list(range(1, len(indices) + 1)):
                raise ValueError("工件 {} 的工序序号必须从1连续编号，当前为 {}".format(job.job_id, indices))

    def clone_job(self, template_job_id: str, new_job_id: str, release_time: float, due_date: float,
                  priority: str, tardiness_weight: float) -> JobSpec:
        """按既有工件类型生成仿真插入订单。"""

        template = self.jobs[template_job_id]
        operations: List[OperationSpec] = []
        for operation in template.operations:
            new_operation_id = "{}-O{}".format(new_job_id, operation.op_index)
            candidates = [
                OperationOption(
                    operation_id=new_operation_id,
                    job_id=new_job_id,
                    op_index=option.op_index,
                    machine_id=option.machine_id,
                    processing_time=option.processing_time,
                    setup_time=option.setup_time,
                    running_energy=option.running_energy,
                    auxiliary_energy=option.auxiliary_energy,
                    total_energy=option.total_energy,
                )
                for option in operation.candidates
            ]
            operations.append(
                OperationSpec(
                    operation_id=new_operation_id,
                    job_id=new_job_id,
                    op_index=operation.op_index,
                    family=operation.family,
                    candidates=candidates,
                )
            )
        return JobSpec(
            job_id=new_job_id,
            name="{}-紧急插入".format(template.name),
            family=template.family,
            release_time=release_time,
            due_date=due_date,
            priority=priority,
            tardiness_weight=tardiness_weight,
            operations=operations,
        )

    def add_job(self, job: JobSpec) -> None:
        self.jobs[job.job_id] = job
        for operation in job.operations:
            self.operations[operation.operation_id] = operation
