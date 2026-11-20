"""第 3 组 Markdown 数据解析器。"""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .data_model import JobSpec, MachineSpec, OperationOption, OperationSpec, ProblemData


def _clean_cell(value: str) -> str:
    return value.strip().replace("**", "")


def _to_float(value: str, default: float = 0.0) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group(0)) if match else default


def _to_int(value: str, default: int = 0) -> int:
    return int(round(_to_float(value, default)))


def _split_table(line: str) -> List[str]:
    parts = line.strip().strip("|").split("|")
    return [_clean_cell(part) for part in parts]


def _tables_after_heading(lines: List[str], heading_text: str) -> List[Dict[str, str]]:
    """读取某个二级标题下的第一个 Markdown 表格。"""

    start = None
    for index, line in enumerate(lines):
        if heading_text in line and line.startswith("##"):
            start = index + 1
            break
    if start is None:
        return []

    header_index = None
    for index in range(start, len(lines)):
        if lines[index].startswith("## "):
            break
        if lines[index].strip().startswith("|") and index + 1 < len(lines) and "---" in lines[index + 1]:
            header_index = index
            break
    if header_index is None:
        return []

    headers = _split_table(lines[header_index])
    rows: List[Dict[str, str]] = []
    for line in lines[header_index + 2 :]:
        if line.startswith("## ") or not line.strip().startswith("|"):
            if rows:
                break
            continue
        values = _split_table(line)
        if len(values) != len(headers):
            continue
        rows.append(dict(zip(headers, values)))
    return rows


def _parse_maintenance(value: str) -> List[Tuple[float, float]]:
    """解析纯数字维护窗口；中文星期窗口保留原文，由实验配置另行注入。"""

    windows: List[Tuple[float, float]] = []
    for start, end in re.findall(r"(\d+(?:\.\d+)?)\s*[-~至]\s*(\d+(?:\.\d+)?)", value):
        start_value = float(start)
        end_value = float(end)
        if end_value > start_value:
            windows.append((start_value, end_value))
    return windows


def _operation_index(operation_name: str) -> int:
    match = re.search(r"(\d+)", operation_name)
    if not match:
        raise ValueError("无法解析工序编号: {}".format(operation_name))
    return int(match.group(1))


def _matrix_after_heading(lines: List[str]) -> Tuple[List[str], List[Dict[str, str]]]:
    """读取纯编号加工时间矩阵，并保留首列工件编号。"""

    start = None
    for index, line in enumerate(lines):
        if line.startswith("##") and "加工时间矩阵" in line:
            start = index + 1
            break
    if start is None:
        return [], []
    header_index = None
    for index in range(start, len(lines)):
        if lines[index].strip().startswith("|") and index + 1 < len(lines) and "---" in lines[index + 1]:
            header_index = index
            break
    if header_index is None:
        return [], []
    headers = _split_table(lines[header_index])
    machine_ids = [header for header in headers[1:] if header]
    rows: List[Dict[str, str]] = []
    for line in lines[header_index + 2 :]:
        if line.startswith("## ") or not line.strip().startswith("|"):
            if rows:
                break
            continue
        values = _split_table(line)
        if len(values) != len(headers) or not values[0]:
            continue
        row = {"job_id": values[0]}
        row.update({machine_id: value for machine_id, value in zip(machine_ids, values[1:])})
        rows.append(row)
    return machine_ids, rows


def _machine_ids_from_text(text: str, machine_ids: List[str]) -> List[str]:
    """从路线/能力说明中提取已知机器编号，支持 M1-M5 和斜杠组合。"""

    result: List[str] = []
    # 先展开连续编号范围，避免 ``M1-M5`` 只识别到边界两台机器。
    for start, end in re.findall(r"(M\d+)\s*[-~至]\s*(M\d+)", str(text)):
        start_number = int(start[1:])
        end_number = int(end[1:])
        for machine_id in machine_ids:
            match = re.fullmatch(r"M(\d+)", machine_id)
            if match and min(start_number, end_number) <= int(match.group(1)) <= max(start_number, end_number):
                if machine_id not in result:
                    result.append(machine_id)
    for machine_id in machine_ids:
        if re.search(r"(?<![A-Za-z0-9]){}(?![A-Za-z0-9])".format(re.escape(machine_id)), str(text)):
            if machine_id not in result:
                result.append(machine_id)
    return result


def _load_generic_matrix_dataset(
    lines: List[str],
    path: Path,
    job_rows: List[Dict[str, str]],
    machine_rows: List[Dict[str, str]],
    order_rows: List[Dict[str, str]],
) -> ProblemData:
    """解析第 1/2/4/5 组使用的 ``工件 × 机器`` 纯编号矩阵。

    这些数据没有显式的工件-工序-机器长表，因此按照每个工件的工艺路线和工序
    能力说明将一行矩阵拆成多道工序。矩阵中的 0 被视为不具备加工资格。
    """

    matrix_machine_ids, matrix_rows = _matrix_after_heading(lines)
    if not matrix_rows:
        raise ValueError("未找到扩展候选机器表，也未找到加工时间矩阵")
    machine_ids = list(dict.fromkeys(matrix_machine_ids + [row.get("机器编号", "") for row in machine_rows if row.get("机器编号", "")]))
    operation_rows = _tables_after_heading(lines, "工序信息表")
    setup_rows = _tables_after_heading(lines, "工序参数与能耗口径")
    operation_labels = [row.get("工序编号", "") for row in operation_rows if row.get("工序编号", "")]
    if not operation_labels:
        route_lengths = [
            len([token for token in str(row.get("工艺路线", "")).split("->") if token.strip()])
            for row in job_rows
        ]
        operation_count = max(route_lengths or [1])
        operation_labels = ["O{}".format(index) for index in range(1, operation_count + 1)]
    operation_capabilities = {
        row.get("工序编号", ""): _machine_ids_from_text(row.get("可用机器/阶段", ""), machine_ids)
        for row in operation_rows
        if row.get("工序编号", "")
    }
    setup_by_label = {
        row.get("工序/阶段", ""): _to_float(row.get("典型换型(min)", "0"))
        for row in setup_rows
        if row.get("工序/阶段", "")
    }

    jobs: Dict[str, JobSpec] = {}
    for row in job_rows:
        job_id = row.get("工件编号", "").strip()
        if not job_id:
            continue
        jobs[job_id] = JobSpec(
            job_id=job_id,
            name=row.get("工件名称", job_id),
            family=row.get("工件类别", "未知") or "未知",
            release_time=0.0,
            due_date=0.0,
            priority="C",
            tardiness_weight=1.0,
        )
    for row in order_rows:
        job_id = row.get("工件编号", "")
        if job_id in jobs:
            jobs[job_id].release_time = _to_float(row.get("释放时间(min)", "0"))
            jobs[job_id].due_date = _to_float(row.get("承诺完工(min)", "0"))
            jobs[job_id].priority = row.get("优先级", "C") or "C"
            jobs[job_id].tardiness_weight = _to_float(row.get("延期权重", "1"), 1.0)

    machines: Dict[str, MachineSpec] = {}
    for row in machine_rows:
        machine_id = row.get("机器编号", "").strip()
        if not machine_id:
            continue
        machines[machine_id] = MachineSpec(
            machine_id=machine_id,
            running_power=_to_float(row.get("运行功率(kW)", "0")),
            idle_power=_to_float(row.get("待机功率(kW)", "0")),
            oee=_to_float(row.get("OEE", "1"), 1.0),
            mtbf_hours=_to_float(row.get("MTBF(h)", "0")),
            average_setup=_to_float(row.get("平均换型(min)", "0")),
            maintenance_text=row.get("维护窗口", ""),
            status=row.get("数据状态", "可用") or "可用",
            maintenance_windows=_parse_maintenance(row.get("维护窗口", "")),
        )
    for machine_id in machine_ids:
        machines.setdefault(machine_id, MachineSpec(machine_id=machine_id))

    matrix_by_job = {row["job_id"]: row for row in matrix_rows}
    operations: Dict[str, OperationSpec] = {}
    for job_id, job in jobs.items():
        matrix = matrix_by_job.get(job_id, {})
        route = next((row.get("工艺路线", "") for row in job_rows if row.get("工件编号", "") == job_id), "")
        route_tokens = [token.strip() for token in route.split("->") if token.strip()]
        job_operations: List[OperationSpec] = []
        for op_index, label in enumerate(operation_labels, start=1):
            token = route_tokens[op_index - 1] if op_index <= len(route_tokens) else ""
            route_machines = _machine_ids_from_text(token, machine_ids)
            capability_machines = operation_capabilities.get(label, [])
            allowed = route_machines or capability_machines or machine_ids
            candidates: List[OperationOption] = []
            for machine_id in allowed:
                processing_time = _to_float(matrix.get(machine_id, "0"))
                if processing_time <= 0:
                    continue
                machine = machines[machine_id]
                total_energy = processing_time / 60.0 * machine.running_power if machine.running_power > 0 else 0.0
                operation_id = "{}-O{}".format(job_id, op_index)
                candidates.append(OperationOption(
                    operation_id=operation_id,
                    job_id=job_id,
                    op_index=op_index,
                    machine_id=machine_id,
                    processing_time=processing_time,
                    setup_time=setup_by_label.get(label, 0.0),
                    running_energy=total_energy,
                    auxiliary_energy=0.0,
                    total_energy=total_energy,
                ))
            operation_id = "{}-O{}".format(job_id, op_index)
            operation = OperationSpec(operation_id, job_id, op_index, job.family, candidates)
            operations[operation_id] = operation
            job_operations.append(operation)
        job.operations = job_operations

    dataset = ProblemData(
        jobs=jobs,
        machines=machines,
        operations=operations,
        source_path=str(path),
        source_notes="纯编号加工时间矩阵已按工艺路线恢复候选机器；能耗按运行功率估算",
    )
    dataset.validate()
    return dataset


def load_dataset(path: Path) -> ProblemData:
    """从第 3 组 Markdown 长表生成 ProblemData。"""

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    job_rows = _tables_after_heading(lines, "工件信息表")
    candidate_rows = _tables_after_heading(lines, "工件—工序—候选机器加工数据")
    machine_rows = _tables_after_heading(lines, "设备状态与能耗参数")
    order_rows = _tables_after_heading(lines, "订单释放与交付参数")

    if not candidate_rows:
        return _load_generic_matrix_dataset(lines, path, job_rows, machine_rows, order_rows)

    jobs: Dict[str, JobSpec] = {}
    for row in job_rows:
        job_id = row.get("工件编号", "").strip()
        if not job_id:
            continue
        jobs[job_id] = JobSpec(
            job_id=job_id,
            name=row.get("工件名称", job_id),
            family=row.get("工件类别", "未知") or "未知",
            release_time=0.0,
            due_date=0.0,
            priority="C",
            tardiness_weight=1.0,
        )

    for row in order_rows:
        job_id = row.get("工件编号", "")
        if job_id not in jobs:
            continue
        jobs[job_id].release_time = _to_float(row.get("释放时间(min)", "0"))
        jobs[job_id].due_date = _to_float(row.get("承诺完工(min)", "0"))
        jobs[job_id].priority = row.get("优先级", "C") or "C"
        jobs[job_id].tardiness_weight = _to_float(row.get("延期权重", "1"), 1.0)

    machine_map: Dict[str, MachineSpec] = {}
    for row in machine_rows:
        machine_id = row.get("机器编号", "")
        if not machine_id:
            continue
        machine_map[machine_id] = MachineSpec(
            machine_id=machine_id,
            running_power=_to_float(row.get("运行功率(kW)", "0")),
            idle_power=_to_float(row.get("待机功率(kW)", "0")),
            oee=_to_float(row.get("OEE", "1"), 1.0),
            mtbf_hours=_to_float(row.get("MTBF(h)", "0")),
            average_setup=_to_float(row.get("平均换型(min)", "0")),
            maintenance_text=row.get("维护窗口", ""),
            status=row.get("数据状态", "可用"),
            maintenance_windows=_parse_maintenance(row.get("维护窗口", "")),
        )

    operations: Dict[str, OperationSpec] = {}
    for row in candidate_rows:
        job_id = row.get("工件", "")
        operation_name = row.get("工序", "")
        machine_id = row.get("候选机器", "")
        if not job_id or not operation_name or not machine_id:
            continue
        if job_id not in jobs:
            jobs[job_id] = JobSpec(job_id, job_id, "未知", 0.0, 0.0, "C", 1.0)
        op_index = _operation_index(operation_name)
        operation_id = "{}-O{}".format(job_id, op_index)
        if operation_id not in operations:
            operations[operation_id] = OperationSpec(
                operation_id=operation_id,
                job_id=job_id,
                op_index=op_index,
                family=jobs[job_id].family,
            )
        option = OperationOption(
            operation_id=operation_id,
            job_id=job_id,
            op_index=op_index,
            machine_id=machine_id,
            processing_time=_to_float(row.get("加工时间(min)", "0")),
            setup_time=_to_float(row.get("换型(min)", "0")),
            running_energy=_to_float(row.get("运行能耗(kWh)", "0")),
            auxiliary_energy=_to_float(row.get("辅助能耗(kWh)", "0")),
            total_energy=_to_float(row.get("总能耗(kWh)", "0")),
        )
        operations[operation_id].candidates.append(option)
        if machine_id not in machine_map:
            machine_map[machine_id] = MachineSpec(machine_id=machine_id)

    for job in jobs.values():
        job.operations = sorted(
            [operation for operation in operations.values() if operation.job_id == job.job_id],
            key=lambda item: item.op_index,
        )

    dataset = ProblemData(
        jobs=jobs,
        machines=machine_map,
        operations=operations,
        source_path=str(path),
        source_notes="第3组扩展FJSP；动态故障时长和插入订单为构造场景",
    )
    dataset.validate()
    return dataset
