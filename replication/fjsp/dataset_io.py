"""通用调度数据输入层。

支持：
- 论文当前 Markdown 数据；
- JSON schema（推荐作为通用交换格式）；
- 长表 CSV；
- Excel 的 jobs / operations / machines 工作表；
- 独立加工时间矩阵和能耗矩阵。

所有输入最后都转换为 :class:`ProblemData`，算法层无需关心来源格式。
"""

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .data_loader import load_dataset as load_markdown_dataset
from .data_model import JobSpec, MachineSpec, OperationOption, OperationSpec, ProblemData


ALIASES = {
    "job_id": ["job_id", "job", "工件", "工件编号", "订单", "订单编号"],
    "job_name": ["job_name", "name", "工件名称", "订单名称"],
    "family": ["family", "工件类别", "类别", "族"],
    "op_id": ["operation_id", "op_id", "operation", "工序", "工序编号"],
    "op_index": ["op_index", "operation_index", "op_no", "工序序号", "工序编号"],
    "machine_id": ["machine_id", "machine", "候选机器", "机器", "机器编号"],
    "processing_time": ["processing_time", "ptime", "time", "加工时间", "加工时间(min)", "processing_time(min)"],
    "setup_time": ["setup_time", "setup", "换型", "换型时间", "换型(min)"],
    "running_energy": ["running_energy", "运行能耗", "运行能耗(kWh)"],
    "auxiliary_energy": ["auxiliary_energy", "辅助能耗", "辅助能耗(kWh)"],
    "total_energy": ["total_energy", "energy", "能耗", "总能耗", "总能耗(kWh)"],
    "release_time": ["release_time", "release", "释放时间", "释放时间(min)"],
    "due_date": ["due_date", "due", "承诺完工", "交期", "承诺完工(min)"],
    "priority": ["priority", "优先级"],
    "tardiness_weight": ["tardiness_weight", "delay_weight", "延期权重"],
    "running_power": ["running_power", "运行功率", "运行功率(kW)"],
    "idle_power": ["idle_power", "待机功率", "待机功率(kW)"],
    "oee": ["oee", "OEE"],
    "mtbf_hours": ["mtbf_hours", "mtbf", "MTBF", "MTBF(h)"],
    "average_setup": ["average_setup", "平均换型", "平均换型(min)"],
    "status": ["status", "数据状态", "设备状态"],
    "maintenance": ["maintenance", "维护窗口"],
}


def _key(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "").replace("_", "")


def _get(row: Mapping[str, Any], field: str, default: Any = "") -> Any:
    normalized = {_key(key): value for key, value in row.items()}
    for alias in ALIASES[field]:
        if _key(alias) in normalized:
            value = normalized[_key(alias)]
            if value is not None and str(value).strip() != "":
                return value
    return default


def _normalise_records(value: Any, id_field: str) -> List[Mapping[str, Any]]:
    """允许 JSON 用数组或 ``{id: attributes}`` 映射表达实体。"""

    if value is None:
        return []
    if isinstance(value, dict):
        if id_field in value or "job_id" in value or "machine_id" in value:
            return [value]
        result: List[Mapping[str, Any]] = []
        for key, attributes in value.items():
            if isinstance(attributes, dict):
                item = dict(attributes)
                item.setdefault(id_field, key)
            else:
                item = {id_field: key, "value": attributes}
            result.append(item)
        return result
    if isinstance(value, (list, tuple)):
        return [{id_field: item} if isinstance(item, (str, int, float)) else item for item in value]
    return []


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else default


def _operation_index(value: Any, fallback: int = 1) -> int:
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else fallback


def _machine_sort(machine_id: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", machine_id)
    return (int(match.group(1)), machine_id) if match else (10**9, machine_id)


def records_to_problem(
    operation_records: Sequence[Mapping[str, Any]],
    job_records: Optional[Sequence[Mapping[str, Any]]] = None,
    machine_records: Optional[Sequence[Mapping[str, Any]]] = None,
    source_path: str = "",
) -> ProblemData:
    """将长表记录转换成 ProblemData。"""

    operation_records = _normalise_records(operation_records, "operation_id")
    job_records = _normalise_records(job_records, "job_id")
    machine_records = _normalise_records(machine_records, "machine_id")
    jobs: Dict[str, JobSpec] = {}
    machines: Dict[str, MachineSpec] = {}
    operations: Dict[str, OperationSpec] = {}

    for row in job_records or []:
        job_id = str(_get(row, "job_id", "")).strip()
        if not job_id:
            continue
        jobs[job_id] = JobSpec(
            job_id=job_id,
            name=str(_get(row, "job_name", job_id)),
            family=str(_get(row, "family", "未知")) or "未知",
            release_time=_float(_get(row, "release_time", 0)),
            due_date=_float(_get(row, "due_date", 0)),
            priority=str(_get(row, "priority", "C")) or "C",
            tardiness_weight=_float(_get(row, "tardiness_weight", 1), 1.0),
        )

    for row in machine_records or []:
        machine_id = str(_get(row, "machine_id", "")).strip()
        if not machine_id:
            continue
        machines[machine_id] = MachineSpec(
            machine_id=machine_id,
            running_power=_float(_get(row, "running_power", 0)),
            idle_power=_float(_get(row, "idle_power", 0)),
            oee=_float(_get(row, "oee", 1), 1.0),
            mtbf_hours=_float(_get(row, "mtbf_hours", 0)),
            average_setup=_float(_get(row, "average_setup", 0)),
            maintenance_text=str(_get(row, "maintenance", "")),
            status=str(_get(row, "status", "可用")) or "可用",
        )

    for row in operation_records:
        job_id = str(_get(row, "job_id", "")).strip()
        if not job_id:
            continue
        op_raw = _get(row, "op_id", _get(row, "op_index", "1"))
        op_index = _operation_index(op_raw)
        operation_id = "{}-O{}".format(job_id, op_index)
        machine_id = str(_get(row, "machine_id", "")).strip()
        if not machine_id:
            continue
        family = str(_get(row, "family", jobs.get(job_id, JobSpec(job_id, job_id, "未知", 0, 0, "C", 1)).family)) or "未知"
        if job_id not in jobs:
            jobs[job_id] = JobSpec(
                job_id=job_id,
                name=str(_get(row, "job_name", job_id)),
                family=family,
                release_time=_float(_get(row, "release_time", 0)),
                due_date=_float(_get(row, "due_date", 0)),
                priority=str(_get(row, "priority", "C")) or "C",
                tardiness_weight=_float(_get(row, "tardiness_weight", 1), 1.0),
            )
        elif family != "未知" and jobs[job_id].family == "未知":
            jobs[job_id].family = family
        if operation_id not in operations:
            operations[operation_id] = OperationSpec(
                operation_id=operation_id,
                job_id=job_id,
                op_index=op_index,
                family=jobs[job_id].family,
            )
        running_energy = _float(_get(row, "running_energy", 0))
        auxiliary_energy = _float(_get(row, "auxiliary_energy", 0))
        total_energy = _float(_get(row, "total_energy", running_energy + auxiliary_energy), running_energy + auxiliary_energy)
        option = OperationOption(
            operation_id=operation_id,
            job_id=job_id,
            op_index=op_index,
            machine_id=machine_id,
            processing_time=_float(_get(row, "processing_time", 0)),
            setup_time=_float(_get(row, "setup_time", 0)),
            running_energy=running_energy,
            auxiliary_energy=auxiliary_energy,
            total_energy=total_energy,
        )
        operations[operation_id].candidates.append(option)
        if machine_id not in machines:
            machines[machine_id] = MachineSpec(machine_id=machine_id)

    for job in jobs.values():
        job.operations = sorted(
            [operation for operation in operations.values() if operation.job_id == job.job_id],
            key=lambda item: item.op_index,
        )
    problem = ProblemData(jobs=jobs, machines=machines, operations=operations, source_path=source_path)
    problem.validate()
    return problem


def _energy_lookup(energy_matrix: Optional[Any]) -> Dict[Tuple[str, int, str], float]:
    """将能耗矩阵统一展开为 ``(job, operation, machine)`` 索引。

    这里不限制机器编号必须是 ``M1`` 这类形式。论文数据通常使用 M 开头的编号，
    但实际车间也可能使用 ``Laser-A``、``机台-01`` 等名称。
    """

    if energy_matrix is None:
        return {}
    records = _matrix_records(energy_matrix, None)
    lookup: Dict[Tuple[str, int, str], float] = {}
    for row in records:
        # 递归展开时没有能耗字段，矩阵数值会暂存在 processing_time 中。
        value = _float(row.get("total_energy", 0))
        if value <= 0:
            value = _float(row.get("processing_time", 0))
        lookup[(str(row["job_id"]), int(row["op_index"]), str(row["machine_id"]))] = value
    return lookup


def _merge_matrix_records(
    base_records: Sequence[Mapping[str, Any]],
    processing_matrix: Any,
    energy_matrix: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """把独立矩阵覆盖/补充到长表，不丢失工件和机器元数据。"""

    matrix_records = _matrix_records(processing_matrix, energy_matrix)
    if not base_records:
        return matrix_records
    energy_lookup = _energy_lookup(energy_matrix)
    merged: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for row in base_records:
        job_id = str(_get(row, "job_id", "")).strip()
        op_index = _operation_index(_get(row, "op_id", _get(row, "op_index", 1)))
        machine_id = str(_get(row, "machine_id", "")).strip()
        if job_id and machine_id:
            merged[(job_id, op_index, machine_id)] = dict(row)
    for row in matrix_records:
        key = (str(row["job_id"]), int(row["op_index"]), str(row["machine_id"]))
        target = dict(merged.get(key, {}))
        if target:
            # 外部加工时间矩阵只覆盖时间；主表中的换型、能耗和其他属性继续保留。
            target["processing_time"] = row.get("processing_time", target.get("processing_time", 0))
        else:
            target.update(row)
        if key in energy_lookup:
            target["total_energy"] = energy_lookup[key]
        merged[key] = target
    return list(merged.values())


def _apply_energy_matrix(base_records: Sequence[Mapping[str, Any]], energy_matrix: Any) -> List[Dict[str, Any]]:
    """仅覆盖已有长表的能耗字段。"""

    lookup = _energy_lookup(energy_matrix)
    result: List[Dict[str, Any]] = []
    for row in base_records:
        item = dict(row)
        job_id = str(_get(row, "job_id", "")).strip()
        op_index = _operation_index(_get(row, "op_id", _get(row, "op_index", 1)))
        machine_id = str(_get(row, "machine_id", "")).strip()
        if (job_id, op_index, machine_id) in lookup:
            item["total_energy"] = lookup[(job_id, op_index, machine_id)]
        result.append(item)
    return result


def _matrix_records(matrix: Any, energy_matrix: Optional[Any] = None) -> List[Dict[str, Any]]:
    """把常见矩阵结构展开为长表记录。

    支持：
    ``{job: {op: {machine: value}}}``、``{job-op: {machine: value}}``，
    以及带 ``job_id``/``op_index`` 的 CSV 宽表。字典中的机器键可以是任意字符串。
    """

    records: List[Dict[str, Any]] = []

    energy_lookup = _energy_lookup(energy_matrix) if energy_matrix is not None else {}

    def energy_for(job_id: str, op_index: int, machine_id: str) -> float:
        return energy_lookup.get((str(job_id), int(op_index), str(machine_id)), 0.0)

    if isinstance(matrix, list):
        for row in matrix:
            if isinstance(row, dict):
                job_id = str(row.get("job_id", row.get("job", row.get("工件", ""))))
                op_raw = row.get("op_index", row.get("operation", row.get("工序", "1")))
                op_index = _operation_index(op_raw)
                # 长表矩阵：每行已经明确给出机器编号和数值。
                explicit_machine = row.get("machine_id", row.get("machine", row.get("机器", "")))
                explicit_time = row.get("processing_time", row.get("time", row.get("加工时间", None)))
                explicit_energy = row.get("total_energy", row.get("energy", row.get("能耗", None)))
                explicit_value = explicit_time if explicit_time is not None and str(explicit_time).strip() != "" else explicit_energy
                if str(explicit_machine).strip() and explicit_value is not None and str(explicit_value).strip() != "":
                    records.append({
                        "job_id": job_id,
                        "op_index": op_index,
                        "machine_id": str(explicit_machine).strip(),
                        "processing_time": _float(explicit_value),
                        "total_energy": energy_for(job_id, op_index, str(explicit_machine).strip()),
                    })
                    continue
                for machine_id, value in row.items():
                    if str(machine_id).lower() in {
                        "job_id", "job", "工件", "op_index", "operation", "工序", "machine_id", "machine", "机器",
                        "processing_time", "time", "加工时间", "energy", "total_energy", "能耗",
                    }:
                        continue
                    if _float(value, 0.0) <= 0:
                        continue
                    records.append({
                        "job_id": job_id,
                        "op_index": op_index,
                        "machine_id": machine_id,
                        "processing_time": _float(value),
                        "total_energy": energy_for(job_id, op_index, str(machine_id)),
                    })
    elif isinstance(matrix, dict):
        for job_key, job_value in matrix.items():
            if not isinstance(job_value, dict):
                continue
            # ``job-o`` 直接指向机器字典，例如 {"J1-O1": {"Laser-A": 20}}。
            # 不能通过机器编号前缀判断结构，否则会拒绝任意命名的机器。
            scalar_values = all(not isinstance(value, (dict, list, tuple)) for value in job_value.values())
            if scalar_values:
                job_id, op_index = job_key.split("-O", 1) if "-O" in str(job_key) else (job_key, 1)
                for machine_id, value in job_value.items():
                    if _float(value, 0.0) > 0:
                        records.append({"job_id": job_id, "op_index": _operation_index(op_index), "machine_id": machine_id, "processing_time": _float(value), "total_energy": energy_for(job_id, _operation_index(op_index), machine_id)})
            else:
                for op_key, machine_values in job_value.items():
                    if not isinstance(machine_values, dict):
                        continue
                    op_index = _operation_index(op_key)
                    for machine_id, value in machine_values.items():
                        if _float(value, 0.0) > 0:
                            records.append({"job_id": job_key, "op_index": op_index, "machine_id": machine_id, "processing_time": _float(value), "total_energy": energy_for(job_key, op_index, machine_id)})
    return records


def _read_matrix_file(path: Optional[Path]) -> Optional[Any]:
    if path is None:
        return None
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


def _json_to_problem(path: Path, processing_matrix_path: Optional[Path], energy_matrix_path: Optional[Path]) -> ProblemData:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = _normalise_records(payload.get("jobs", []), "job_id")
    machines = _normalise_records(payload.get("machines", []), "machine_id")
    operation_records: List[Dict[str, Any]] = []
    if jobs and isinstance(jobs[0], dict) and any("operations" in job for job in jobs):
        for job in jobs:
            for operation in job.get("operations", []):
                candidates = operation.get("candidates", operation.get("options", []))
                if isinstance(candidates, dict):
                    candidates = [
                        {"machine_id": machine_id, "processing_time": value}
                        for machine_id, value in candidates.items()
                    ]
                for candidate in candidates:
                    record = dict(candidate)
                    record.update({
                        "job_id": job.get("job_id", job.get("id")),
                        "job_name": job.get("name", ""),
                        "family": job.get("family", "未知"),
                        "op_index": operation.get("op_index", operation.get("index", operation.get("id", 1))),
                        "release_time": job.get("release_time", 0),
                        "due_date": job.get("due_date", 0),
                        "priority": job.get("priority", "C"),
                        "tardiness_weight": job.get("tardiness_weight", 1),
                    })
                    operation_records.append(record)
        job_records = jobs
    else:
        operation_records = payload.get("operations", payload.get("operation_records", []))
        expanded_records: List[Dict[str, Any]] = []
        for operation in operation_records:
            if not isinstance(operation, dict):
                continue
            candidates = operation.get("candidates", operation.get("options"))
            if candidates is None:
                expanded_records.append(operation)
                continue
            if isinstance(candidates, dict):
                candidates = [{"machine_id": machine_id, "processing_time": value} for machine_id, value in candidates.items()]
            for candidate in candidates:
                record = dict(operation)
                record.pop("candidates", None)
                record.pop("options", None)
                record.update(candidate)
                expanded_records.append(record)
        operation_records = expanded_records
        job_records = payload.get("job_records", jobs)
    if not operation_records and payload.get("processing_time_matrix") is not None:
        operation_records = _matrix_records(payload["processing_time_matrix"], payload.get("energy_matrix"))
    if processing_matrix_path is not None:
        operation_records = _merge_matrix_records(
            operation_records,
            _read_matrix_file(processing_matrix_path),
            _read_matrix_file(energy_matrix_path),
        )
    elif energy_matrix_path is not None:
        operation_records = _apply_energy_matrix(
            operation_records,
            _read_matrix_file(energy_matrix_path),
        )
    return records_to_problem(operation_records, job_records, machines, str(path))


def _tabular_to_problem(path: Path, processing_matrix_path: Optional[Path], energy_matrix_path: Optional[Path]) -> ProblemData:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        operation_frame = next((frame for name, frame in sheets.items() if str(name).lower() in {"operations", "operation", "工序", "加工数据"}), None)
        if operation_frame is None:
            operation_frame = next(iter(sheets.values()))
        job_frame = next((frame for name, frame in sheets.items() if str(name).lower() in {"jobs", "job", "工件", "订单"}), None)
        machine_frame = next((frame for name, frame in sheets.items() if str(name).lower() in {"machines", "machine", "设备", "机器"}), None)
    else:
        operation_frame = pd.read_csv(path)
        job_frame = None
        machine_frame = None
    records = operation_frame.fillna("").to_dict(orient="records")
    if processing_matrix_path is not None:
        records = _merge_matrix_records(
            records,
            _read_matrix_file(processing_matrix_path),
            _read_matrix_file(energy_matrix_path),
        )
    elif energy_matrix_path is not None:
        records = _apply_energy_matrix(
            records,
            _read_matrix_file(energy_matrix_path),
        )
    return records_to_problem(
        records,
        None if job_frame is None else job_frame.fillna("").to_dict(orient="records"),
        None if machine_frame is None else machine_frame.fillna("").to_dict(orient="records"),
        str(path),
    )


def _overlay_problem_matrices(
    problem: ProblemData,
    processing_matrix_path: Optional[Path],
    energy_matrix_path: Optional[Path],
) -> ProblemData:
    """在已解析的 Markdown 数据上覆盖外部时间/能耗矩阵。"""

    processing = _read_matrix_file(processing_matrix_path)
    energy = _read_matrix_file(energy_matrix_path)
    if processing is None and energy is None:
        return problem
    processing_records = _matrix_records(processing, energy) if processing is not None else []
    energy_lookup = _energy_lookup(energy) if energy is not None else {}

    def update_option(job_id: str, op_index: int, machine_id: str, time_value: Optional[float], energy_value: Optional[float]) -> bool:
        operation = problem.operations.get("{}-O{}".format(job_id, op_index))
        if operation is None:
            return False
        option = next((item for item in operation.candidates if item.machine_id == machine_id), None)
        if option is None:
            return False
        if time_value is not None and time_value > 0:
            option.processing_time = time_value
        if energy_value is not None:
            option.total_energy = energy_value
        return True

    for row in processing_records:
        job_id = str(row.get("job_id", ""))
        op_index = int(row.get("op_index", 1))
        machine_id = str(row.get("machine_id", ""))
        value = _float(row.get("processing_time", 0))
        energy_value = energy_lookup.get((job_id, op_index, machine_id))
        # 结构化矩阵优先按明确的工序覆盖；纯编号矩阵则尝试按机器资格唯一映射。
        if update_option(job_id, op_index, machine_id, value, energy_value):
            continue
        matches = [
            operation for operation in problem.jobs.get(job_id, JobSpec(job_id, job_id, "未知", 0, 0, "C", 1)).operations
            if any(option.machine_id == machine_id for option in operation.candidates)
        ]
        if len(matches) == 1:
            update_option(job_id, matches[0].op_index, machine_id, value, energy_value)

    if processing is None and energy is not None:
        for (job_id, op_index, machine_id), value in energy_lookup.items():
            if not update_option(job_id, op_index, machine_id, None, value):
                matches = [
                    operation for operation in problem.jobs.get(job_id, JobSpec(job_id, job_id, "未知", 0, 0, "C", 1)).operations
                    if any(option.machine_id == machine_id for option in operation.candidates)
                ]
                if len(matches) == 1:
                    update_option(job_id, matches[0].op_index, machine_id, None, value)
    problem.source_notes = (problem.source_notes + "; 外部矩阵已覆盖" if problem.source_notes else "外部矩阵已覆盖")
    return problem


def load_any_dataset(
    path: Path,
    processing_matrix_path: Optional[Path] = None,
    energy_matrix_path: Optional[Path] = None,
) -> ProblemData:
    """按扩展名自动选择输入解析器。"""

    suffix = path.suffix.lower()
    if suffix == ".md":
        return _overlay_problem_matrices(load_markdown_dataset(path), processing_matrix_path, energy_matrix_path)
    if suffix == ".json":
        return _json_to_problem(path, processing_matrix_path, energy_matrix_path)
    if suffix in {".csv", ".xlsx", ".xls"}:
        return _tabular_to_problem(path, processing_matrix_path, energy_matrix_path)
    raise ValueError("不支持的数据格式: {}".format(path.suffix))
