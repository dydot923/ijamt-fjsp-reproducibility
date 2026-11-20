"""论文实验编排和结果导出。"""

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .config import AlgorithmConfig, DynamicConfig, ExperimentConfig
from .data_model import ProblemData, Schedule
from .dataset_io import load_any_dataset
from .dynamic import EventDrivenRescheduler, build_failure_event, build_insertion_event
from .baselines import (
    ALGORITHM_LABELS,
    DEFAULT_COMPARISON_ALGORITHMS,
    run_algorithm_with_history,
)
from .nsga_vns import IGA_NSGA_VNS, Individual
from .objectives import summarize_schedule
from .metrics import response_metrics
from .plotting import (
    export_schedule_csv,
    save_baseline_comparison,
    save_algorithm_convergence,
    save_convergence,
    save_dynamic_summary,
    save_gantt,
    save_job_tardiness,
    save_machine_load,
    save_metrics,
    save_pareto,
)


def _select_knee(front: Sequence[Individual]) -> Individual:
    """用归一化目标和选择动态场景的代表性膝点解。"""

    if not front:
        raise ValueError("Pareto 前沿为空")
    dimensions = list(zip(*[item.objectives for item in front]))
    mins = [min(values) for values in dimensions]
    maxs = [max(values) for values in dimensions]

    def score(item: Individual) -> float:
        return sum(
            (value - low) / (high - low) if high - low > 1e-12 else 0.0
            for value, low, high in zip(item.objectives, mins, maxs)
        )

    return min(front, key=score)


def _write_history(history: Sequence[Dict[str, float]], path: Path) -> None:
    pd.DataFrame(history).to_csv(path, index=False, encoding="utf-8-sig")


def run_static(
    data: ProblemData,
    algorithm_config: AlgorithmConfig,
    output_dir: Path,
    make_plots: bool = True,
) -> Tuple[Schedule, List[Individual], IGA_NSGA_VNS]:
    output_dir.mkdir(parents=True, exist_ok=True)
    solver = IGA_NSGA_VNS(data, algorithm_config)
    front = solver.run()
    knee = _select_knee(front)
    schedule = knee.schedule
    if schedule is None:
        raise RuntimeError("静态算法没有生成代表性调度方案")
    _write_history(solver.history, output_dir / "static_convergence.csv")
    pd.DataFrame(
        [
            {"solution_index": index, "makespan": item.objectives[0], "weighted_tardiness": item.objectives[1], "energy": item.objectives[2]}
            for index, item in enumerate(front)
        ]
    ).to_csv(output_dir / "static_pareto.csv", index=False, encoding="utf-8-sig")
    export_schedule_csv(data, schedule, output_dir / "static_schedule.csv")
    save_metrics(
        data,
        schedule,
        output_dir / "static_metrics.json",
        extra={
            "front_size": len(front),
            "method": "Adaptive elite NSGA-II-VNS",
            "objective_mode": algorithm_config.objective_mode,
            "objective_weights": list(algorithm_config.objective_weights),
        },
    )
    if make_plots:
        save_gantt(data, schedule, output_dir / "static_gantt.png", "Static multi-objective FJSP schedule")
        save_convergence(solver.history, output_dir / "static_convergence.png")
        save_pareto(front, output_dir / "static_pareto.png")
        save_machine_load(data, schedule, output_dir / "static_machine_load.png")
        save_job_tardiness(data, schedule, output_dir / "static_job_tardiness.png")
    return schedule, front, solver


def run_baseline_comparison(
    data: ProblemData,
    algorithm_config: AlgorithmConfig,
    output_dir: Path,
    selected_algorithms: Optional[Sequence[str]] = None,
) -> None:
    """运行用户勾选的经典算法，并导出统一指标表和对比图。"""

    rows = []
    convergence_rows = []
    selected = list(selected_algorithms or DEFAULT_COMPARISON_ALGORITHMS)
    for algorithm_name in selected:
        front, history = run_algorithm_with_history(algorithm_name, data, algorithm_config)
        for item in front:
            rows.append({
                "method": algorithm_name,
                "method_label": ALGORITHM_LABELS.get(algorithm_name, algorithm_name),
                "makespan": item.objectives[0],
                "weighted_tardiness": item.objectives[1],
                "energy": item.objectives[2],
                "front_size": len(front),
            })
        for point in history:
            convergence_rows.append({
                "method": algorithm_name,
                "method_label": ALGORITHM_LABELS.get(algorithm_name, algorithm_name),
                **point,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "baseline_comparison.csv", index=False, encoding="utf-8-sig")
    save_baseline_comparison(frame, output_dir / "baseline_comparison.png")
    convergence = pd.DataFrame(convergence_rows)
    convergence.to_csv(output_dir / "algorithm_convergence.csv", index=False, encoding="utf-8-sig")
    save_algorithm_convergence(convergence, output_dir / "algorithm_convergence.png")


def _save_dynamic_result(data: ProblemData, baseline: Schedule, result: object, output_dir: Path, label: str, make_plots: bool) -> None:
    result_dir = output_dir / label
    result_dir.mkdir(parents=True, exist_ok=True)
    schedule = result.schedule
    export_schedule_csv(data, schedule, result_dir / "schedule.csv")
    save_metrics(
        data,
        schedule,
        result_dir / "metrics.json",
        baseline=baseline,
        extra={
            "method": "event-driven local-complete rescheduling",
            "event_type": result.event_type,
            "event_time": result.event_time,
            "strategy": result.strategy,
            "affected_operation_count": len(result.affected_operation_ids),
            "frozen_operation_count": len(result.frozen_operation_ids),
            **result.metadata,
        },
    )
    if make_plots:
        save_gantt(data, schedule, result_dir / "gantt.png", "{}: {} rescheduling".format(result.event_type, result.strategy))
        save_machine_load(data, schedule, result_dir / "machine_load.png", "{} machine load".format(label))
        if result.convergence_history:
            _write_history(result.convergence_history, result_dir / "convergence.csv")
            save_convergence(result.convergence_history, result_dir / "convergence.png", "{} convergence".format(label))
    elif result.convergence_history:
        _write_history(result.convergence_history, result_dir / "convergence.csv")


def run_dynamic(
    source_data: ProblemData,
    baseline: Schedule,
    algorithm_config: AlgorithmConfig,
    dynamic_config: DynamicConfig,
    output_dir: Path,
    make_plots: bool = True,
    all_scenarios: bool = False,
) -> List[Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = dynamic_config.failure_time_ratios if all_scenarios else [dynamic_config.representative_failure_ratio]
    durations = dynamic_config.failure_durations if all_scenarios else [dynamic_config.representative_failure_duration]
    insertion_ratios = dynamic_config.insertion_time_ratios if all_scenarios else [dynamic_config.representative_insertion_ratio]
    reports: List[Dict[str, object]] = []

    for ratio in ratios:
        for duration in durations:
            data = copy.deepcopy(source_data)
            event = build_failure_event(data, baseline, ratio, duration, dynamic_config.failure_machine_id)
            result = EventDrivenRescheduler(data, algorithm_config, dynamic_config).reschedule(baseline.copy(), failure=event)
            label = "failure_r{:.2f}_d{}".format(ratio, int(duration))
            _save_dynamic_result(data, baseline, result, output_dir, label, make_plots)
            reports.append({"scenario": label, "strategy": result.strategy, "event_type": result.event_type, **response_metrics(data, baseline, result.schedule)})

    for ratio in insertion_ratios:
        data = copy.deepcopy(source_data)
        event = build_insertion_event(
            data,
            baseline,
            ratio,
            "J21",
            dynamic_config.inserted_due_window,
            dynamic_config.inserted_priority,
            dynamic_config.inserted_weight,
        )
        result = EventDrivenRescheduler(data, algorithm_config, dynamic_config).reschedule(baseline.copy(), insertion=event)
        label = "insertion_r{:.2f}".format(ratio)
        _save_dynamic_result(data, baseline, result, output_dir, label, make_plots)
        reports.append({"scenario": label, "strategy": result.strategy, "event_type": result.event_type, **response_metrics(data, baseline, result.schedule)})

    if not all_scenarios:
        data = copy.deepcopy(source_data)
        failure = build_failure_event(
            data,
            baseline,
            dynamic_config.representative_failure_ratio,
            dynamic_config.representative_failure_duration,
            dynamic_config.failure_machine_id,
        )
        insertion = build_insertion_event(
            data,
            baseline,
            dynamic_config.representative_insertion_ratio,
            "J21",
            dynamic_config.inserted_due_window,
            dynamic_config.inserted_priority,
            dynamic_config.inserted_weight,
        )
        result = EventDrivenRescheduler(data, algorithm_config, dynamic_config).reschedule(baseline.copy(), failure=failure, insertion=insertion)
        label = "combined_r{:.2f}".format(dynamic_config.representative_failure_ratio)
        _save_dynamic_result(data, baseline, result, output_dir, label, make_plots)
        reports.append({"scenario": label, "strategy": result.strategy, "event_type": result.event_type, **response_metrics(data, baseline, result.schedule)})

    frame = pd.DataFrame(reports)
    frame.to_csv(output_dir / "dynamic_summary.csv", index=False, encoding="utf-8-sig")
    if make_plots:
        save_dynamic_summary(frame, output_dir / "dynamic_summary.png")
    (output_dir / "dynamic_summary.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return reports


def run_experiment(config: ExperimentConfig, mode: str = "all", all_scenarios: bool = False) -> Dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_any_dataset(config.data_path, config.processing_matrix_path, config.energy_matrix_path)
    report: Dict[str, object] = {
        "data_path": str(config.data_path),
        "source_notes": data.source_notes,
        "jobs": len(data.jobs),
        "operations": len(data.operations),
        "machines": len(data.machines),
        "machine_ids": data.machine_ids,
        "comparison_algorithms": list(config.comparison_algorithms) if config.run_baselines else [],
        "objective_mode": config.algorithm.objective_mode,
        "objective_weights": list(config.algorithm.objective_weights),
    }
    (config.output_dir / "experiment_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (config.output_dir / "dataset_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if mode in {"static", "all", "dynamic"}:
        static_dir = config.output_dir / "static"
        baseline, front, _ = run_static(data, config.algorithm, static_dir, config.make_plots)
        report["static"] = summarize_schedule(data, baseline)
        if config.run_baselines:
            run_baseline_comparison(data, config.algorithm, static_dir, config.comparison_algorithms)
        if mode in {"dynamic", "all"}:
            report["dynamic"] = run_dynamic(data, baseline, config.algorithm, config.dynamic, config.output_dir / "dynamic", config.make_plots, all_scenarios)
    return report
