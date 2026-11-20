"""论文常用结果图导出。"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import pandas as pd

# 优先使用本地中文字体，保证图例和中文工件族名称在导出 PNG 中不丢字。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from .data_model import ProblemData, Schedule
from .objectives import summarize_schedule


def _job_colors(data: ProblemData) -> Dict[str, object]:
    families = sorted(set(job.family for job in data.jobs.values()))
    cmap = plt.get_cmap("tab20")
    return {family: cmap(index % 20) for index, family in enumerate(families)}


def export_schedule_csv(data: ProblemData, schedule: Schedule, path: Path) -> None:
    rows = []
    for entry in sorted(schedule.entries.values(), key=lambda item: (item.machine_id, item.setup_start, item.operation_id)):
        rows.append({
            "operation_id": entry.operation_id,
            "job_id": entry.job_id,
            "operation_index": entry.op_index,
            "machine_id": entry.machine_id,
            "setup_start": entry.setup_start,
            "start": entry.start,
            "end": entry.end,
            "processing_time": entry.processing_time,
            "setup_time": entry.setup_time,
            "energy": entry.energy,
            "frozen": entry.frozen,
        })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def save_gantt(data: ProblemData, schedule: Schedule, path: Path, title: str = "Flexible Job Shop Schedule") -> None:
    """导出适合论文排版的高分辨率甘特图，不绘制网格线。"""

    machines = data.machine_ids or sorted({entry.machine_id for entry in schedule.entries.values()})
    colors = _job_colors(data)
    height = max(5.6, 0.48 * len(machines) + 1.8)
    fig, ax = plt.subplots(figsize=(15.2, height), facecolor="white")
    ax.set_facecolor("#fbfcfc")
    machine_y = {machine_id: index for index, machine_id in enumerate(machines)}
    # 轻微交替底色帮助追踪机器行，但不形成网格线。
    for index in range(len(machines)):
        if index % 2 == 0:
            ax.axhspan(index - 0.46, index + 0.46, color="#f4f7f7", zorder=0)
    entries = sorted(
        schedule.entries.values(),
        key=lambda item: (machine_y.get(item.machine_id, 0), item.start, item.operation_id),
    )
    for entry in entries:
        y = machine_y.get(entry.machine_id, 0)
        family = data.jobs[entry.job_id].family if entry.job_id in data.jobs else "unknown"
        color = colors.get(family, "#4c78a8")
        if entry.setup_time > 1e-9:
            ax.barh(
                y,
                entry.setup_time,
                left=entry.setup_start,
                height=0.66,
                color="#dce5e7",
                edgecolor="#8ea5aa",
                linewidth=0.45,
                hatch="//",
                zorder=2,
            )
        ax.barh(
            y,
            max(0.01, entry.end - entry.start),
            left=entry.start,
            height=0.66,
            color=color,
            alpha=0.58 if entry.frozen else 0.96,
            edgecolor="#20343d",
            linewidth=0.65,
            zorder=3,
        )
        if entry.end - entry.start >= max(8.0, schedule.makespan * 0.018):
            ax.text(
                (entry.start + entry.end) / 2.0,
                y,
                "{} · O{}".format(entry.job_id, entry.op_index),
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="600",
                color="white",
                clip_on=True,
                zorder=4,
            )
    ax.set_yticks(list(range(len(machines))))
    ax.set_yticklabels(machines, fontsize=9)
    ax.set_xlabel("Time (min)", labelpad=8)
    ax.set_ylabel("Machine", labelpad=8)
    ax.set_title(title, loc="left", fontsize=14, fontweight="700", color="#172936", pad=16)
    ax.text(
        0,
        1.012,
        "Processing blocks by machine · setup stages shown with hatch",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#71838a",
    )
    ax.invert_yaxis()
    ax.set_xlim(left=0, right=max(1.0, schedule.makespan) * 1.03)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis="x", colors="#6b7d87", labelsize=8.5, length=3)
    ax.tick_params(axis="y", colors="#425a64", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#c7d4d7")
    ax.spines["bottom"].set_color("#c7d4d7")
    families = sorted(set(job.family for job in data.jobs.values()))
    handles = [Patch(facecolor=colors[family], edgecolor="#20343d", label=family) for family in families if family in colors]
    handles.append(Patch(facecolor="#dce5e7", edgecolor="#8ea5aa", hatch="//", label="Setup / changeover"))
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=min(5, len(handles)), frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.18)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_convergence(history: Sequence[Dict[str, object]], path: Path, title: str = "Adaptive elite NSGA-II-VNS convergence") -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    objective_mode = str(frame["objective_mode"].iloc[0]) if "objective_mode" in frame.columns else "multi"
    left_field = "hypervolume" if objective_mode == "multi" else "best_score"
    axes[0].plot(frame["generation"].to_numpy(), frame[left_field].to_numpy(), color="#2f6f9f", linewidth=2.2)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Hypervolume" if objective_mode == "multi" else "Best objective score")
    axes[0].set_title("Pareto quality" if objective_mode == "multi" else "Single-objective convergence")
    axes[1].plot(frame["generation"].to_numpy(), frame["best_makespan"].to_numpy(), label="Cmax", linewidth=2)
    axes[1].plot(frame["generation"].to_numpy(), frame["best_tardiness"].to_numpy(), label="Weighted tardiness", linewidth=2)
    if "best_energy" in frame.columns:
        axes[1].plot(frame["generation"].to_numpy(), frame["best_energy"].to_numpy(), label="TEC", linewidth=2)
    if "best_disturbance" in frame.columns:
        axes[1].plot(frame["generation"].to_numpy(), frame["best_disturbance"].to_numpy(), label="Disturbance", linewidth=2)
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Objective value")
    axes[1].legend()
    axes[1].set_title("Best objective values")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_pareto(front: Sequence[object], path: Path, title: str = "Pareto Front") -> None:
    if not front:
        return
    values = [item.objectives for item in front]
    if len(values[0]) < 3:
        return
    frame = pd.DataFrame(values, columns=["Cmax", "Weighted tardiness", "TEC"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter = ax.scatter(frame.iloc[:, 0].to_numpy(), frame.iloc[:, 1].to_numpy(), c=frame.iloc[:, 2].to_numpy(), cmap="viridis", s=50, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Cmax")
    ax.set_ylabel("Weighted tardiness")
    ax.set_title(title)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(frame.columns[2])
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_machine_load(data: ProblemData, schedule: Schedule, path: Path, title: str = "Machine Load") -> None:
    loads = []
    horizon = max(1.0, schedule.makespan)
    for machine_id in data.machine_ids:
        busy = sum(entry.end - entry.start for entry in schedule.entries_for_machine(machine_id))
        loads.append({"machine_id": machine_id, "busy_time": busy, "load_ratio": busy / horizon})
    frame = pd.DataFrame(loads)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(frame["machine_id"].to_numpy(), frame["load_ratio"].to_numpy(), color="#4c78a8")
    ax.set_ylim(0, max(1.0, frame["load_ratio"].max() * 1.15))
    ax.set_ylabel("Load ratio")
    ax.set_xlabel("Machine")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_baseline_comparison(frame: pd.DataFrame, path: Path, title: str = "Algorithm comparison") -> None:
    """从 baseline_comparison.csv 生成多指标对比图。"""

    if frame.empty:
        return
    # 每个方法可能对应多个 Pareto 解，绘图时用三项目标的最小值作为摘要。
    summary = frame.groupby("method", as_index=False)[["makespan", "weighted_tardiness", "energy"]].min()
    labels = (
        frame.drop_duplicates("method").set_index("method")["method_label"].to_dict()
        if "method_label" in frame.columns
        else {method: method for method in summary["method"]}
    )
    x = list(range(len(summary)))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fields = [("makespan", "Cmax"), ("weighted_tardiness", "Weighted tardiness"), ("energy", "Energy")]
    for axis, (field, label) in zip(axes, fields):
        axis.bar([value - width for value in x], summary[field].to_numpy(), width=width, label=label, color="#2f6f9f")
        axis.set_xticks(x)
        axis.set_xticklabels([labels.get(method, method) for method in summary["method"]], rotation=35, ha="right")
        axis.set_title(label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_algorithm_convergence(frame: pd.DataFrame, path: Path, title: str = "Algorithm convergence comparison") -> None:
    """导出多算法收敛曲线；不使用网格线以保持论文图面干净。"""

    if frame.empty or "iteration" not in frame.columns:
        return
    fig, ax = plt.subplots(figsize=(11.5, 5.8), facecolor="white")
    ax.set_facecolor("#fbfcfc")
    palette = list(plt.get_cmap("tab20").colors)
    labels = {}
    for index, (method, group) in enumerate(frame.groupby("method", sort=False)):
        group = group.sort_values("iteration")
        label = group["method_label"].iloc[0] if "method_label" in group.columns else method
        labels[method] = label
        ax.plot(
            group["iteration"].to_numpy(),
            group["score"].to_numpy(),
            linewidth=2.0,
            color=palette[index % len(palette)],
            marker="o" if len(group) <= 12 else None,
            markersize=3.5,
            label=label,
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized composite objective")
    ax.set_title(title, loc="left", fontsize=14, fontweight="700", color="#172936", pad=14)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    ax.tick_params(colors="#6b7d87", labelsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#c7d4d7")
    ax.spines["bottom"].set_color("#c7d4d7")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_dynamic_summary(frame: pd.DataFrame, path: Path, title: str = "Dynamic rescheduling comparison") -> None:
    """导出动态事件下的四目标对比图。"""

    if frame.empty:
        return
    frame = frame.copy()
    x = list(range(len(frame)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.0))
    fields = [("makespan", "Cmax"), ("weighted_tardiness", "Weighted tardiness"), ("total_energy", "TEC"), ("disturbance", "Rescheduling disturbance")]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for axis, (field, label), color in zip(axes.flat, fields, colors):
        if field not in frame.columns:
            continue
        axis.bar(x, frame[field].to_numpy(), color=color)
        axis.set_xticks(x)
        axis.set_xticklabels(frame["scenario"].tolist(), rotation=45, ha="right")
        axis.set_title(label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_job_tardiness(data: ProblemData, schedule: Schedule, path: Path, title: str = "Job tardiness") -> None:
    """导出每个工件的延期柱状图。"""

    completion = {}
    for entry in schedule.entries.values():
        completion[entry.job_id] = max(completion.get(entry.job_id, 0.0), entry.end)
    jobs = list(data.job_ids)
    tardiness = [max(0.0, completion.get(job_id, 0.0) - data.jobs[job_id].due_date) for job_id in jobs]
    weights = [data.jobs[job_id].tardiness_weight for job_id in jobs]
    fig, axis = plt.subplots(figsize=(13, 4.8))
    bars = axis.bar(jobs, tardiness, color=["#d95f02" if weight > 1.0 else "#7570b3" for weight in weights])
    axis.set_xlabel("Job")
    axis.set_ylabel("Tardiness (min)")
    axis.set_title(title)
    for bar, value in zip(bars, tardiness):
        if value > 0:
            axis.text(bar.get_x() + bar.get_width() / 2.0, value, "{:.0f}".format(value), ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_metrics(data: ProblemData, schedule: Schedule, path: Path, baseline: Optional[Schedule] = None, extra: Optional[Dict[str, object]] = None) -> None:
    payload: Dict[str, object] = summarize_schedule(data, schedule, baseline)
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
