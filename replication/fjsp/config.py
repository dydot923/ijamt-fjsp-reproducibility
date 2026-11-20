"""全局配置。

配置集中放置，便于论文实验统一随机种子、运行次数和计算预算。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class AlgorithmConfig:
    """IGA-NSGA-II-VNS 参数。"""

    population_size: int = 40
    generations: int = 60
    crossover_rate: float = 0.90
    mutation_rate: float = 0.20
    machine_mutation_rate: float = 0.20
    vns_probability: float = 0.35
    elite_local_search_ratio: float = 0.20
    stagnation_limit: int = 10
    tournament_size: int = 2
    seed: int = 42
    archive_size: int = 100
    objective_mode: str = "multi"
    objective_weights: Tuple[float, ...] = (1.0, 1.0, 1.0)
    restart_ratio: float = 0.15
    local_search_trials: int = 6
    # ``hybrid`` 使用规则种子与随机扰动；``random`` 用于可复现的初始化消融。
    initialization_mode: str = "hybrid"
    verbose: bool = True
    # None 保留原代数协议；新实验的变异进展使用已耗解码预算。
    evaluation_budget: Optional[int] = None


@dataclass
class DynamicConfig:
    """动态事件和重调度参数。"""

    local_disturbance_threshold: float = 0.25
    local_tardiness_increase_threshold: float = 0.35
    failure_time_ratios: List[float] = field(default_factory=lambda: [0.25, 0.50, 0.75])
    failure_durations: List[float] = field(default_factory=lambda: [30.0, 60.0, 120.0])
    insertion_time_ratios: List[float] = field(default_factory=lambda: [0.25, 0.50, 0.75])
    inserted_due_window: float = 180.0
    inserted_priority: str = "A"
    inserted_weight: float = 1.50
    # 代表性场景参数；为空时仍可使用 failure_time_ratios 等批量实验参数。
    representative_failure_ratio: float = 0.50
    representative_failure_duration: float = 60.0
    representative_insertion_ratio: float = 0.50
    failure_machine_id: Optional[str] = None


@dataclass
class ExperimentConfig:
    """实验目录和绘图设置。"""

    data_path: Path
    output_dir: Path
    processing_matrix_path: Optional[Path] = None
    energy_matrix_path: Optional[Path] = None
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    dynamic: DynamicConfig = field(default_factory=DynamicConfig)
    make_plots: bool = True
    save_population: bool = False
    run_baselines: bool = False
    comparison_algorithms: List[str] = field(default_factory=lambda: [
        "standard_nsga_ii",
        "simplified_moead_alns",
        "simulated_annealing",
        "tabu_search",
        "pso",
        "aco",
    ])

    @classmethod
    def from_workspace(cls, workspace: Path, output_dir: Optional[Path] = None) -> "ExperimentConfig":
        if output_dir is None:
            output_dir = workspace / "code" / "results"
        return cls(
            data_path=workspace / "data" / "第3组_柔性作业车间_20x10.md",
            output_dir=output_dir,
        )
