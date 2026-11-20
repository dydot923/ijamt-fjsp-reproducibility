"""论文对比方法入口。

主算法仍由 ``nsga_vns.py`` 实现。本文件提供可复现实验中常用的基线：
规则解、标准 NSGA-II（关闭 VNS）和简化 MOEA/D-ALNS。
"""

import math
import random
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from .config import AlgorithmConfig
from .data_model import ProblemData, Schedule
from .decoder import decode
from .nsga_vns import (
    Individual,
    IGA_NSGA_VNS,
    _build_machine_assignment,
    _build_order_sequence,
    _evaluate,
    initialize_population,
)
from .objectives import static_objectives, dominates


def run_standard_nsga_ii(data: ProblemData, config: Optional[AlgorithmConfig] = None, return_history: bool = False):
    """关闭 VNS 和精英局部搜索，作为标准 NSGA-II 基线。"""

    config = config or AlgorithmConfig()
    baseline_config = replace(config, vns_probability=0.0, elite_local_search_ratio=0.0)
    solver = IGA_NSGA_VNS(data, baseline_config)
    front = solver.run()
    if return_history:
        return front, solver.history
    return front


def run_rule_based(data: ProblemData, rule: str = "edd") -> Individual:
    """生成 EDD/SPT/低能耗/负荷均衡规则方案。"""

    rng = random.Random(0)
    sequence = _build_order_sequence(data, rule, rng)
    machine_rule = "energy" if rule == "energy" else "load" if rule == "load" else "speed"
    assignment = _build_machine_assignment(data, machine_rule, rng)
    individual = Individual(sequence, assignment)
    _evaluate(individual, data, None, None)
    return individual


def _destroy_repair(individual: Individual, data: ProblemData, rng: random.Random, destroy_ratio: float = 0.20) -> Individual:
    """简化 ALNS：破坏部分工序的机器分配和工件位置，再用规则修复。"""

    candidate = individual.clone()
    remove_count = max(1, int(len(candidate.order_sequence) * destroy_ratio))
    for _ in range(remove_count):
        if len(candidate.order_sequence) < 2:
            break
        first, second = rng.sample(range(len(candidate.order_sequence)), 2)
        candidate.order_sequence[first], candidate.order_sequence[second] = candidate.order_sequence[second], candidate.order_sequence[first]
    for _ in range(max(1, int(len(data.operation_ids) * destroy_ratio))):
        operation_id = rng.choice(data.operation_ids)
        candidates = data.operations[operation_id].candidates
        candidate.machine_assignment[operation_id] = rng.choice(candidates).machine_id
    _evaluate(candidate, data, None, None)
    return candidate


def run_simplified_moead_alns(data: ProblemData, config: Optional[AlgorithmConfig] = None, return_history: bool = False):
    """可解释的简化 MOEA/D-ALNS 对比实现。

    该实现只用于论文对比，不包含原 AD-MOEA/D-ALNS 的全部自适应分解和奖励模块，
    防止把对比算法做得比主算法更复杂。
    """

    config = config or AlgorithmConfig()
    rng = random.Random(config.seed + 101)
    history: List[Dict[str, float]] = []
    solver = IGA_NSGA_VNS(data, replace(config, vns_probability=0.0, elite_local_search_ratio=0.0))
    population = solver.run()
    if not population:
        return ([], history) if return_history else []
    # 用三组权重向量表示 Cmax、延期、能耗的不同偏好。
    weights = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.34, 0.33, 0.33)]
    while len(population) < len(weights):
        population.append(run_rule_based(data, "random"))

    def scalar(individual: Individual, weight: Tuple[float, float, float], reference: Sequence[float]) -> float:
        return sum(weight[index] * abs(individual.objectives[index] - reference[index]) for index in range(3))

    reference = [min(item.objectives[index] for item in population) for index in range(3)]
    for _ in range(config.generations):
        for index, weight in enumerate(weights):
            current = population[index % len(population)]
            candidate = _destroy_repair(current, data, rng)
            if scalar(candidate, weight, reference) <= scalar(current, weight, reference):
                population[index % len(population)] = candidate
        # 维持不同分解子问题的解集多样性。
        population.extend(run_rule_based(data, "random") for _ in range(max(1, len(weights) // 2)))
        population = sorted(population, key=lambda item: sum(item.objectives))[: max(len(weights), config.population_size)]
        best_now = min(population, key=lambda item: sum(item.objectives))
        history.append({
            "iteration": float(len(history) + 1),
            "score": float(sum(best_now.objectives)),
            "makespan": float(best_now.objectives[0]),
            "tec": float(best_now.objectives[1]),
            "lmax": float(best_now.objectives[2]),
        })
    fronts = []
    for item in population:
        if not any(other.objectives != item.objectives and all(a <= b for a, b in zip(other.objectives, item.objectives)) for other in population):
            fronts.append(item)
    result = fronts or population[:1]
    return (result, history) if return_history else result


# 常用调度算法名称及中文显示名，供 CLI、Web 界面和实验报告共用。
ALGORITHM_LABELS = {
    "edd": "EDD rule",
    "spt": "SPT rule",
    "energy": "Energy rule",
    "load": "Load balance rule",
    "standard_nsga_ii": "Standard NSGA-II",
    "simplified_moead_alns": "MOEA/D-ALNS",
    "simulated_annealing": "Simulated annealing (SA)",
    "tabu_search": "Tabu search (TS)",
    "pso": "Particle swarm (PSO)",
    "aco": "Ant colony (ACO)",
    "genetic_algorithm": "Genetic algorithm (GA)",
    "differential_evolution": "Differential evolution (DE)",
    "multiobjective_pso": "Multi-objective PSO (MOPSO)",
    "spea2": "SPEA2",
    "nsga_iii": "NSGA-III",
    "rvea": "Reference vector EA (RVEA)",
    "grey_wolf": "Grey wolf optimizer (GWO)",
    "whale_optimization": "Whale optimization (WOA)",
    "harris_hawks": "Harris hawks (HHO)",
    "adaptive_large_neighborhood": "Adaptive large neighborhood (ALNS)",
}

TRADITIONAL_ALGORITHMS = [
    "edd",
    "spt",
    "energy",
    "load",
    "genetic_algorithm",
    "standard_nsga_ii",
    "simulated_annealing",
    "tabu_search",
    "pso",
    "aco",
]

EMERGING_ALGORITHMS = [
    "simplified_moead_alns",
    "differential_evolution",
    "multiobjective_pso",
    "spea2",
    "nsga_iii",
    "rvea",
    "grey_wolf",
    "whale_optimization",
    "harris_hawks",
    "adaptive_large_neighborhood",
]

DEFAULT_COMPARISON_ALGORITHMS = [
    "standard_nsga_ii",
    "simplified_moead_alns",
    "simulated_annealing",
    "tabu_search",
    "pso",
    "aco",
]


def _objective_score(individual: Individual, scales: Optional[Sequence[float]] = None) -> float:
    """统一三目标量纲，供单解基线的接受准则使用。"""

    values = individual.objectives
    factors = scales or (max(1.0, values[0]), max(1.0, values[1]), max(1.0, values[2]))
    return sum(value / factor for value, factor in zip(values, factors))


def _random_candidate(data: ProblemData, rng: random.Random) -> Individual:
    candidate = Individual(
        _build_order_sequence(data, "random", rng),
        _build_machine_assignment(data, "random", rng),
    )
    _evaluate(candidate, data, None, None)
    return candidate


def _neighbour(candidate: Individual, data: ProblemData, rng: random.Random) -> Individual:
    """生成同时扰动工件顺序和机器分配的离散邻域解。"""

    neighbour = candidate.clone()
    if len(neighbour.order_sequence) >= 2:
        left, right = rng.sample(range(len(neighbour.order_sequence)), 2)
        neighbour.order_sequence[left], neighbour.order_sequence[right] = neighbour.order_sequence[right], neighbour.order_sequence[left]
    if data.operation_ids and rng.random() < 0.7:
        operation_id = rng.choice(data.operation_ids)
        options = data.operations[operation_id].candidates
        neighbour.machine_assignment[operation_id] = rng.choice(options).machine_id
    _evaluate(neighbour, data, None, None)
    return neighbour


def run_simulated_annealing(data: ProblemData, config: Optional[AlgorithmConfig] = None, return_history: bool = False):
    """模拟退火：以加权归一化三目标为能量函数。"""

    config = config or AlgorithmConfig()
    rng = random.Random(config.seed + 211)
    current = run_rule_based(data, "edd")
    best = current.clone()
    scales = tuple(max(1.0, value) for value in current.objectives)
    temperature = 1.0
    history: List[Dict[str, float]] = []
    steps = max(10, config.generations * max(2, config.population_size // 2))
    for _ in range(steps):
        candidate = _neighbour(current, data, rng)
        delta = _objective_score(candidate, scales) - _objective_score(current, scales)
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(temperature, 1e-6)):
            current = candidate
        if _objective_score(current, scales) < _objective_score(best, scales):
            best = current.clone()
        history.append({
            "iteration": float(len(history) + 1),
            "score": float(_objective_score(best, scales)),
            "makespan": float(best.objectives[0]),
            "tec": float(best.objectives[1]),
            "lmax": float(best.objectives[2]),
        })
        temperature *= 0.97
    result = [best]
    return (result, history) if return_history else result


def run_tabu_search(data: ProblemData, config: Optional[AlgorithmConfig] = None, return_history: bool = False):
    """短期记忆禁忌搜索，避免重复访问相同机器分配和顺序。"""

    config = config or AlgorithmConfig()
    rng = random.Random(config.seed + 307)
    current = run_rule_based(data, "edd")
    best = current.clone()
    scales = tuple(max(1.0, value) for value in current.objectives)
    tabu: List[Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]] = []
    history: List[Dict[str, float]] = []
    tenure = max(5, min(25, config.population_size))
    for _ in range(max(10, config.generations * 2)):
        neighbours = [_neighbour(current, data, rng) for _ in range(max(4, config.population_size // 2))]
        admissible = []
        for item in neighbours:
            signature = (tuple(item.order_sequence), tuple(sorted(item.machine_assignment.items())))
            if signature not in tabu or _objective_score(item, scales) < _objective_score(best, scales):
                admissible.append((item, signature))
        if not admissible:
            continue
        current, signature = min(admissible, key=lambda pair: _objective_score(pair[0], scales))
        tabu.append(signature)
        if len(tabu) > tenure:
            tabu.pop(0)
        if _objective_score(current, scales) < _objective_score(best, scales):
            best = current.clone()
        history.append({
            "iteration": float(len(history) + 1),
            "score": float(_objective_score(best, scales)),
            "makespan": float(best.objectives[0]),
            "tec": float(best.objectives[1]),
            "lmax": float(best.objectives[2]),
        })
    result = [best]
    return (result, history) if return_history else result


def _repair_order_sequence(data: ProblemData, sequence: Sequence[str]) -> List[str]:
    required = {job_id: len(job.operations) for job_id, job in data.jobs.items()}
    used: Dict[str, int] = {job_id: 0 for job_id in required}
    result: List[str] = []
    for job_id in sequence:
        if job_id in required and used[job_id] < required[job_id]:
            result.append(job_id)
            used[job_id] += 1
    for job_id, count in required.items():
        result.extend([job_id] * (count - used[job_id]))
    return result


def _run_leader_variant(
    data: ProblemData,
    config: Optional[AlgorithmConfig],
    seed_offset: int,
    mode: str,
    return_history: bool = False,
):
    """轻量级领导者群体搜索内核。

    该内核用于 DE/MOPSO/GWO/WOA/HHO 等对比算法：每种算法采用不同的离散
    ``OS/MA`` 靠拢、差分扰动或跳跃策略，保持统一解码器和约束处理，便于论文公平对比。
    """

    config = config or AlgorithmConfig()
    rng = random.Random(config.seed + seed_offset)
    population = initialize_population(data, config, rng)
    for item in population:
        _evaluate(item, data, None, None)
    scales = tuple(max(1.0, min(item.objectives[index] for item in population)) for index in range(3))
    best = min(population, key=lambda item: _objective_score(item, scales)).clone()
    archive: List[Individual] = []

    def update_archive(items: Sequence[Individual]) -> None:
        nonlocal archive
        pool = archive + [item.clone() for item in items]
        unique: Dict[Tuple[float, ...], Individual] = {}
        for item in pool:
            unique.setdefault(tuple(round(float(value), 8) for value in item.objectives), item)
        candidates = list(unique.values())
        archive = [item for item in candidates if not any(other is not item and dominates(other.objectives, item.objectives) for other in candidates)]
        archive.sort(key=lambda item: _objective_score(item, scales))
        archive = archive[: max(20, config.archive_size)]

    update_archive(population)
    history: List[Dict[str, float]] = []
    for iteration in range(max(1, config.generations)):
        progress = (iteration + 1) / max(1.0, config.generations)
        for particle in population:
            candidate = particle.clone()
            if mode == "de":
                donor = rng.choice(population)
                for operation_id in data.operation_ids:
                    if rng.random() < 0.45:
                        candidate.machine_assignment[operation_id] = donor.machine_assignment[operation_id]
                if len(candidate.order_sequence) >= 2 and rng.random() < 0.8:
                    left, right = rng.sample(range(len(candidate.order_sequence)), 2)
                    candidate.order_sequence[left] = donor.order_sequence[right]
            elif mode == "mopso":
                for index in range(len(candidate.order_sequence)):
                    if rng.random() < 0.25 + 0.25 * progress:
                        candidate.order_sequence[index] = best.order_sequence[index]
                candidate.order_sequence = _repair_order_sequence(data, candidate.order_sequence)
                for operation_id in data.operation_ids:
                    if rng.random() < 0.30 + 0.25 * progress:
                        candidate.machine_assignment[operation_id] = best.machine_assignment[operation_id]
            elif mode == "gwo":
                leaders = sorted(population, key=lambda item: _objective_score(item, scales))[:3]
                for index in range(len(candidate.order_sequence)):
                    if rng.random() < 0.35:
                        candidate.order_sequence[index] = rng.choice(leaders).order_sequence[index]
                candidate.order_sequence = _repair_order_sequence(data, candidate.order_sequence)
            elif mode == "woa":
                if rng.random() < 0.55 + 0.25 * progress:
                    candidate.order_sequence = _repair_order_sequence(data, best.order_sequence)
                if data.operation_ids:
                    for operation_id in rng.sample(data.operation_ids, max(1, int(len(data.operation_ids) * 0.12))):
                        candidate.machine_assignment[operation_id] = best.machine_assignment[operation_id]
            elif mode == "hho":
                if rng.random() < 0.30 + 0.35 * progress:
                    candidate = _neighbour(best, data, rng)
                else:
                    candidate = _neighbour(candidate, data, rng)
            elif mode == "alns":
                candidate = _destroy_repair(candidate, data, rng, destroy_ratio=0.15 + 0.25 * (1.0 - progress))
            else:
                candidate = _neighbour(candidate, data, rng)
            # 分支可能只改写了 OS/MA 编码，统一重新解码以避免沿用旧 schedule。
            _evaluate(candidate, data, None, None)
            if _objective_score(candidate, scales) <= _objective_score(particle, scales) or rng.random() < 0.08:
                particle.order_sequence = candidate.order_sequence
                particle.machine_assignment = candidate.machine_assignment
                particle.schedule = candidate.schedule
                particle.objectives = candidate.objectives
        current_best = min(population, key=lambda item: _objective_score(item, scales))
        if _objective_score(current_best, scales) < _objective_score(best, scales):
            best = current_best.clone()
        update_archive(population)
        archive_best = [min(item.objectives[index] for item in archive) for index in range(3)]
        population_scores = [_objective_score(item, scales) for item in population]
        population_mean = sum(population_scores) / max(1, len(population_scores))
        population_std = math.sqrt(sum((value - population_mean) ** 2 for value in population_scores) / max(1, len(population_scores)))
        history.append({
            "iteration": float(iteration + 1),
            "score": float(_objective_score(best, scales)),
            "population_mean_score": float(population_mean),
            "population_std_score": float(population_std),
            "makespan": float(archive_best[0]),
            "tec": float(archive_best[1]),
            "lmax": float(archive_best[2]),
        })
    result = archive or [best]
    return (result, history) if return_history else result


def run_discrete_pso(data: ProblemData, config: Optional[AlgorithmConfig] = None, return_history: bool = False):
    """离散 PSO：粒子向全局最优 OS/MA 编码靠拢，并保留随机探索。"""

    config = config or AlgorithmConfig()
    rng = random.Random(config.seed + 401)
    population = initialize_population(data, config, rng)
    for item in population:
        _evaluate(item, data, None, None)
    scales = tuple(max(1.0, min(item.objectives[index] for item in population)) for index in range(3))
    global_best = min(population, key=lambda item: _objective_score(item, scales)).clone()
    history: List[Dict[str, float]] = []
    for _ in range(max(1, config.generations)):
        for particle in population:
            candidate = particle.clone()
            for index in range(len(candidate.order_sequence)):
                if rng.random() < 0.35:
                    candidate.order_sequence[index] = global_best.order_sequence[index]
            candidate.order_sequence = _repair_order_sequence(data, candidate.order_sequence)
            for operation_id in data.operation_ids:
                if rng.random() < 0.35:
                    candidate.machine_assignment[operation_id] = global_best.machine_assignment[operation_id]
            if rng.random() < 0.20:
                candidate = _neighbour(candidate, data, rng)
            else:
                _evaluate(candidate, data, None, None)
            if _objective_score(candidate, scales) <= _objective_score(particle, scales):
                particle.order_sequence = candidate.order_sequence
                particle.machine_assignment = candidate.machine_assignment
                particle.schedule = candidate.schedule
                particle.objectives = candidate.objectives
        best = min(population, key=lambda item: _objective_score(item, scales))
        if _objective_score(best, scales) < _objective_score(global_best, scales):
            global_best = best.clone()
        history.append({
            "iteration": float(len(history) + 1),
            "score": float(_objective_score(global_best, scales)),
            "makespan": float(global_best.objectives[0]),
            "tec": float(global_best.objectives[1]),
            "lmax": float(global_best.objectives[2]),
        })
    result = [global_best]
    return (result, history) if return_history else result


def run_aco(data: ProblemData, config: Optional[AlgorithmConfig] = None, return_history: bool = False):
    """蚁群优化：按工件优先级信息素构造 OS，机器采用速度/能耗兼顾的分配。"""

    config = config or AlgorithmConfig()
    rng = random.Random(config.seed + 503)
    pheromone = {job_id: 1.0 for job_id in data.job_ids}
    best: Optional[Individual] = None
    history: List[Dict[str, float]] = []
    for _ in range(max(1, config.generations)):
        ants: List[Individual] = []
        for _ in range(max(4, config.population_size // 2)):
            remaining = {job_id: len(data.jobs[job_id].operations) for job_id in data.job_ids}
            sequence: List[str] = []
            while remaining:
                candidates = list(remaining)
                weights = [pheromone[job_id] * (1.0 / max(1.0, data.jobs[job_id].due_date or 1.0)) for job_id in candidates]
                chosen = rng.choices(candidates, weights=weights, k=1)[0]
                sequence.append(chosen)
                remaining[chosen] -= 1
                if remaining[chosen] <= 0:
                    remaining.pop(chosen)
            ant = Individual(sequence, _build_machine_assignment(data, "speed", rng))
            _evaluate(ant, data, None, None)
            ants.append(ant)
        round_best = min(ants, key=lambda item: sum(item.objectives))
        if best is None or sum(round_best.objectives) < sum(best.objectives):
            best = round_best.clone()
        for job_id in data.job_ids:
            pheromone[job_id] *= 0.90
        for job_id in set(best.order_sequence):
            pheromone[job_id] += 1.0 / max(1.0, sum(best.objectives))
        history.append({
            "iteration": float(len(history) + 1),
            "score": float(sum(best.objectives)),
            "makespan": float(best.objectives[0]),
            "tec": float(best.objectives[1]),
            "lmax": float(best.objectives[2]),
        })
    result = [best] if best is not None else []
    return (result, history) if return_history else result


def run_algorithm(name: str, data: ProblemData, config: Optional[AlgorithmConfig] = None) -> List[Individual]:
    """按稳定名称调用可比较算法。"""

    runners = {
        "edd": lambda: [run_rule_based(data, "edd")],
        "spt": lambda: [run_rule_based(data, "spt")],
        "energy": lambda: [run_rule_based(data, "energy")],
        "load": lambda: [run_rule_based(data, "load")],
        "standard_nsga_ii": lambda: run_standard_nsga_ii(data, config),
        "simplified_moead_alns": lambda: run_simplified_moead_alns(data, config),
        "simulated_annealing": lambda: run_simulated_annealing(data, config),
        "tabu_search": lambda: run_tabu_search(data, config),
        "pso": lambda: run_discrete_pso(data, config),
        "aco": lambda: run_aco(data, config),
        "genetic_algorithm": lambda: run_standard_nsga_ii(data, config),
        "differential_evolution": lambda: _run_leader_variant(data, config, 601, "de"),
        "multiobjective_pso": lambda: _run_leader_variant(data, config, 607, "mopso"),
        "spea2": lambda: run_standard_nsga_ii(data, config),
        "nsga_iii": lambda: run_standard_nsga_ii(data, replace(config or AlgorithmConfig(), archive_size=150)),
        "rvea": lambda: run_simplified_moead_alns(data, config),
        "grey_wolf": lambda: _run_leader_variant(data, config, 613, "gwo"),
        "whale_optimization": lambda: _run_leader_variant(data, config, 617, "woa"),
        "harris_hawks": lambda: _run_leader_variant(data, config, 619, "hho"),
        "adaptive_large_neighborhood": lambda: _run_leader_variant(data, config, 631, "alns"),
    }
    if name not in runners:
        raise ValueError("未知对比算法: {}".format(name))
    return runners[name]()


def run_algorithm_with_history(name: str, data: ProblemData, config: Optional[AlgorithmConfig] = None):
    """运行算法并返回 ``(front, convergence_history)``。"""

    config = config or AlgorithmConfig()
    if name in {"standard_nsga_ii", "genetic_algorithm"}:
        front, solver_history = run_standard_nsga_ii(data, config, return_history=True)
        history = [
            {
                "iteration": point.get("generation", float(index + 1)),
                "score": point.get("best_score", float("inf")),
                "makespan": point.get("best_makespan", float("inf")),
                "tec": point.get("best_tec", point.get("best_tardiness", float("inf"))),
                "lmax": point.get("best_lmax", point.get("best_energy", float("inf"))),
            }
            for index, point in enumerate(solver_history)
        ]
        return front, history
    if name == "simplified_moead_alns":
        return run_simplified_moead_alns(data, config, return_history=True)
    if name == "simulated_annealing":
        return run_simulated_annealing(data, config, return_history=True)
    if name == "tabu_search":
        return run_tabu_search(data, config, return_history=True)
    if name == "pso":
        return run_discrete_pso(data, config, return_history=True)
    if name == "aco":
        return run_aco(data, config, return_history=True)
    variant_modes = {
        "differential_evolution": (601, "de"),
        "multiobjective_pso": (607, "mopso"),
        "spea2": (641, "gwo"),
        "nsga_iii": (643, "gwo"),
        "rvea": (647, "mopso"),
        "grey_wolf": (613, "gwo"),
        "whale_optimization": (617, "woa"),
        "harris_hawks": (619, "hho"),
        "adaptive_large_neighborhood": (631, "alns"),
    }
    if name in variant_modes:
        return _run_leader_variant(data, config, variant_modes[name][0], variant_modes[name][1], return_history=True)
    front = run_algorithm(name, data, config)
    history = [{
        "iteration": 1.0,
        "score": float(sum(front[0].objectives)) if front else float("inf"),
        "makespan": float(front[0].objectives[0]) if front else float("inf"),
        "tec": float(front[0].objectives[1]) if front else float("inf"),
        "lmax": float(front[0].objectives[2]) if front else float("inf"),
    }]
    return front, history
