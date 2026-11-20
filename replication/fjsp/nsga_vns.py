"""面向 FJSP 的自适应精英 NSGA-II-VNS。

算法方法块：
1. 多规则、扰动式双层编码初始化；
2. 支持单目标、加权目标和 Pareto 多目标的统一选择策略；
3. POX/PPX 交叉、自适应变异与重复个体抑制；
4. 面向瓶颈机器、关键工序和延期工件的 VNS；
5. 外部非支配档案、固定参考尺度 HV 和停滞重启；
6. 解码结果缓存，减少重复适应度计算。
"""

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .config import AlgorithmConfig
from .data_model import ProblemData, Schedule
from .decoder import DecodeContext, decode, repair_machine_assignment
from .objectives import approximate_hypervolume, dominates, dynamic_objectives, static_objectives


OBJECTIVE_MODES = ("multi", "makespan", "tardiness", "energy", "weighted_sum")


@dataclass
class Individual:
    """一个 OS/MA 双层编码个体。"""

    order_sequence: List[str]
    machine_assignment: Dict[str, str]
    schedule: Optional[Schedule] = None
    objectives: Tuple[float, ...] = tuple()
    rank: int = 0
    crowding: float = 0.0

    def clone(self) -> "Individual":
        return Individual(
            order_sequence=list(self.order_sequence),
            machine_assignment=dict(self.machine_assignment),
            schedule=self.schedule.copy() if self.schedule is not None else None,
            objectives=tuple(self.objectives),
            rank=self.rank,
            crowding=self.crowding,
        )


@dataclass
class _ObjectivePolicy:
    """统一单目标和多目标的比较、标量化尺度。"""

    mode: str
    weights: Tuple[float, ...]
    ideal: Tuple[float, ...]
    nadir: Tuple[float, ...]

    @classmethod
    def from_population(cls, population: Sequence[Individual], config: AlgorithmConfig) -> "_ObjectivePolicy":
        if not population or not population[0].objectives:
            raise ValueError("无法从空种群建立目标尺度")
        mode = str(config.objective_mode).strip().lower()
        if mode not in OBJECTIVE_MODES:
            raise ValueError("objective_mode 必须是 {} 之一".format(", ".join(OBJECTIVE_MODES)))
        dimension_count = len(population[0].objectives)
        raw_weights = tuple(max(0.0, float(value)) for value in config.objective_weights)
        if len(raw_weights) != dimension_count:
            raise ValueError("objective_weights 数量必须与目标数量一致")
        weight_sum = sum(raw_weights)
        weights = tuple(value / weight_sum for value in raw_weights) if weight_sum > 1e-12 else tuple(1.0 / dimension_count for _ in range(dimension_count))
        dimensions = list(zip(*[item.objectives for item in population]))
        return cls(
            mode=mode,
            weights=weights,
            # 三项目标均为非负代价，0 是跨代稳定且可解释的理论理想点。
            ideal=tuple(0.0 for _ in dimensions),
            nadir=tuple(max(values) for values in dimensions),
        )

    def normalized(self, objectives: Sequence[float]) -> Tuple[float, ...]:
        values = []
        for value, low, high in zip(objectives, self.ideal, self.nadir):
            scale = max(1e-12, high - low, abs(high) * 0.05, 1.0)
            values.append((value - low) / scale)
        return tuple(values)

    def score(self, individual: Individual, weights: Optional[Sequence[float]] = None) -> float:
        target_index = {"makespan": 0, "tardiness": 1, "energy": 1}.get(self.mode)
        if target_index is not None:
            return float(individual.objectives[target_index])
        active_weights = tuple(weights) if weights is not None else self.weights
        normalized = self.normalized(individual.objectives)
        weighted = [weight * value for weight, value in zip(active_weights, normalized)]
        # 增广 Tchebycheff 在兼顾总改善的同时，避免某一目标严重退化。
        return max(weighted) + 0.05 * sum(weighted)

    @property
    def is_multiobjective(self) -> bool:
        return self.mode == "multi"


def _job_operation_count(data: ProblemData) -> Dict[str, int]:
    return {job_id: len(job.operations) for job_id, job in data.jobs.items()}


def _job_priority_order(data: ProblemData, strategy: str, rng: random.Random) -> List[str]:
    if strategy == "edd":
        return sorted(data.job_ids, key=lambda job_id: (data.jobs[job_id].due_date, data.jobs[job_id].release_time))
    if strategy == "spt":
        return sorted(
            data.job_ids,
            key=lambda job_id: sum(min(option.processing_time for option in op.candidates) for op in data.jobs[job_id].operations),
        )
    if strategy == "energy":
        return sorted(
            data.job_ids,
            key=lambda job_id: sum(min(option.total_energy for option in op.candidates) for op in data.jobs[job_id].operations),
        )
    if strategy == "load":
        return sorted(data.job_ids, key=lambda job_id: -sum(len(op.candidates) for op in data.jobs[job_id].operations))
    jobs = list(data.job_ids)
    rng.shuffle(jobs)
    return jobs


def _build_order_sequence(data: ProblemData, strategy: str, rng: random.Random) -> List[str]:
    """按规则生成合法的重复工件 OS 编码。"""

    counts = _job_operation_count(data)
    if strategy == "random":
        sequence = [job_id for job_id, count in counts.items() for _ in range(count)]
        rng.shuffle(sequence)
        return sequence
    ordered_jobs = _job_priority_order(data, strategy, rng)
    sequence: List[str] = []
    remaining = dict(counts)
    while remaining:
        for job_id in ordered_jobs:
            if remaining.get(job_id, 0) <= 0:
                continue
            sequence.append(job_id)
            remaining[job_id] -= 1
            if remaining[job_id] <= 0:
                remaining.pop(job_id, None)
    return sequence


def _build_machine_assignment(data: ProblemData, strategy: str, rng: random.Random) -> Dict[str, str]:
    assignment: Dict[str, str] = {}
    machine_load: Dict[str, float] = {machine_id: 0.0 for machine_id in data.machine_ids}
    for operation_id in data.operation_ids:
        operation = data.operations[operation_id]
        if strategy == "energy":
            option = min(operation.candidates, key=lambda item: (item.total_energy, item.processing_time))
        elif strategy == "speed":
            option = min(operation.candidates, key=lambda item: (item.processing_time, item.total_energy))
        elif strategy == "load":
            option = min(operation.candidates, key=lambda item: (machine_load[item.machine_id] + item.processing_time, item.processing_time))
        else:
            option = rng.choice(operation.candidates)
        assignment[operation_id] = option.machine_id
        machine_load[option.machine_id] += option.processing_time
    return repair_machine_assignment(data, assignment)


def _perturb_sequence(sequence: List[str], rng: random.Random, strength: int = 1) -> None:
    if len(sequence) < 2 or len(set(sequence)) < 2:
        return
    for _ in range(max(1, strength)):
        different = [(left, right) for left, right in (rng.sample(range(len(sequence)), 2) for _ in range(6)) if sequence[left] != sequence[right]]
        if not different:
            return
        left, right = different[0]
        if rng.random() < 0.5:
            sequence[left], sequence[right] = sequence[right], sequence[left]
        else:
            value = sequence.pop(left)
            sequence.insert(right, value)


def _individual_signature(individual: Individual, data: ProblemData) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    return tuple(individual.order_sequence), tuple(individual.machine_assignment.get(operation_id, "") for operation_id in data.operation_ids)


def initialize_population(data: ProblemData, config: AlgorithmConfig, rng: random.Random) -> List[Individual]:
    """构造极值规则种子，并在规则附近生成分散解。"""

    order_strategies = ["edd", "spt", "energy", "load", "random"]
    machine_strategies = ["speed", "energy", "load", "random"]
    combinations = [(order, machine) for order in order_strategies for machine in machine_strategies]
    population: List[Individual] = []
    signatures: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()
    for index in range(max(1, config.population_size)):
        if str(config.initialization_mode).strip().lower() == "random":
            order_strategy, machine_strategy = "random", "random"
        else:
            order_strategy, machine_strategy = combinations[index % len(combinations)]
        sequence = _build_order_sequence(data, order_strategy, rng)
        assignment = _build_machine_assignment(data, machine_strategy, rng)
        if index >= len(order_strategies):
            _perturb_sequence(sequence, rng, 1 + index // max(1, len(combinations)))
        if index >= len(machine_strategies):
            change_count = max(1, min(len(data.operation_ids), 1 + index // max(1, len(combinations))))
            for operation_id in rng.sample(data.operation_ids, change_count):
                assignment[operation_id] = rng.choice(data.operations[operation_id].candidates).machine_id
        individual = Individual(sequence, repair_machine_assignment(data, assignment))
        for retry in range(4):
            signature = _individual_signature(individual, data)
            if signature not in signatures:
                break
            _perturb_sequence(individual.order_sequence, rng, retry + 1)
            operation_id = rng.choice(data.operation_ids)
            individual.machine_assignment[operation_id] = rng.choice(data.operations[operation_id].candidates).machine_id
        signatures.add(_individual_signature(individual, data))
        population.append(individual)
    return population


def _evaluate(
    individual: Individual,
    data: ProblemData,
    context: Optional[DecodeContext],
    dynamic_baseline: Optional[Schedule],
) -> Individual:
    individual.machine_assignment = repair_machine_assignment(data, individual.machine_assignment)
    individual.schedule = decode(data, individual.order_sequence, individual.machine_assignment, context)
    if dynamic_baseline is None:
        individual.objectives = static_objectives(data, individual.schedule)
    else:
        individual.objectives = dynamic_objectives(data, individual.schedule, dynamic_baseline)
    return individual


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """Deb 快速非支配排序。"""

    if not population:
        return []
    fronts: List[List[Individual]] = [[]]
    domination_count: Dict[int, int] = {}
    dominated: Dict[int, List[int]] = {}
    population_index = {id(individual): index for index, individual in enumerate(population)}
    for index, individual in enumerate(population):
        dominated[index] = []
        domination_count[index] = 0
        for other_index, other in enumerate(population):
            if index == other_index:
                continue
            if dominates(individual.objectives, other.objectives):
                dominated[index].append(other_index)
            elif dominates(other.objectives, individual.objectives):
                domination_count[index] += 1
        if domination_count[index] == 0:
            individual.rank = 0
            fronts[0].append(individual)
    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: List[Individual] = []
        for individual in fronts[front_index]:
            source_index = population_index[id(individual)]
            for target_index in dominated[source_index]:
                domination_count[target_index] -= 1
                if domination_count[target_index] == 0:
                    population[target_index].rank = front_index + 1
                    next_front.append(population[target_index])
        if next_front:
            fronts.append(next_front)
        front_index += 1
    return fronts


def assign_crowding(front: List[Individual]) -> None:
    """计算量纲归一化的拥挤距离。"""

    if not front:
        return
    for individual in front:
        individual.crowding = 0.0
    for dimension in range(len(front[0].objectives)):
        front.sort(key=lambda item: item.objectives[dimension])
        front[0].crowding = float("inf")
        front[-1].crowding = float("inf")
        low = front[0].objectives[dimension]
        high = front[-1].objectives[dimension]
        if abs(high - low) < 1e-12:
            continue
        for index in range(1, len(front) - 1):
            if front[index].crowding != float("inf"):
                front[index].crowding += (front[index + 1].objectives[dimension] - front[index - 1].objectives[dimension]) / (high - low)


def _deduplicate_genotypes(population: Sequence[Individual], data: Optional[ProblemData] = None) -> List[Individual]:
    unique: List[Individual] = []
    signatures: Set[object] = set()
    for individual in population:
        if data is None:
            signature = (tuple(individual.order_sequence), tuple(sorted(individual.machine_assignment.items())))
        else:
            signature = _individual_signature(individual, data)
        if signature not in signatures:
            signatures.add(signature)
            unique.append(individual)
    return unique


def environmental_selection(
    population: List[Individual],
    target_size: int,
    policy: Optional[_ObjectivePolicy] = None,
    data: Optional[ProblemData] = None,
) -> List[Individual]:
    """单目标采用精英排序，多目标采用去重后的 NSGA-II 环境选择。"""

    if not population or target_size <= 0:
        return []
    candidates = _deduplicate_genotypes(population, data)
    if policy is None:
        dimensions = list(zip(*[item.objectives for item in candidates]))
        dimension_count = len(dimensions)
        policy = _ObjectivePolicy(
            "multi",
            tuple(1.0 / dimension_count for _ in range(dimension_count)),
            tuple(min(values) for values in dimensions),
            tuple(max(values) for values in dimensions),
        )
    if not policy.is_multiobjective:
        candidates.sort(key=lambda item: (policy.score(item), sum(item.objectives), tuple(item.objectives)))
        # 70% 直接按目标择优，余下位置由辅助 Pareto 选择补足，降低单目标早熟风险。
        elite_count = min(len(candidates), max(1, int(target_size * 0.70)))
        selected = candidates[:elite_count]
        auxiliary = _ObjectivePolicy("multi", policy.weights, policy.ideal, policy.nadir)
        diverse = environmental_selection(candidates, target_size, auxiliary, data)
        selected_signatures = {
            _individual_signature(item, data) if data is not None else (tuple(item.order_sequence), tuple(sorted(item.machine_assignment.items())))
            for item in selected
        }
        for item in diverse:
            signature = _individual_signature(item, data) if data is not None else (tuple(item.order_sequence), tuple(sorted(item.machine_assignment.items())))
            if signature not in selected_signatures:
                selected.append(item)
                selected_signatures.add(signature)
            if len(selected) >= target_size:
                break
        if len(selected) < min(target_size, len(candidates)):
            selected_ids = {id(item) for item in selected}
            selected.extend(item for item in candidates if id(item) not in selected_ids)
            selected = selected[:target_size]
        selected.sort(key=lambda item: (policy.score(item), sum(item.objectives)))
        for rank, item in enumerate(selected):
            item.rank = rank
            item.crowding = 1.0 / (1.0 + rank)
        return [item.clone() for item in selected]

    selected: List[Individual] = []
    duplicate_objectives: List[Individual] = []
    for front in fast_non_dominated_sort(candidates):
        assign_crowding(front)
        seen_objectives: Set[Tuple[float, ...]] = set()
        distinct: List[Individual] = []
        for item in sorted(front, key=lambda value: value.crowding, reverse=True):
            key = tuple(round(value, 9) for value in item.objectives)
            if key in seen_objectives:
                duplicate_objectives.append(item)
            else:
                seen_objectives.add(key)
                distinct.append(item)
        if len(selected) + len(distinct) <= target_size:
            selected.extend(distinct)
        else:
            selected.extend(distinct[: target_size - len(selected)])
            break
        if len(selected) >= target_size:
            break
    if len(selected) < target_size:
        chosen_ids = {id(item) for item in selected}
        fallback = [item for item in duplicate_objectives if id(item) not in chosen_ids]
        fallback.sort(key=lambda item: (item.rank, -item.crowding, policy.score(item)))
        selected.extend(fallback[: target_size - len(selected)])
    return [item.clone() for item in selected]


def _tournament(population: List[Individual], rng: random.Random, size: int, policy: _ObjectivePolicy) -> Individual:
    competitors = rng.sample(population, min(max(2, size), len(population)))
    if policy.is_multiobjective:
        return min(competitors, key=lambda item: (item.rank, -item.crowding, policy.score(item)))
    return min(competitors, key=lambda item: (policy.score(item), item.rank, -item.crowding))


def _ppx(parent_a: Sequence[str], parent_b: Sequence[str], rng: random.Random) -> List[str]:
    """优先级保持交叉，严格保持各工件出现次数。"""

    required: Dict[str, int] = {}
    for job_id in parent_a:
        required[job_id] = required.get(job_id, 0) + 1
    pointers = [0, 0]
    parents = [parent_a, parent_b]
    result: List[str] = []
    used: Dict[str, int] = {}
    while len(result) < len(parent_a):
        source = rng.randrange(2)
        for _ in range(2):
            parent = parents[source]
            while pointers[source] < len(parent) and used.get(parent[pointers[source]], 0) >= required[parent[pointers[source]]]:
                pointers[source] += 1
            if pointers[source] < len(parent):
                break
            source = 1 - source
        parent = parents[source]
        if pointers[source] >= len(parent):
            break
        job_id = parent[pointers[source]]
        pointers[source] += 1
        result.append(job_id)
        used[job_id] = used.get(job_id, 0) + 1
    return result


def _pox(parent_a: Sequence[str], parent_b: Sequence[str], rng: random.Random) -> List[str]:
    """工件集合保持交叉，保留父代关键工件的绝对位置。"""

    jobs = sorted(set(parent_a))
    if len(jobs) <= 1:
        return list(parent_a)
    selected = set(rng.sample(jobs, rng.randint(1, len(jobs) - 1)))
    child: List[Optional[str]] = [None] * len(parent_a)
    for index, job_id in enumerate(parent_a):
        if job_id in selected:
            child[index] = job_id
    fill = [job_id for job_id in parent_b if job_id not in selected]
    fill_index = 0
    for index, value in enumerate(child):
        if value is None:
            child[index] = fill[fill_index]
            fill_index += 1
    return [str(value) for value in child]


def crossover(parent_a: Individual, parent_b: Individual, data: ProblemData, rng: random.Random) -> Tuple[Individual, Individual]:
    if rng.random() < 0.5:
        child_a_sequence = _pox(parent_a.order_sequence, parent_b.order_sequence, rng)
        child_b_sequence = _pox(parent_b.order_sequence, parent_a.order_sequence, rng)
    else:
        child_a_sequence = _ppx(parent_a.order_sequence, parent_b.order_sequence, rng)
        child_b_sequence = _ppx(parent_b.order_sequence, parent_a.order_sequence, rng)
    child_a_machine: Dict[str, str] = {}
    child_b_machine: Dict[str, str] = {}
    for operation_id in data.operation_ids:
        if rng.random() < 0.5:
            child_a_machine[operation_id] = parent_a.machine_assignment[operation_id]
            child_b_machine[operation_id] = parent_b.machine_assignment[operation_id]
        else:
            child_a_machine[operation_id] = parent_b.machine_assignment[operation_id]
            child_b_machine[operation_id] = parent_a.machine_assignment[operation_id]
    return Individual(child_a_sequence, child_a_machine), Individual(child_b_sequence, child_b_machine)


def _preferred_machine(data: ProblemData, operation_id: str, mode: str, weights: Sequence[float]) -> str:
    candidates = data.operations[operation_id].candidates
    if mode in {"makespan", "tardiness"}:
        return min(candidates, key=lambda item: (item.processing_time, item.setup_time, item.total_energy)).machine_id
    if mode == "energy":
        return min(candidates, key=lambda item: (item.total_energy, item.processing_time)).machine_id
    min_time = min(item.processing_time for item in candidates)
    max_time = max(item.processing_time for item in candidates)
    min_energy = min(item.total_energy for item in candidates)
    max_energy = max(item.total_energy for item in candidates)

    def score(option: object) -> float:
        time_score = (option.processing_time - min_time) / max(1e-12, max_time - min_time)
        energy_score = (option.total_energy - min_energy) / max(1e-12, max_energy - min_energy)
        # 静态目标顺序为 Cmax, TEC, Lmax；加工时间是负载的代理。
        return (weights[0] + weights[2]) * time_score + weights[1] * energy_score

    return min(candidates, key=score).machine_id


def mutate(
    individual: Individual,
    data: ProblemData,
    mutation_rate: float,
    machine_rate: float,
    rng: random.Random,
    objective_mode: str = "multi",
    objective_weights: Sequence[float] = (1.0, 1.0, 1.0),
) -> None:
    """OS 交换/插入和 MA 引导式变异。"""

    if len(individual.order_sequence) >= 2 and rng.random() < mutation_rate:
        for _ in range(6):
            left, right = rng.sample(range(len(individual.order_sequence)), 2)
            if individual.order_sequence[left] != individual.order_sequence[right]:
                individual.order_sequence[left], individual.order_sequence[right] = individual.order_sequence[right], individual.order_sequence[left]
                break
    if len(individual.order_sequence) >= 2 and rng.random() < mutation_rate:
        source = rng.randrange(len(individual.order_sequence))
        target = rng.randrange(len(individual.order_sequence))
        value = individual.order_sequence.pop(source)
        individual.order_sequence.insert(target, value)
    if rng.random() < machine_rate:
        operation_id = rng.choice(data.operation_ids)
        alternatives = [item.machine_id for item in data.operations[operation_id].candidates if item.machine_id != individual.machine_assignment[operation_id]]
        if alternatives:
            preferred = _preferred_machine(data, operation_id, objective_mode, objective_weights)
            individual.machine_assignment[operation_id] = preferred if preferred in alternatives and rng.random() < 0.70 else rng.choice(alternatives)


def _score_pair(
    left: Individual,
    right: Individual,
    policy: Optional[_ObjectivePolicy] = None,
    weights: Optional[Sequence[float]] = None,
) -> bool:
    """局部搜索接受准则：支配优先，否则使用固定尺度标量化。"""

    if dominates(left.objectives, right.objectives):
        return True
    if dominates(right.objectives, left.objectives):
        return False
    if policy is None:
        dimensions = list(zip(left.objectives, right.objectives))
        count = len(dimensions)
        policy = _ObjectivePolicy(
            "multi",
            tuple(1.0 / count for _ in range(count)),
            tuple(min(values) for values in dimensions),
            tuple(max(values) for values in dimensions),
        )
    return policy.score(left, weights) < policy.score(right, weights) - 1e-12


def _operation_position(sequence: Sequence[str], job_id: str, op_index: int) -> Optional[int]:
    occurrence = 0
    for index, value in enumerate(sequence):
        if value == job_id:
            occurrence += 1
            if occurrence == op_index:
                return index
    return None


def _target_job(individual: Individual, data: ProblemData, mode: str) -> str:
    if individual.schedule is None or not individual.schedule.entries:
        return min(data.job_ids, key=lambda job_id: data.jobs[job_id].due_date)
    if mode == "tardiness":
        completions = {
            job_id: max((entry.end for entry in individual.schedule.entries_for_job(job_id)), default=0.0)
            for job_id in data.job_ids
        }
        return max(
            data.job_ids,
            key=lambda job_id: data.jobs[job_id].tardiness_weight * max(0.0, completions[job_id] - data.jobs[job_id].due_date),
        )
    return max(individual.schedule.entries.values(), key=lambda entry: entry.end).job_id


def _target_operation(individual: Individual, data: ProblemData, mode: str, rng: random.Random) -> str:
    if individual.schedule is None or not individual.schedule.entries:
        return rng.choice(data.operation_ids)
    entries = list(individual.schedule.entries.values())
    if mode == "energy":
        return max(entries, key=lambda entry: entry.energy).operation_id
    if mode == "tardiness":
        job_id = _target_job(individual, data, mode)
        return max(individual.schedule.entries_for_job(job_id), key=lambda entry: entry.end).operation_id
    return max(entries, key=lambda entry: entry.end).operation_id


def _propose_neighbor(
    current: Individual,
    neighborhood: str,
    data: ProblemData,
    policy: _ObjectivePolicy,
    weights: Sequence[float],
    rng: random.Random,
) -> Individual:
    candidate = current.clone()
    candidate.schedule = None
    candidate.objectives = tuple()
    if neighborhood == "swap":
        _perturb_sequence(candidate.order_sequence, rng, 1)
    elif neighborhood == "insert":
        job_id = _target_job(current, data, policy.mode)
        positions = [index for index, value in enumerate(candidate.order_sequence) if value == job_id and index > 0]
        if positions:
            source = rng.choice(positions)
            target = rng.randrange(source)
            value = candidate.order_sequence.pop(source)
            candidate.order_sequence.insert(target, value)
    elif neighborhood == "critical_block":
        if current.schedule is not None:
            machine_id = max(
                data.machine_ids,
                key=lambda machine: sum(entry.duration for entry in current.schedule.entries_for_machine(machine)),
            )
            entries = current.schedule.entries_for_machine(machine_id)
            pairs = [(left, right) for left, right in zip(entries, entries[1:]) if left.job_id != right.job_id]
            if pairs:
                left_entry, right_entry = rng.choice(pairs)
                left = _operation_position(candidate.order_sequence, left_entry.job_id, left_entry.op_index)
                right = _operation_position(candidate.order_sequence, right_entry.job_id, right_entry.op_index)
                if left is not None and right is not None:
                    candidate.order_sequence[left], candidate.order_sequence[right] = candidate.order_sequence[right], candidate.order_sequence[left]
    elif neighborhood == "machine":
        operation_id = _target_operation(current, data, policy.mode, rng)
        alternatives = [item.machine_id for item in data.operations[operation_id].candidates if item.machine_id != candidate.machine_assignment[operation_id]]
        if alternatives:
            if current.schedule is not None and policy.mode in {"multi", "makespan", "tardiness", "weighted_sum"}:
                machine_load = {
                    machine_id: sum(entry.duration for entry in current.schedule.entries_for_machine(machine_id))
                    for machine_id in data.machine_ids
                }
                option_by_machine = {item.machine_id: item for item in data.operations[operation_id].candidates}
                preferred = min(alternatives, key=lambda machine_id: machine_load[machine_id] + option_by_machine[machine_id].processing_time)
            else:
                preferred = _preferred_machine(data, operation_id, policy.mode, weights)
            candidate.machine_assignment[operation_id] = preferred if preferred in alternatives else rng.choice(alternatives)
    return candidate


def _vns(
    individual: Individual,
    data: ProblemData,
    context: Optional[DecodeContext],
    baseline: Optional[Schedule],
    rng: random.Random,
    max_trials: int = 6,
    policy: Optional[_ObjectivePolicy] = None,
    weights: Optional[Sequence[float]] = None,
    evaluator: Optional[Callable[[Individual], Individual]] = None,
) -> Individual:
    """有评估预算上限的目标引导 VNS。"""

    current = individual.clone()
    if policy is None:
        count = len(current.objectives)
        policy = _ObjectivePolicy("multi", tuple(1.0 / count for _ in range(count)), tuple(current.objectives), tuple(current.objectives))
    active_weights = tuple(weights) if weights is not None else policy.weights
    evaluate_candidate = evaluator or (lambda item: _evaluate(item, data, context, baseline))
    neighborhoods = ["insert", "critical_block", "machine", "swap"]
    budget = max(1, max_trials) * len(neighborhoods)
    evaluations = 0
    neighborhood_index = 0
    while neighborhood_index < len(neighborhoods) and evaluations < budget:
        improved = False
        for _ in range(max(1, max_trials)):
            if evaluations >= budget:
                break
            candidate = _propose_neighbor(current, neighborhoods[neighborhood_index], data, policy, active_weights, rng)
            if _individual_signature(candidate, data) == _individual_signature(current, data):
                continue
            evaluate_candidate(candidate)
            evaluations += 1
            if _score_pair(candidate, current, policy, active_weights):
                current = candidate
                improved = True
                break
        neighborhood_index = 0 if improved else neighborhood_index + 1
    return current


def _local_weights(index: int, policy: _ObjectivePolicy, rng: random.Random) -> Tuple[float, ...]:
    if not policy.is_multiobjective:
        return policy.weights
    dimension_count = len(policy.weights)
    if index < dimension_count:
        return tuple(1.0 if dimension == index else 0.05 for dimension in range(dimension_count))
    raw = [rng.random() + 0.05 for _ in range(dimension_count)]
    total = sum(raw)
    return tuple(value / total for value in raw)


def _update_archive(
    archive: Sequence[Individual],
    candidates: Sequence[Individual],
    policy: _ObjectivePolicy,
    archive_size: int,
) -> List[Individual]:
    combined = list(archive) + list(candidates)
    unique: Dict[Tuple[float, ...], Individual] = {}
    for item in combined:
        key = tuple(round(value, 9) for value in item.objectives)
        if key not in unique or policy.score(item) < policy.score(unique[key]):
            unique[key] = item
    values = list(unique.values())
    if not policy.is_multiobjective:
        values.sort(key=lambda item: (policy.score(item), sum(item.objectives)))
        return [item.clone() for item in values[: max(1, archive_size)]]
    front = fast_non_dominated_sort(values)[0] if values else []
    assign_crowding(front)
    front.sort(key=lambda item: item.crowding, reverse=True)
    return [item.clone() for item in front[: max(1, archive_size)]]


def _random_immigrants(data: ProblemData, count: int, rng: random.Random) -> List[Individual]:
    immigrants = []
    for _ in range(max(0, count)):
        sequence = _build_order_sequence(data, "random", rng)
        assignment = _build_machine_assignment(data, "random", rng)
        _perturb_sequence(sequence, rng, 2)
        immigrants.append(Individual(sequence, assignment))
    return immigrants


class IGA_NSGA_VNS:
    """可复现的自适应精英 NSGA-II-VNS 求解器。"""

    def __init__(self, data: ProblemData, config: Optional[AlgorithmConfig] = None):
        self.data = data
        self.config = config or AlgorithmConfig()
        self.rng = random.Random(self.config.seed)
        self.history: List[Dict[str, object]] = []
        self.evaluation_count = 0
        self.cache_hits = 0
        self.restart_count = 0
        self._evaluation_cache: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Tuple[Schedule, Tuple[float, ...]]] = {}

    def _evaluate_cached(
        self,
        individual: Individual,
        context: Optional[DecodeContext],
        dynamic_baseline: Optional[Schedule],
    ) -> Individual:
        individual.machine_assignment = repair_machine_assignment(self.data, individual.machine_assignment)
        signature = _individual_signature(individual, self.data)
        cached = self._evaluation_cache.get(signature)
        if cached is not None:
            schedule, objectives = cached
            individual.schedule = schedule.copy()
            individual.objectives = tuple(objectives)
            self.cache_hits += 1
            return individual
        _evaluate(individual, self.data, context, dynamic_baseline)
        self.evaluation_count += 1
        if individual.schedule is not None:
            self._evaluation_cache[signature] = (individual.schedule.copy(), tuple(individual.objectives))
        return individual

    @staticmethod
    def _reference_point(policy: _ObjectivePolicy) -> Tuple[float, ...]:
        reference = []
        for low, high in zip(policy.ideal, policy.nadir):
            margin = max(1.0, (high - low) * 0.20, abs(high) * 0.05)
            reference.append(high + margin)
        return tuple(reference)

    def run(
        self,
        context: Optional[DecodeContext] = None,
        dynamic_baseline: Optional[Schedule] = None,
    ) -> List[Individual]:
        self.history = []
        self.evaluation_count = 0
        self.cache_hits = 0
        self.restart_count = 0
        self._evaluation_cache = {}
        population = initialize_population(self.data, self.config, self.rng)
        for individual in population:
            self._evaluate_cached(individual, context, dynamic_baseline)
        policy = _ObjectivePolicy.from_population(population, self.config)
        reference_point = self._reference_point(policy)
        population = environmental_selection(population, self.config.population_size, policy, self.data)
        archive = _update_archive([], population, policy, self.config.archive_size)
        best_indicator = -float("inf")
        stagnation = 0

        for generation in range(max(0, self.config.generations)):
            if policy.is_multiobjective:
                fronts = fast_non_dominated_sort(population)
                for front in fronts:
                    assign_crowding(front)
            progress = (self.evaluation_count / max(1, self.config.evaluation_budget)
                        if self.config.evaluation_budget else generation / max(1, self.config.generations - 1))
            pressure = stagnation / max(1, self.config.stagnation_limit)
            mutation_rate = min(0.85, max(0.02, self.config.mutation_rate * (1.15 - 0.35 * progress + 1.50 * pressure)))
            machine_rate = min(0.75, max(0.01, self.config.machine_mutation_rate * (1.05 - 0.25 * progress + pressure)))
            offspring: List[Individual] = []
            known_signatures = {_individual_signature(item, self.data) for item in population}
            while len(offspring) < self.config.population_size:
                parent_a = _tournament(population, self.rng, self.config.tournament_size, policy)
                parent_b = _tournament(population, self.rng, self.config.tournament_size, policy)
                if self.rng.random() < self.config.crossover_rate:
                    child_a, child_b = crossover(parent_a, parent_b, self.data, self.rng)
                else:
                    child_a, child_b = parent_a.clone(), parent_b.clone()
                    child_a.schedule = child_b.schedule = None
                    child_a.objectives = child_b.objectives = tuple()
                for child in (child_a, child_b):
                    mutate(child, self.data, mutation_rate, machine_rate, self.rng, policy.mode, policy.weights)
                    for _ in range(2):
                        signature = _individual_signature(child, self.data)
                        if signature not in known_signatures:
                            break
                        mutate(child, self.data, min(1.0, mutation_rate * 2.0), min(1.0, machine_rate * 2.0), self.rng, policy.mode, policy.weights)
                    known_signatures.add(_individual_signature(child, self.data))
                    self._evaluate_cached(child, context, dynamic_baseline)
                    offspring.append(child)
                    if len(offspring) >= self.config.population_size:
                        break

            combined = population + offspring
            preselected = environmental_selection(combined, self.config.population_size, policy, self.data)
            if policy.is_multiobjective:
                elite_pool = fast_non_dominated_sort(preselected)[0]
                assign_crowding(elite_pool)
                extremes = [min(elite_pool, key=lambda item: item.objectives[index]) for index in range(len(policy.weights))]
                diverse = sorted(elite_pool, key=lambda item: item.crowding, reverse=True)
                elite_pool = []
                elite_signatures: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()
                for item in extremes + diverse:
                    signature = _individual_signature(item, self.data)
                    if signature not in elite_signatures:
                        elite_signatures.add(signature)
                        elite_pool.append(item)
            else:
                elite_pool = sorted(preselected, key=policy.score)
            local_count = (max(1, int(self.config.population_size * self.config.elite_local_search_ratio))
                           if self.config.elite_local_search_ratio > 0 and self.config.vns_probability > 0 else 0)
            local_results: List[Individual] = []
            for index, elite in enumerate(elite_pool[:local_count]):
                # 每代至少精修一个 Cmax 极值解，其余精英按设定概率进入 VNS。
                if index > 0 and self.rng.random() >= self.config.vns_probability:
                    continue
                weights = _local_weights(index, policy, self.rng)
                improved = _vns(
                    elite,
                    self.data,
                    context,
                    dynamic_baseline,
                    self.rng,
                    max_trials=self.config.local_search_trials,
                    policy=policy,
                    weights=weights,
                    evaluator=lambda item: self._evaluate_cached(item, context, dynamic_baseline),
                )
                if _score_pair(improved, elite, policy, weights):
                    local_results.append(improved)
            population = environmental_selection(combined + local_results, self.config.population_size, policy, self.data)
            archive = _update_archive(archive, population + local_results, policy, self.config.archive_size)

            if policy.is_multiobjective:
                indicator = approximate_hypervolume([item.objectives for item in archive], policy.ideal, reference_point)
            else:
                indicator = -min(policy.score(item) for item in archive)
            if indicator > best_indicator + 1e-10:
                best_indicator = indicator
                stagnation = 0
            else:
                stagnation += 1

            if self.config.restart_ratio > 0 and stagnation >= max(1, self.config.stagnation_limit) and generation + 1 < self.config.generations:
                immigrant_count = min(
                    max(1, int(self.config.population_size * max(0.0, min(0.50, self.config.restart_ratio)))),
                    max(1, self.config.population_size - 1),
                )
                immigrants = _random_immigrants(self.data, immigrant_count, self.rng)
                for immigrant in immigrants:
                    self._evaluate_cached(immigrant, context, dynamic_baseline)
                survivors = environmental_selection(population, self.config.population_size - immigrant_count, policy, self.data)
                population = survivors + immigrants
                self.restart_count += 1
                stagnation = 0

            front_for_log = archive if policy.is_multiobjective else [min(archive, key=policy.score)]
            hv = approximate_hypervolume([item.objectives for item in archive], policy.ideal, reference_point)
            total_lookups = self.evaluation_count + self.cache_hits
            self.history.append({
                "generation": float(generation + 1),
                "objective_mode": policy.mode,
                "hypervolume": hv,
                "front_size": float(len(archive) if policy.is_multiobjective else 1),
                "best_makespan": min(item.objectives[0] for item in archive),
                "best_tec": min(item.objectives[1] for item in archive),
                "best_lmax": min(item.objectives[2] for item in archive),
                "best_disturbance": min(item.objectives[3] for item in archive) if len(archive[0].objectives) > 3 else 0.0,
                "best_score": min(policy.score(item) for item in archive),
                "unique_ratio": len({_individual_signature(item, self.data) for item in population}) / max(1.0, float(len(population))),
                "evaluations": float(self.evaluation_count),
                "cache_hit_rate": self.cache_hits / max(1.0, float(total_lookups)),
                "restarts": float(self.restart_count),
            })
            if self.config.verbose and (generation == 0 or (generation + 1) % 10 == 0):
                print(
                    "[IGA-NSGA-II-VNS] generation={} archive={} HV={:.4f} best={:.4f}".format(
                        generation + 1,
                        len(front_for_log),
                        hv,
                        min(policy.score(item) for item in archive),
                    )
                )

        if not archive:
            archive = _update_archive([], population, policy, self.config.archive_size)
        if policy.is_multiobjective:
            return [item.clone() for item in archive]
        return [min(archive, key=policy.score).clone()]
