import random
from itertools import combinations, product
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

from data import (
    Product, TARGETS, MIN_COST, MAX_COST,
    K, MAX_GRAMS, STEP,
    POP_SIZE, GENERATIONS, CROSSOVER_RATE, MUTATION_RATE,
    TOURNAMENT_SIZE, COST_PENALTY_WEIGHT,
)


class DietProblem:
    def __init__(self, products: List[Product], targets: np.ndarray,
                 min_cost: float, max_cost: float, k: int):
        self.products = products
        self.targets = targets
        self.min_cost = min_cost
        self.max_cost = max_cost
        self.k = k
        self.N = len(products)
        self.M = len(targets)
        self.prices = np.array([p.price for p in products], dtype=float)
        self.nutrients = np.array([p.nutrients for p in products], dtype=float).T

    def decode(self, chrom: np.ndarray) -> Tuple[float, float, np.ndarray]:
        grams = chrom / 100.0
        totals = self.nutrients @ grams
        cost = float(self.prices @ grams)
        deviation = float(np.sum((totals - self.targets) ** 2))
        return cost, deviation, totals

    def is_cost_in_range(self, cost: float) -> bool:
        return self.min_cost <= cost <= self.max_cost


class Chromosome:
    def __init__(self, genes: np.ndarray):
        self.genes = genes

    @classmethod
    def random(cls, problem: DietProblem,
               max_grams: int = MAX_GRAMS, step: int = STEP) -> "Chromosome":
        genes = np.zeros(problem.N, dtype=int)
        idx = random.sample(range(problem.N), problem.k)
        for i in idx:
            genes[i] = random.randint(1, max_grams // step) * step
        return cls(genes)

    @classmethod
    def from_genes(cls, genes: np.ndarray) -> "Chromosome":
        return cls(genes.copy())

    def repair(self, problem: DietProblem,
               max_grams: int = MAX_GRAMS, step: int = STEP) -> "Chromosome":
        genes = self.genes.copy()
        nonzero = np.where(genes > 0)[0]
        if len(nonzero) > problem.k:
            keep = set(random.sample(list(nonzero), problem.k))
            for i in nonzero:
                if i not in keep:
                    genes[i] = 0
        elif len(nonzero) < problem.k:
            zero = np.where(genes == 0)[0]
            add = random.sample(list(zero), problem.k - len(nonzero))
            for i in add:
                genes[i] = random.randint(1, max_grams // step) * step
        return Chromosome(genes)

    def copy(self) -> "Chromosome":
        return Chromosome(self.genes.copy())

    def nonzero_indices(self) -> np.ndarray:
        return np.where(self.genes > 0)[0]

    def zero_indices(self) -> np.ndarray:
        return np.where(self.genes == 0)[0]

    def __len__(self) -> int:
        return len(self.genes)


class FitnessFunction:
    def __init__(self, problem: DietProblem,
                 cost_penalty_weight: float = COST_PENALTY_WEIGHT):
        self.problem = problem
        self.W = cost_penalty_weight

    def penalty(self, cost: float) -> float:
        lo, hi = self.problem.min_cost, self.problem.max_cost
        if cost < lo:
            return ((lo - cost) ** 2) * self.W
        if cost > hi:
            return ((cost - hi) ** 2) * self.W
        return 0.0

    def evaluate(self, chrom: Chromosome) -> Tuple[float, float, float, np.ndarray]:
        cost, deviation, totals = self.problem.decode(chrom.genes)
        pen = self.penalty(cost)
        fitness = 1.0 / (deviation + pen + 1e-6)
        return fitness, deviation, cost, totals


class SelectionOperator:
    def __init__(self, tournament_size: int = TOURNAMENT_SIZE):
        self.tournament_size = tournament_size

    def select(self, population: List[Chromosome],
               fitnesses: List[float]) -> Chromosome:
        chosen = random.sample(range(len(population)), self.tournament_size)
        best = max(chosen, key=lambda i: fitnesses[i])
        return population[best].copy()


class CrossoverOperator:
    def __init__(self, problem: DietProblem,
                 max_grams: int = MAX_GRAMS, step: int = STEP):
        self.problem = problem
        self.max_grams = max_grams
        self.step = step

    def _repair(self, genes: np.ndarray) -> Chromosome:
        return Chromosome(genes).repair(self.problem, self.max_grams, self.step)

    def cross(self, p1: Chromosome, p2: Chromosome) -> Chromosome:
        raise NotImplementedError


class UniformCrossover(CrossoverOperator):
    def cross(self, p1: Chromosome, p2: Chromosome) -> Chromosome:
        mask = np.random.rand(len(p1)) < 0.5
        genes = np.where(mask, p1.genes, p2.genes)
        return self._repair(genes)


class SinglePointCrossover(CrossoverOperator):
    def cross(self, p1: Chromosome, p2: Chromosome) -> Chromosome:
        point = random.randint(1, len(p1) - 1)
        genes = np.concatenate([p1.genes[:point], p2.genes[point:]])
        return self._repair(genes)


class ArithmeticCrossover(CrossoverOperator):
    def cross(self, p1: Chromosome, p2: Chromosome) -> Chromosome:
        alpha = random.random()
        genes = (alpha * p1.genes + (1 - alpha) * p2.genes).astype(int)
        genes = (genes // self.step) * self.step
        return self._repair(genes)


class MutationOperator:
    def __init__(self, problem: DietProblem,
                 max_grams: int = MAX_GRAMS, step: int = STEP):
        self.problem = problem
        self.max_grams = max_grams
        self.step = step

    def mutate(self, chrom: Chromosome) -> Chromosome:
        raise NotImplementedError

    def _repair(self, genes: np.ndarray) -> Chromosome:
        return Chromosome(genes).repair(self.problem, self.max_grams, self.step)


class GaussMutation(MutationOperator):
    def mutate(self, chrom: Chromosome) -> Chromosome:
        nz = chrom.nonzero_indices()
        if len(nz) == 0:
            return chrom.copy()
        genes = chrom.genes.copy()
        i = random.choice(nz)
        delta = int(np.random.normal(0, 50))
        new_val = max(self.step, min(self.max_grams, genes[i] + delta))
        genes[i] = (new_val // self.step) * self.step
        return Chromosome(genes)


class SwapMutation(MutationOperator):
    def mutate(self, chrom: Chromosome) -> Chromosome:
        nz, z = chrom.nonzero_indices(), chrom.zero_indices()
        if len(nz) == 0 or len(z) == 0:
            return chrom.copy()
        genes = chrom.genes.copy()
        i, j = random.choice(nz), random.choice(z)
        genes[j] = genes[i]
        genes[i] = 0
        return self._repair(genes)


class ResetMutation(MutationOperator):
    def mutate(self, chrom: Chromosome) -> Chromosome:
        nz, z = chrom.nonzero_indices(), chrom.zero_indices()
        if len(nz) == 0 or len(z) == 0:
            return chrom.copy()
        genes = chrom.genes.copy()
        i, j = random.choice(nz), random.choice(z)
        genes[i] = 0
        genes[j] = random.randint(1, self.max_grams // self.step) * self.step
        return self._repair(genes)


class GeneticAlgorithm:
    def __init__(self, problem: DietProblem, fitness_fn: FitnessFunction,
                 selection: SelectionOperator, crossover: CrossoverOperator,
                 mutation: MutationOperator,
                 pop_size: int = POP_SIZE, generations: int = GENERATIONS,
                 crossover_rate: float = CROSSOVER_RATE,
                 mutation_rate: float = MUTATION_RATE,
                 max_grams: int = MAX_GRAMS, step: int = STEP):
        self.problem = problem
        self.fitness_fn = fitness_fn
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_grams = max_grams
        self.step = step

    def _initial_population(self) -> List[Chromosome]:
        return [Chromosome.random(self.problem, self.max_grams, self.step)
                for _ in range(self.pop_size)]

    def run(self) -> Tuple[Chromosome, List[float]]:
        population = self._initial_population()
        history: List[float] = []
        best_individual = population[0].copy()
        best_fitness = -1.0

        for _ in range(self.generations):
            fitnesses = [self.fitness_fn.evaluate(ind)[0] for ind in population]

            elite_idx = int(np.argmax(fitnesses))
            if fitnesses[elite_idx] > best_fitness:
                best_fitness = fitnesses[elite_idx]
                best_individual = population[elite_idx].copy()
            history.append(best_fitness)

            new_population = [population[elite_idx].copy()]
            while len(new_population) < self.pop_size:
                p1 = self.selection.select(population, fitnesses)
                p2 = self.selection.select(population, fitnesses)

                if random.random() < self.crossover_rate:
                    child = self.crossover.cross(p1, p2)
                else:
                    child = p1.copy()

                if random.random() < self.mutation_rate:
                    child = self.mutation.mutate(child)

                new_population.append(child)

            population = new_population

        return best_individual, history


class BruteForceSolver:
    def __init__(self, problem: DietProblem, fitness_fn: FitnessFunction,
                 max_grams: int = MAX_GRAMS, step: int = STEP):
        self.problem = problem
        self.fitness_fn = fitness_fn
        self.max_grams = max_grams
        self.step = step

    def solve(self, n_small: int = 10, k_small: int = 3,
              step_override: int = None):
        step = step_override if step_override is not None else self.step
        grams_options = list(range(step, self.max_grams + 1, step))

        best = None
        best_dev = float("inf")
        checked = 0

        for combo in combinations(range(n_small), k_small):
            for grams in product(grams_options, repeat=k_small):
                genes = np.zeros(self.problem.N, dtype=int)
                for idx, g in zip(combo, grams):
                    genes[idx] = g
                chrom = Chromosome.from_genes(genes)
                _, dev, cost, totals = self.fitness_fn.evaluate(chrom)

                if self.problem.is_cost_in_range(cost) and dev < best_dev:
                    best_dev = dev
                    best = (combo, grams, dev, cost, totals)
                checked += 1

        print(f"Полный перебор: проверено {checked} вариантов "
              f"(шаг {step} г, n={n_small}, k={k_small})")
        return best


class ExperimentRunner:
    def __init__(self, problem: DietProblem, fitness_fn: FitnessFunction,
                 max_grams: int = MAX_GRAMS, step: int = STEP,
                 pop_size: int = POP_SIZE, generations: int = GENERATIONS):
        self.problem = problem
        self.fitness_fn = fitness_fn
        self.max_grams = max_grams
        self.step = step
        self.pop_size = pop_size
        self.generations = generations

    def _make_crossovers(self) -> Dict[str, CrossoverOperator]:
        return {
            "Uniform":     UniformCrossover(self.problem, self.max_grams, self.step),
            "SinglePoint": SinglePointCrossover(self.problem, self.max_grams, self.step),
            "Arithmetic":  ArithmeticCrossover(self.problem, self.max_grams, self.step),
        }

    def _make_mutations(self) -> Dict[str, MutationOperator]:
        return {
            "Gauss": GaussMutation(self.problem, self.max_grams, self.step),
            "Swap":  SwapMutation(self.problem, self.max_grams, self.step),
            "Reset": ResetMutation(self.problem, self.max_grams, self.step),
        }

    def _run_one(self, crossover: CrossoverOperator,
                 mutation: MutationOperator) -> Tuple[Chromosome, List[float]]:
        ga = GeneticAlgorithm(
            problem=self.problem,
            fitness_fn=self.fitness_fn,
            selection=SelectionOperator(TOURNAMENT_SIZE),
            crossover=crossover,
            mutation=mutation,
            pop_size=self.pop_size,
            generations=self.generations,
            max_grams=self.max_grams,
            step=self.step,
        )
        return ga.run()

    def run_all(self) -> Dict[Tuple[str, str], List[float]]:
        crossovers = self._make_crossovers()
        mutations = self._make_mutations()
        results: Dict[Tuple[str, str], List[float]] = {}

        for cname, cfn in crossovers.items():
            for mname, mfn in mutations.items():
                print(f"\n=== Кроссовер: {cname} | Мутация: {mname} ===")
                best_ind, history = self._run_one(cfn, mfn)
                _, dev, cost, totals = self.fitness_fn.evaluate(best_ind)
                print(f"Отклонение: {dev:.2f}")
                print(f"Стоимость: {cost:.2f} руб. "
                      f"(в рамках: {self.problem.is_cost_in_range(cost)})")
                print(f"Характеристики: {np.round(totals, 1)}")
                print(f"Продукты: "
                      f"{[self.problem.products[i].name for i in best_ind.nonzero_indices()]}")
                print(f"Граммовка: "
                      f"{[int(best_ind.genes[i]) for i in best_ind.nonzero_indices()]}")
                results[(cname, mname)] = history

        return results

    def plot(self, results: Dict[Tuple[str, str], List[float]],
             path_fitness: str = "ga_variant3_comparison.png",
             path_deviation: str = "ga_variant3_deviation.png"):
        plt.figure(figsize=(13, 8))
        for (cname, mname), hist in results.items():
            plt.plot(hist, label=f"{cname} + {mname}")
        plt.xlabel("Поколение")
        plt.ylabel("Лучшая приспособленность")
        plt.title("Вариант 3: сравнение операторов кроссовера и мутации")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path_fitness, dpi=120)
        plt.show()

        plt.figure(figsize=(13, 8))
        for (cname, mname), hist in results.items():
            devs = [1.0 / f for f in hist]
            plt.plot(devs, label=f"{cname} + {mname}")
        plt.xlabel("Поколение")
        plt.ylabel("Отклонение (обратное к fitness)")
        plt.title("Вариант 3: динамика отклонения от нормы")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path_deviation, dpi=120)
        plt.show()