import numpy as np

from data import (
    PRODUCTS, TARGETS, MIN_COST, MAX_COST,
    K, MAX_GRAMS, STEP, POP_SIZE, GENERATIONS,
)
from ga import DietProblem, FitnessFunction, BruteForceSolver, ExperimentRunner


def main():
    problem = DietProblem(
        products=PRODUCTS,
        targets=TARGETS,
        min_cost=MIN_COST,
        max_cost=MAX_COST,
        k=K,
    )
    fitness_fn = FitnessFunction(problem)

    print("Вариант 3: лучший по характеристикам рацион в ценовых рамках")
    print(f"N = {problem.N}, m = {problem.M}, k = {problem.k}")
    print(f"Целевые значения: {TARGETS}")
    print(f"Ценовые рамки: [{MIN_COST}; {MAX_COST}] руб.")

    print("\n--- Полный перебор ---")
    bf = BruteForceSolver(problem, fitness_fn, MAX_GRAMS, STEP)
    result = bf.solve(n_small=10, k_small=3, step_override=50)
    if result:
        combo, grams, dev, cost, totals = result
        print(f"Лучший рацион: {[PRODUCTS[i].name for i in combo]}")
        print(f"Граммовка: {grams}")
        print(f"Отклонение: {dev:.2f}, стоимость: {cost:.2f}")
        print(f"Характеристики: {np.round(totals, 1)}")
    else:
        print("Решение в ценовых рамках не найдено.")

    print("\n--- Генетический алгоритм ---")
    runner = ExperimentRunner(problem, fitness_fn, MAX_GRAMS, STEP,
                              POP_SIZE, GENERATIONS)
    results = runner.run_all()
    runner.plot(results)


if __name__ == "__main__":
    main()