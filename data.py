import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Product:
    name: str
    price: float
    kcal: float
    protein: float
    fat: float
    carbs: float

    @property
    def nutrients(self) -> Tuple[float, float, float, float]:
        return self.kcal, self.protein, self.fat, self.carbs


PRODUCTS: List[Product] = [
    Product("Гречка",          8.0,  343, 12.6,  3.3, 62.1),
    Product("Рис",             9.0,  344,  6.7,  0.7, 78.9),
    Product("Макароны",        7.0,  350, 10.4,  1.1, 69.7),
    Product("Хлеб ржаной",     4.5,  210,  6.6,  1.2, 40.7),
    Product("Хлеб пшеничный",  5.0,  250,  8.1,  1.0, 48.8),
    Product("Картофель",       3.0,   77,  2.0,  0.4, 16.3),
    Product("Морковь",         4.0,   35,  1.3,  0.1,  6.9),
    Product("Капуста",         2.5,   28,  1.8,  0.1,  4.7),
    Product("Лук",             3.5,   41,  1.4,  0.2,  8.2),
    Product("Помидор",         8.0,   20,  1.1,  0.2,  3.7),
    Product("Огурец",          6.0,   15,  0.8,  0.1,  2.8),
    Product("Яблоко",          7.0,   52,  0.4,  0.4, 13.8),
    Product("Банан",          10.0,   96,  1.5,  0.2, 21.0),
    Product("Апельсин",        9.0,   47,  0.9,  0.2, 11.8),
    Product("Курица",         25.0,  165, 31.0,  3.6,  0.0),
    Product("Говядина",       45.0,  187, 18.9, 12.4,  0.0),
    Product("Свинина",        35.0,  263, 16.0, 21.6,  0.0),
    Product("Рыба (минтай)",  18.0,   72, 16.0,  0.9,  0.0),
    Product("Яйцо",            8.0,  157, 12.7, 11.5,  0.7),
    Product("Молоко",          6.0,   64,  3.2,  3.6,  4.8),
    Product("Кефир",           5.5,   51,  3.0,  2.5,  4.0),
    Product("Творог",         22.0,  121, 17.0,  5.0,  3.0),
    Product("Сыр",            40.0,  364, 25.0, 29.0,  0.0),
    Product("Масло слив.",    80.0,  748,  0.5, 82.5,  0.8),
    Product("Масло раст.",    12.0,  899,  0.0, 99.9,  0.0),
    Product("Сахар",           5.0,  399,  0.0,  0.0, 99.8),
    Product("Соль",            1.0,    0,  0.0,  0.0,  0.0),
    Product("Фасоль",         12.0,  298, 21.0,  2.0, 54.0),
    Product("Горох",          10.0,  298, 20.5,  2.0, 49.5),
    Product("Овсянка",         6.0,  366, 12.3,  6.1, 59.5),
]

TARGETS = np.array([2200.0, 85.0, 75.0, 350.0])

MIN_COST = 80.0
MAX_COST = 150.0

K = 5
MAX_GRAMS = 500
STEP = 10

POP_SIZE = 100
GENERATIONS = 200
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.15
TOURNAMENT_SIZE = 3
COST_PENALTY_WEIGHT = 100.0