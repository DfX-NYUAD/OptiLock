from enum import Enum


class FitnessType(Enum):
    MAX = 0,
    MIN = 1


class Fitness:

    def __init__(self, type, value):
        self.type = type
        self.value = value

    def is_better_then(self, other_fitness):
        if self.type == FitnessType.MAX:
            return self.value > other_fitness.value
        else:
            return self.value < other_fitness.value
