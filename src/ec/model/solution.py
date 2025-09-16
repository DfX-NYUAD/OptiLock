from abc import ABC, abstractmethod
from ec.model.fitness import Fitness


class Solution(ABC):

    def __init__(self, id):
        self.fitness = None
        self.id = id

    def get_fitness(self) -> Fitness:
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

    def is_better_than(self, other_solution):
        return self.fitness.is_better_then(other_solution.get_fitness())

    def get_id(self):
        return self.id

    @abstractmethod
    def to_string(self):
        pass
