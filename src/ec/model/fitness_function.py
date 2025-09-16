from abc import ABC, abstractmethod


class FitnessFunction(ABC):

    def evaluate_population(self, population):
        self.evaluate(population.solutions)

    @abstractmethod
    def evaluate(self, solutions):
        pass
