from ec.impl.kgs_ops.muxlink_base import MuxLinkBase
from ec.impl.kgs_solution import KGSSolution
from ec.model.fitness import Fitness, FitnessType
from ec.model.fitness_function import FitnessFunction


class MuxLinkFitnessFunctionPlus(FitnessFunction):

    def __init__(self, target_file_path, locked_file_path, h_hop, chosen_sol, train_mark, epochs=100):
        # print("what is my target file path", target_file_path)
        MuxLinkBase.instance().load(target_file_path, locked_file_path, h_hop, chosen_sol, train_mark, epochs)

    def attack_load(self, solutions: [KGSSolution]):
        # here I lode the train links for the attacking 
        # and then we do not need to run for this part all the times
        for solution in solutions:
            MuxLinkBase.instance().attack_get_train_links(solution)
    
    def attack_test(self, sol_index):
        MuxLinkBase.instance().attack_get_test_links(sol_index)
    
    def evaluate(self, solutions: [KGSSolution]):

        kpa_list = []
        
        for solution in solutions:
            # evaluate (attack) kgss
            key_size = solution.get_key_size()
            acc, prec, kpa = MuxLinkBase.instance().attack_merge_results(key_size)

            # # create new fitness object
            # fitness = Fitness(type=FitnessType.MIN, value=kpa)
            # kpa_list.append(kpa)
            # create new fitness object
            fitness = Fitness(type=FitnessType.MIN, value=acc)
            kpa_list.append(acc)
            # assign fitness
            solution.set_fitness(fitness)
        print("what is my fitness value")
        print(kpa_list)
        return kpa_list
    
