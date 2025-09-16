from ec.impl.kgs_ops.muxlink_base_large import MuxLinkBaseLarge
from ec.impl.kgs_solution import KGSSolution
from ec.model.fitness import Fitness, FitnessType
from ec.model.fitness_function import FitnessFunction


class MuxLinkFitnessFunctionLarge(FitnessFunction):

    def __init__(self, target_file_path, locked_file_path, h_hop, chosen_sol, train_mark, epochs=100):
        # print("what is my target file path", target_file_path)
        MuxLinkBaseLarge.instance().load(target_file_path, locked_file_path, h_hop, chosen_sol, train_mark, epochs)

    def evaluate(self, solutions: [KGSSolution]):

        kpa_list = []
        for solution in solutions:
            # evaluate (attack) kgss
            # get the target name
            target_name = MuxLinkBaseLarge.instance().get_target()
            print("what is my target name2", target_name, flush=True)
            acc, prec, kpa = MuxLinkBaseLarge.instance().attack(solution)

            # create new fitness object
            # fitness = Fitness(type=FitnessType.MIN, value=kpa)
            # kpa_list.append(kpa)
            fitness = Fitness(type=FitnessType.MIN, value=acc)
            kpa_list.append([acc, prec, kpa])
            # assign fitness
            solution.set_fitness(fitness)
        print("what is my fitness value")
        print(kpa_list)
        return kpa_list
    
    def evaluate_exact(self, solutions: [KGSSolution]):

        kpa_list = []
        for solution in solutions:
            # evaluate (attack) kgss
            # get the target name
            target_name = MuxLinkBaseLarge.instance().get_target()
            print("what is my target name2", target_name)
            kpa, uni_list, wrong_list = MuxLinkBaseLarge.instance().attack_exact(solution)

            # create new fitness object
            fitness = Fitness(type=FitnessType.MIN, value=kpa)
            kpa_list.append(kpa)
            kpa_list.append(uni_list)
            kpa_list.append(wrong_list)
            # assign fitness
            solution.set_fitness(fitness)
        print("what is my fitness value")
        print(kpa_list)
        return kpa_list

    # added the thread_evaluate function
    def evaluate_thread(self, file_num,solutions: [KGSSolution]):
        kpa_list = []
        for solution in solutions:
            # evaluate (attack) kgss
            acc, prec, kpa = MuxLinkBaseLarge.instance().attack_thread(file_num, solution)

            # create new fitness object
            fitness = Fitness(type=FitnessType.MIN, value=kpa)
            kpa_list.append(kpa)
            # assign fitness
            solution.set_fitness(fitness)
        print("what is my fitness value")
        print(kpa_list)
        return kpa_list
