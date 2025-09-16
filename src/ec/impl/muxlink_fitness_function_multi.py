from ec.impl.kgs_ops.muxlink_base import MuxLinkBase
from ec.impl.kgs_solution import KGSSolution
from ec.model.fitness import Fitness, FitnessType
from ec.model.fitness_function import FitnessFunction


class MuxLinkFitnessFunctionMulti(FitnessFunction):

    def __init__(self, target_file_path, locked_file_path, h_hop, kgss_data_index, train_mark, start_num, epochs=100):
        # print("what is my target file path", target_file_path)
        self.configurations = []
        # instances = []
        for i in range(start_num, start_num+5):
            circuit_name = target_file_path.split('/')[-1].split('.')[0]
            target_file_path_temp = "../data/original/" +circuit_name + str(i) + ".bench"
            if "b" in circuit_name:
                target_file_path_temp = "../data/original/" +circuit_name + str(i) + "_C.bench"
            config = (target_file_path_temp, locked_file_path, h_hop, kgss_data_index, train_mark, epochs)
            self.configurations.append(config)
        

    def evaluate(self, solutions: [KGSSolution]):

        kpa_list = []
        for solution in solutions:
            # evaluate (attack) kgss
            # get the target name
            # target_name = MuxLinkBase.instance().get_target()
            # print("what is my target name2", target_name)
            kpa_result_list = []
            # here we should run them in parallel

            for config in self.configurations:
                mux_instance = MuxLinkBase.instance()
                mux_instance.load(config[0], config[1], config[2], config[3], config[4], config[5])
                acc, prec, kpa_temp = mux_instance.attack(solution)
                kpa_result_list.append(float(kpa_temp))
            kpa = sum(kpa_result_list)/len(kpa_result_list)


            # create new fitness object
            fitness = Fitness(type=FitnessType.MIN, value=kpa)
            kpa_list.append(kpa)
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
            acc, prec, kpa = MuxLinkBase.instance().attack_thread(file_num, solution)

            # create new fitness object
            fitness = Fitness(type=FitnessType.MIN, value=kpa)
            kpa_list.append(kpa)
            # assign fitness
            solution.set_fitness(fitness)
        print("what is my fitness value")
        print(kpa_list)
        return kpa_list
