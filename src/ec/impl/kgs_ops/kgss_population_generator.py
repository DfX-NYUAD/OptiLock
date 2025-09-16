import random
from ec.model.population import Population
from ec.model.population_generator import PopulationGenerator
from ec.impl.kgs_ops.muxlink_base import MuxLinkBase
from muxlink.muxlink import MuxLink


class KGSSPopulationGenerator(PopulationGenerator):

    def __init__(self, population_size, key_size, alg_type, netlist_str):
        self.population_size = population_size
        # TODO zeng: add config params such as key size and alg types,...
        self.key_size = key_size # here we make default value to be 64 or None?
        self.alg_type = alg_type # D-MUX -> it looks as two mux type[eD-MUX/ D-MUX]
        # self.solution_id = None # which is used for muxlink
        self.netlist = None
        self.netlist_str = netlist_str


    def execute(self):
        population = Population()
        # kgs_list = []
        print("what is my population")
        for i in range(self.population_size):
            # TODO zeng
            # create new kgss
            self.muxlink = MuxLink()
            locked_bench_str = MuxLinkBase.instance().lock(self.netlist_str, self.key_size, self.alg_type)
            kgss = self.muxlink.encode(locked_bench_str)
            print(kgss.data)
            # kgs_list.append(kgss)
            # append each kgss solution to the code
            population.register_solution(kgss)

        return population

