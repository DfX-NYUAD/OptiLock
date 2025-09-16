from ec.impl.kgs_solution import KGSSolution
from muxlink.muxlink import MuxLink
from utils.bench_parser import BenchParser
from utils.singleton import Singleton


@Singleton
class MuxLinkBaseScope:

    def __init__(self):
        self.loading_done = False
        self.target = None
        self.id_map = {}
        self.layer_map = {}
        self.muxlink = None
        self.initialized = False
        self.netlist = None
        self.netlist_str = ""
        self.h_hop = None
        self.chosen_sol = None
        self.key_size = None

    def get_FMFS(self, solution):
        # return FMulti and FSingle for given solution ID
        # TODO for later
        return self.muxlink.get_FMulti_FSingle(solution.get_id())

    def load(self, target_file_path, locked_file_path, h_hop, chosen_sol, train_mark, epochs=5):
        # check if it is locked
        lock_mark = False
        key_string = ""
        with open(locked_file_path, 'r') as f:
            for line in f.readlines():
                if "#key" in line:
                    lock_mark = True
                    key_string = line
                    break
        # load netlist string and store it locally
        self.target = target_file_path.split("/")[-1].split(".bench")[0].split("_")[0]
        self.h_hop = h_hop
        self.chosen_sol = chosen_sol
        self.netlist_str = ""
        netlist_str_locked = ""
        with open(target_file_path, 'r') as f:
            self.netlist_str = f.read()
        # self.netlist = BenchParser.instance().parse_file(target_file_path)
        # self.netlist_str = self.netlist.to_string()
        # # get the locked netlist string
        # netlist_locked = BenchParser.instance().parse_file(locked_file_path)
        # netlist_str_locked = netlist_locked.to_string()
        with open(locked_file_path, 'r') as f:
            netlist_str_locked = f.read()
        # create muxlink object and train it
        self.muxlink = MuxLink(self.target)
        # current not training before testing
        # self.muxlink.train(self.netlist_str, key=2, epochs=epochs)
        if train_mark == True:
            print("why I am here")
            if lock_mark == False:
                self.muxlink.train(self.netlist_str, lock_mark, key=1, epochs=epochs)
            else:
                # locked_netlist_str = key_string + self.netlist_str
                # read the target file as a string to locked_netlist_str
                # locked_netlist_str = ""
                # with open(target_file_path, 'r') as f:
                #     for line in f.readlines():
                #         locked_netlist_str += line
                locked_netlist_str = key_string + netlist_str_locked
                self.muxlink.train(locked_netlist_str, lock_mark, key=0, epochs=epochs)
        # mark init done
        self.initialized = True
    def get_target(self):
        return self.target
    def decode(self, kgss: KGSSolution):
        # self.muxlink = MuxLink()
        return self.muxlink.decode_scope(self.netlist_str, kgss)

    def update(self, netlist_str, new_kgss: KGSSolution):
        # self.muxlink = MuxLink()
        # decode the kgss
        locked_bench_str = self.muxlink.decode(netlist_str, new_kgss)
        # print(locked_bench_str)
        # [x1, y1, z1], [x2, y2, z2]
        # [x1, y2, z2], [x2, y1, z2]
        # changes: merge graph directly 1) removing some 2) adding new
        # encode the kgss data list
        # print(self.muxlink.encode(locked_bench_str).data)
        return self.muxlink.encode(locked_bench_str)

    def attack(self, solution: KGSSolution):
        if not self.initialized:
            raise ValueError("MuxLinkBase must be first initialized before the attack is executed.")
        # self.muxlink = MuxLink()
        locked_bench_str = self.decode(solution)
        # the check.txt file seems to be a debug file
        text_file = open("../ml_data_" + self.target + "/check_"+ self.target + ".txt", "w")
        text_file.write(locked_bench_str)
        text_file.close()
        print("what is my target name", self.target)
        return self.muxlink.attack(locked_bench_str, self.target, self.h_hop)
    
    # this function is used to attack the solution to get the exact result 
    def attack_exact(self, solution: KGSSolution):
        if not self.initialized:
            raise ValueError("MuxLinkBase must be first initialized before the attack is executed.")
        # self.muxlink = MuxLink()
        locked_bench_str = self.decode(solution)
        # the check.txt file seems to be a debug file
        text_file = open("../ml_data_" + self.target + "/check_"+ self.target + ".txt", "w")
        text_file.write(locked_bench_str)
        text_file.close()
        print("what is my target name", self.target)
        return self.muxlink.attack_exact(locked_bench_str, self.target, self.h_hop)

    # this function is used to load the train links for attacking
    def attack_get_train_links(self, solution: KGSSolution):
        if not self.initialized:
            raise ValueError("MuxLinkBase must be first initialized before the attack is executed.")
        # self.muxlink = MuxLink()
        locked_bench_str = self.decode(solution)
        # here we form the check txt for the later D_MUX perl script
        text_file = open("../ml_data_" + self.target + "/check_"+ self.target + ".txt", "w")
        text_file.write(locked_bench_str)
        text_file.close()
        # get the key size 
        self.key_size = solution.get_key_size()
        # load the train links in the current folder
        self.muxlink.attack_get_train_links(locked_bench_str)

    # this function is used to get the test link prediction from attacking
    def attack_get_test_links(self, sol_index):
        self.muxlink.attack_get_test_links(self.chosen_sol, sol_index, self.h_hop)
        # each running will add the fitness_log.txt
        text_file = open("../ml_data_" + self.target + "/fitness_log_" + str(sol_index) + ".txt", "w")
        text_file.write("fitness done!")
        text_file.close()
    
    # this function is used to collect all the prediction results;
    # and then we can use them to calculate the accuracy by D-MUX perl script
    def attack_merge_results(self, key_size):
        # in the main func, we need to check if the fitness_log.txt is exist before collect them
        return self.muxlink.attack_merge_results(self.target, self.h_hop, key_size)


    def attack_thread(self, file_num, solution: KGSSolution):
        if not self.initialized:
            raise ValueError("MuxLinkBase must be first initialized before the attack is executed.")
        # self.muxlink = MuxLink()
        # file_num = self.threadnum
        locked_bench_str = self.decode(solution)
        # the check.txt file seems to be a debug file
        text_file = open("../ml_data_" + self.target + "/check_"+ self.target + "_" + str(file_num) +".txt", "w")
        text_file.write(locked_bench_str)
        text_file.close()
        return self.muxlink.attack_thread(locked_bench_str, self.target, file_num)

    def lock(self, netlist_str, key_size, alg_type):
        self.muxlink = MuxLink()
        return self.muxlink.lock(netlist_str, key_size, alg_type)

    def get_max_id(self, netlist_str):
        # TODO zeng - get the maximum gate id
        # self.muxlink = MuxLink()
        return self.muxlink.get_max_id(netlist_str)

    # def load(self, target):
    #     if self.loading_done:
    #         return
    #
    #     self.extract_gate_info(target)
    #     self.loading_done = True

    # def register_id_entry(self, index, gate):
    #     id = gate.output
    # F
    #     if gate.output in self.id_map:
    #         self.id_map[id].add(index)
    #     else:
    #         vec = {index}
    #         self.id_map[id] = vec
    #
    # def register_layer_entry(self, index, gate):
    #     id = gate.output
    #
    #     if index in self.layer_map:
    #         self.layer_map[index].add(id)
    #     else:
    #         vec = {id}
    #         self.layer_map[index] = vec
    #
    # def extract_gate_info(self, target):
    #     front = set(target.primary_output_nodes)
    #
    #     index = 0
    #     while len(front) > 0:
    #         next_front = set()
    #
    #         for gate in front:
    #             for input in gate.get_inputs():
    #                 if input not in target.node_map:
    #                     input_gate = target.inputs_map[input]
    #                 else:
    #                     input_gate = target.node_map[input]
    #
    #                 self.register_layer_entry(index, input_gate)  # 2 -> [x]
    #                 self.register_id_entry(index, input_gate)  # x -> [2,3]
    #                 next_front.add(input_gate)
    #
    #         index += 1
    #         front = next_front
    #
    # def next_level(self, gate, next_front):
    #     for input in gate.get_inputs():
    #         next_front.add(input)
