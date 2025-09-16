from ec.impl.kgs_ops.muxlink_base import MuxLinkBase
from ec.impl.kgs_solution import KGSSolution
from ec.model.mutation_operator import MutationOperator
import random
import copy

class MuxNodeMutation(MutationOperator):

    def __init__(self, netlist_str, node_mutation_probability=0.05):
        self.node_mutation_probability = node_mutation_probability
        self.netlist_str = netlist_str

    def ExtractSingleMultiOutputNodes(self, G) -> list:
        F_multi = []
        F_single = []
        for n in G.nodes():
            out_degree = G.out_degree(n)
            check = G.nodes[n]['output']
            if out_degree == 1:
                if G.nodes[n]['gate'] != "input" and not check:
                    F_single.append(n)
            elif out_degree > 1:
                if G.nodes[n]['gate'] != "input" and not check:
                    F_multi.append(n)
        return F_multi + F_single

    def generate_child_permute(self, solution:KGSSolution):
        parent1 = solution
        # here we get the g1 and g2, but we need to figure out which g we could change
        # also we also need to figure out which f we could change
        # for each encoding item, [f1, g1] and [f2, g2] are two locking pair we have
        # therefore, we could record them and then permute them
        my_pair_p1 = {}
        parent1_data = parent1.data
        for index in range(len(parent1_data)):
            temp_list = parent1_data[index]
            if int(temp_list[4]) == 0:  # based on the key value, the key pair should be changed
                my_pair_p1[temp_list[2]] = temp_list[0]
                my_pair_p1[temp_list[3]] = temp_list[1]
            else:
                my_pair_p1[temp_list[2]] = temp_list[1]
                my_pair_p1[temp_list[3]] = temp_list[0]
        # because the key value is not too important to the encoding list
        # it is secure to just random shuffle the two pair
        p1_pair_keys = list(my_pair_p1.keys())
        # in order to avoid there is path between [f1, g2] and [f2, g1]
        # here we loop to random choose the keys of the dict
        # and check if the [key1, key2, value1, value2] have some value repeated.
        permuted_p1_list = []
        # here is for parent1
        for index in range(len(parent1_data)):
            p1_flag = True
            while p1_flag:
                key_temp = random.sample(p1_pair_keys, 2)  # random choose 2
                value_temp = [my_pair_p1.get(key) for key in key_temp]
                if not set(key_temp).intersection(set(value_temp)):
                    p1_flag = False
                    if int(parent1_data[index][4]) == 0:
                        temp_list = [value_temp[0], value_temp[1], key_temp[0], key_temp[1], parent1_data[index][4],
                                     parent1_data[index][5]]
                    else:
                        temp_list = [value_temp[0], value_temp[1], key_temp[1], key_temp[0], parent1_data[index][4],
                                     parent1_data[index][5]]
                    permuted_p1_list.append(temp_list)
                    for item in key_temp:  # avoid repeated keys
                        p1_pair_keys.remove(item)
                else:
                    p1_flag = True  # continue searching
        # print(permuted_p1_list)
        child1 = KGSSolution()
        for index in range(len(permuted_p1_list)):
            child1.append_entry(permuted_p1_list[index], index)
        return child1

    def execute(self, solutions: [KGSSolution]):
        max_id = MuxLinkBase.instance().get_max_id(self.netlist_str)

        # TODO: zeng - implement mutation that makes sense for KGSS
        for solution in solutions:
            size = solution.get_size()
            mutate_flag = True
            mutate_flag1 = True
            mutate_flag2 = True
            mutate_count = 0
            solution1 = copy.copy(solution)# copy the original check later
            # check if there is cycles and redundancy
            # counter mightbe
            # g1_g2 = [j for sub in solution.data for j in sub[2:4]]
            # print("g1_g2", g1_g2)
            # f1_f2 = [j for sub in solution.data for j in sub[0:2]]
            # mightbe after mutate some position, it might have cycles; no need to keep running
            while mutate_flag:
                for i in range(size):
                    # decide if node is mutated
                    if random.uniform(0, 1) < self.node_mutation_probability:
                        # here we only mutate the node which is not in the critical path
                        key_value = int(solution.data[i][4]) # here is the key value
                        # here we mutate the g1 or g2 is better choice???
                        # ToDO: if i change the g1/g2, how about its corresponding f1/f2??
                        # let us find the f1/f2, and find it successor from the G
                        # mutated_f_node = random.randrange(0, max_id)

                        # also we need check if there is no repeated g1 or g2,
                        # unless we need to regenerate it again
                        g1_g2 = [j for sub in solution.data for j in sub[2:4]]
                        # print("g1_g2", g1_g2)
                        f1_f2 = [j for sub in solution.data for j in sub[0:2]]
                        # print("f1_f2", f1_f2)
                        G = solution.graph
                        G_info = dict(list(G.nodes(data="count")))
                        G_info_updated = {y: x for x, y in G_info.items()}
                        # print("G_info_updated")
                        # print(G_info_updated)
                        F_choice = self.ExtractSingleMultiOutputNodes(G)
                        # print("F_choice")
                        # print(F_choice)
                        f_node_name = random.choice(F_choice)
                        # print("old_f_name:",f_node_name)
                        mutated_f_node = list(G_info_updated.keys())[list(G_info_updated.values()).index(f_node_name)]
                        # mutated_f_node = str(mutated_f_node)
                        g_node_name = random.choice(list(G.successors(str(f_node_name))))
                        mutated_g_node = list(G_info_updated.keys())[list(G_info_updated.values()).index(g_node_name)]

                        g_flag = True
                        while g_flag:
                            if str(mutated_f_node) not in g1_g2 and str(mutated_f_node) not in f1_f2 and str(mutated_g_node) not in g1_g2 and str(mutated_g_node) not in f1_f2:
                                g_index = random.choice([0, 1]) # random choose f1 or f2
                                # f_node_name = G_info_updated[int(mutated_f_node)]
                                # print("new_f_name", f_node_name)
                                # g_node_name = random.choice(list(G.successors(str(f_node_name))))
                                # mutated_g_node = list(G_info_updated.keys())[list(G_info_updated.values()).index(g_node_name)]
                                # print("muteated_g_node:", mutated_g_node)
                                solution.data[i][g_index] = str(mutated_f_node)
                                solution.data[i][g_index+2] = str(mutated_g_node)
                                g_flag = False
                            else:
                                f_node_name = random.choice(F_choice)
                                mutated_f_node = list(G_info_updated.keys())[list(G_info_updated.values()).index(f_node_name)]
                                mutated_f_node = str(mutated_f_node)
                                g_node_name = random.choice(list(G.successors(str(f_node_name))))
                                mutated_g_node = list(G_info_updated.keys())[list(G_info_updated.values()).index(g_node_name)]
                                g_flag = True
                        # print("here is okay mutation process")
                print(solution.data)
                # after the random mutation, we also do the permutation before checking the redundancy and cycles
                if random.uniform(0, 1) < 0.01:
                    solution = self.generate_child_permute(solution)
                # 1) check if there is repeated element in the crossover data
                if solution.check_redundancy() == False:
                    print("no redundancy")
                    mutate_flag2 = False
                    # 2) update the graph and data
                    solution_updated = MuxLinkBase.instance().update(self.netlist_str, solution)
                    # print(solution_updated.data)
                    # print(solution_updated.graph)
                    # 3) check if there is cycle
                    if solution_updated.check_cycle() == False:
                        print("no cycles")
                        mutate_flag1 = False
                        solution = copy.copy(solution_updated)
                    else:
                        print("cycles")
                        solution.data = solution1.data.copy()
                        mutate_count += 1
                else:
                    # print("are you stucking here")
                    solution.data = solution1.data.copy()
                    mutate_count += 1
                mutate_flag = bool(mutate_flag1 + mutate_flag2)
                # if mutate_count >= 50:
                #     print("the mutation checking is not successful")
                #     print(solution.data)
                #     ## checking it later
                #     break
        return solutions
