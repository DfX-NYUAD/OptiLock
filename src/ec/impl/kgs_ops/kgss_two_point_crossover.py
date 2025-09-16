from ec.impl.kgs_solution import KGSSolution
from ec.impl.kgs_solution_new import KGSSolutionNew
from ec.model.crossover_operator import CrossoverOperator
from ec.impl.kgs_ops.muxlink_base import MuxLinkBase
import random
import copy


class KGSSTwoPointCrossover(CrossoverOperator):

    def __init__(self, netlist_str):
        self.netlist_str = netlist_str
    # crossover first try: based on the random index to shuffle 2 parents randomly
    def generate_children(self, solutions: KGSSolution) -> list:
        if len(solutions) < 2:
            raise ValueError("TwoPointCrossover expects two solutions.")

        parent1 = solutions[0]
        parent2 = solutions[1]
        # TODO: what about the solution id given to KGSS?
        # currently, no need for the solution id

        child1 = KGSSolution()  # id remove
        child2 = KGSSolution()

        # select indexes
        # make sure that index1 != index2
        size = parent1.get_size()
        indices = set()
        index1 = random.randint(0, size - 1)
        indices.add(index1)
        index2 = random.randint(0, size - 1)
        while index2 in indices:
            index2 = random.randint(0, size - 1)
        if index1 > index2:
            index1, index2 = index2, index1
        for j in range(index1):
            child1.append_entry(parent1.data[j], j)
            child2.append_entry(parent2.data[j], j)

        for j in range(index1, index2):
            child1.append_entry(parent2.data[j], j)
            child2.append_entry(parent1.data[j], j)

        for j in range(index2, size):
            child1.append_entry(parent1.data[j], j)
            child2.append_entry(parent2.data[j], j)
        # [x1, y1, z1], [x2,y2,z2] -> [x1, y2, z1], [x2,y1, z2]
        # print("previous one")
        # print(child1.data)
        # child1 = MuxLinkBase.instance().update(self.netlist_str, child1)
        # print("updated one")
        # print(child1.data)
        # child2 = MuxLinkBase.instance().update(self.netlist_str, child2)
        return [child1, child2]
    # crossover second try: based on the repeated g1 and g2, preprocess the parents
    def generate_children_new(self, solutions: KGSSolution) -> list:
        if len(solutions) < 2:
            raise ValueError("TwoPointCrossover expects two solutions.")

        parent1 = solutions[0]
        # sort the g1 and g2
        for item in parent1.data:
            if int(item[2]) > int(item[3]):
                item[2], item[3] = item[3], item[2]
                item[0], item[1] = item[1], item[0]
        parent2 = solutions[1]
        for item in parent2.data:
            if int(item[2]) > int(item[3]):
                item[2], item[3] = item[3], item[2]
                item[0], item[1] = item[1], item[0]

        g1_parent1 = [i for sub in parent1.data for i in sub[2:3]]
        g1_parent2 = [i for sub in parent2.data for i in sub[2:3]]
        g1_common = [value for value in g1_parent1 if value in g1_parent2]
        g1_p1_repeat = [g1_parent1.index(value) for value in g1_common]
        g1_p2_repeat = [g1_parent2.index(value) for value in g1_common]

        for index in range(len(g1_p1_repeat)):
            g1_p1_index = g1_p1_repeat[index]
            g1_p2_index = g1_p2_repeat[index]
            temp = parent1.data[g1_p1_index]
            parent1.data[g1_p1_index] = parent1.data[g1_p2_index]
            parent1.data[g1_p2_index] = temp
        for index in range(len(parent1.data)):
            parent1.data[index][5] = index
        # g1_parent1_new = [i for sub in parent1.data for i in sub[2:3]]
        # g1_common_new = [value for value in g1_parent1_new if value in g1_parent2]
        # g1_p1_repeat_new = [g1_parent1_new.index(value) for value in g1_common_new]
        # g1_p2_repeat_new = [g1_parent2.index(value) for value in g1_common_new]

        # here we also want to reorder the parents based on the g2
        g2_parent1 = [i for sub in parent1.data for i in sub[3:4]]
        g2_parent2 = [i for sub in parent2.data for i in sub[3:4]]
        g2_common = [value for value in g2_parent1 if value in g2_parent2]
        g2_p1_repeat = [g2_parent1.index(value) for value in g2_common]
        g2_p2_repeat = [g2_parent2.index(value) for value in g2_common]
        for index in range(len(g2_p1_repeat)):
            g2_p1_index = g2_p1_repeat[index]
            g2_p2_index = g2_p2_repeat[index]
            temp = parent1.data[g2_p1_index]
            parent1.data[g2_p1_index] = parent1.data[g2_p2_index]
            parent1.data[g2_p2_index] = temp
        for index in range(len(parent1.data)):
            parent1.data[index][5] = index

        child1 = KGSSolution()  # id remove
        child2 = KGSSolution()

        # select indexes
        # make sure that index1 != index2
        size = parent1.get_size()
        indices = set()
        index1 = random.randint(0, size - 1)
        indices.add(index1)
        index2 = random.randint(0, size - 1)
        while index2 in indices:
            index2 = random.randint(0, size - 1)
        if index1 > index2:
            index1, index2 = index2, index1
        for j in range(index1):
            child1.append_entry(parent1.data[j], j)
            child2.append_entry(parent2.data[j], j)

        for j in range(index1, index2):
            child1.append_entry(parent2.data[j], j)
            child2.append_entry(parent1.data[j], j)

        for j in range(index2, size):
            child1.append_entry(parent1.data[j], j)
            child2.append_entry(parent2.data[j], j)
        # [x1, y1, z1], [x2,y2,z2] -> [x1, y2, z1], [x2,y1, z2]
        # print("previous one")
        # print(child1.data)
        # child1 = MuxLinkBase.instance().update(self.netlist_str, child1)
        # print("updated one")
        # print(child1.data)
        # child2 = MuxLinkBase.instance().update(self.netlist_str, child2)
        return [child1, child2]

    # instead of crossover the parents to generate the children
    # here we try to generate the children by permutation
    # we want to implement two different scenarios:
    # 1) just permute each individual parent to get new children
    # parent: [f1, f2, g1, g2, key1] [f3,f4, g3, g4, key2] -> [f1, g1] , [f2, g2],..[f4,g4] -> shuffle -> new children [f1]
    # [f1, g1], [f2, g2], [f3, g3], [f4, g4]
    # [f1, f3, g1, g3, key1], [f2, f4, g2, g4, key2]
    # 2) summary all g names among 2 parents, and then pick up them randomly to form 2 children
    # parent1: [f1, f2, g1, g2]*64, parent2:[f3, f4, g1, g4]*64 -> dict -> merge [f, g] -> randomly 2 from keys and values -> permutation
    # key -> g ; value -> f
    # g1:[f1, f3] -> random.choice([f1, f3]) -> [g1, f3]
    # here it better for the mutation, not for the crossover
    def generate_children_permute1(self, solutions: KGSSolution) -> list:
        if len(solutions) < 2:
            raise ValueError("TwoPointCrossover expects two solutions.")
        parent1 = solutions[0]
        parent2 = solutions[1]

        # here we get the g1 and g2, but we need to figure out which g we could change
        # also we also need to figure out which f we could change
        # for each encoding item, [f1, g1] and [f2, g2] are two locking pair we have
        # therefore, we could record them and then permute them
        my_pair_p1 = {}
        my_pair_p2 = {}
        parent1_data = parent1.data
        parent2_data = parent2.data
        for index in range(len(parent1_data)):
            temp_list = parent1_data[index]
            my_pair_p1[temp_list[2]] = temp_list[0]
            my_pair_p1[temp_list[3]] = temp_list[1]
        for index in range(len(parent2_data)):
            temp_list = parent2_data[index]
            my_pair_p2[temp_list[2]] = temp_list[0]
            my_pair_p2[temp_list[3]] = temp_list[1]

        # because the key value is not too important to the encoding list
        # it is secure to just random shuffle the two pair
        p1_pair_keys = list(my_pair_p1.keys())
        p1_pair_values = list(my_pair_p1.values())
        p2_pair_keys = list(my_pair_p2.keys())
        p2_pair_values = list(my_pair_p2.values())
        # in order to avoid there is path between [f1, g2] and [f2, g1]
        # here we loop to random choose the keys of the dict
        # and check if the [key1, key2, value1, value2] have some value repeated.
        permuted_p1_list = []
        permuted_p2_list = []
        # here is for parent1
        for index in range(len(parent1_data)):
            p1_flag = True
            while p1_flag:
                key_temp = random.sample(p1_pair_keys, 2)  # random choose 2
                value_temp = [my_pair_p1.get(key) for key in key_temp]
                if not set(key_temp).intersection(set(value_temp)):
                    p1_flag = False
                    temp_list = [value_temp[0], value_temp[1], key_temp[0], key_temp[1], parent1_data[index][4],
                                 parent1_data[index][5]]
                    permuted_p1_list.append(temp_list)
                    for item in key_temp:  # avoid repeated keys
                        p1_pair_keys.remove(item)
                else:
                    p1_flag = True  # continue searching
        # here is for parent2
        for index in range(len(parent2_data)):
            p2_flag = True
            while p2_flag:
                key_temp = random.sample(p2_pair_keys, 2)  # random choose 2
                value_temp = [my_pair_p2.get(key) for key in key_temp]
                if not set(key_temp).intersection(set(value_temp)):
                    p2_flag = False
                    temp_list = [value_temp[0], value_temp[1], key_temp[0], key_temp[1], parent2_data[index][4],
                                 parent2_data[index][5]]
                    permuted_p2_list.append(temp_list)
                    for item in key_temp:  # avoid repeated keys
                        p2_pair_keys.remove(item)
                else:
                    p2_flag = True  # continue searching

        child1 = KGSSolution()
        child2 = KGSSolution()
        for index in range(len(permuted_p1_list)):
            child1.append_entry(permuted_p1_list[index], index)
            child2.append_entry(permuted_p2_list[index], index)
        return [child1, child2]

    def generate_children_permute2(self, solutions: KGSSolution) -> list:
        if len(solutions) < 2:
            raise ValueError("TwoPointCrossover expects two solutions.")
        parent1 = solutions[0]
        parent2 = solutions[1]

        # here we get the g1 and g2, but we need to figure out which g we could change
        # also we also need to figure out which f we could change
        # for each encoding item, [f1, g1] and [f2, g2] are two locking pair we have
        # therefore, we could record them and then permute them
        my_pair_p1 = {}
        my_pair_p2 = {}
        parent1_data = parent1.data
        parent2_data = parent2.data
        for index in range(len(parent1_data)):
            temp_list = parent1_data[index]
            my_pair_p1[temp_list[2]] = temp_list[0]
            my_pair_p1[temp_list[3]] = temp_list[1]
        for index in range(len(parent2_data)):
            temp_list = parent2_data[index]
            my_pair_p2[temp_list[2]] = temp_list[0]
            my_pair_p2[temp_list[3]] = temp_list[1]

        # because the key value is not too important to the encoding list
        # it is secure to just random shuffle the two pair
        # here we merge these 2 pairs, and save all the info here
        merged_dict = {}
        for key in my_pair_p1.keys() | my_pair_p2.keys():
            if key in my_pair_p1 and key in my_pair_p2:
                merged_dict[key] = [my_pair_p1[key], my_pair_p2[key]]
            elif key in my_pair_p1:
                merged_dict[key] = my_pair_p1[key]
            else:
                merged_dict[key] = my_pair_p2[key]
        merge_keys = list(merged_dict.keys())
        permuted_p1_list = []
        permuted_p2_list = []
        # here is for child1
        for index in range(len(parent1_data)):
            p1_flag = True
            while p1_flag:
                key_temp = random.sample(merge_keys, 2)  # random choose
                value_te = [merged_dict.get(key) for key in key_temp]
                # print("value_te", value_te)
                value_temp = []
                for item in value_te:
                    if len(item) > 1 and isinstance(item, list):
                        value_temp.append(random.choice(item))
                    else:
                        value_temp.append(item)
                if not set(key_temp).intersection(set(value_temp)):
                    p1_flag = False
                    temp_list = [value_temp[0], value_temp[1], key_temp[0], key_temp[1], parent1_data[index][4],
                                 parent1_data[index][5]]
                    permuted_p1_list.append(temp_list)
                    for item in key_temp:  # avoid repeated keys
                        merge_keys.remove(item)
                else:
                    p1_flag = True  # continue searching
        # here is for child2
        merge_keys = list(merged_dict.keys())
        for index in range(len(parent2_data)):
            p2_flag = True
            while p2_flag:
                key_temp = random.sample(merge_keys, 2)  # random choose
                value_te = [merged_dict.get(key) for key in key_temp]
                # print("value_te", value_te)
                value_temp = []
                for item in value_te:
                    if len(item) > 1 and isinstance(item, list):
                        value_temp.append(random.choice(item))
                    else:
                        value_temp.append(item)
                if not set(key_temp).intersection(set(value_temp)):
                    p2_flag = False
                    temp_list = [value_temp[0], value_temp[1], key_temp[0], key_temp[1], parent2_data[index][4],
                                 parent2_data[index][5]]
                    permuted_p2_list.append(temp_list)
                    for item in key_temp:  # avoid repeated keys
                        merge_keys.remove(item)
                else:
                    p2_flag = True  # continue searching
        child1 = KGSSolution()
        child2 = KGSSolution()
        for index in range(len(permuted_p1_list)):
            child1.append_entry(permuted_p1_list[index], index)
            child2.append_entry(permuted_p2_list[index], index)

        return [child1, child2]

    # crossover is key element for the GA; therefore, it is not good choice to choose permutation
    def generate_children_dictmethod(self, solutions: KGSSolution) -> list:
        if len(solutions) < 2:
            raise ValueError("TwoPointCrossover expects two solutions.")
        parent1 = solutions[0]
        parent2 = solutions[1]
        # here we firstly merge g1 and g2 together;
        # figure our which elements include in these 2 parents
        # dict should be store as like g1-> p0_0, p1_0, g2->p0_0 ...
        parent1_data = parent1.data
        parent2_data = parent2.data
        my_pair_p1 = {}
        my_pair_p2 = {}
        fg_pair_p1 = {}
        fg_pair_p2 = {}
        for index in range(len(parent1_data)):
            temp_list = parent1_data[index]
            # for parent1's g locality, store p1_index
            my_pair_p1[temp_list[2]] = "p1_" + str(index)
            my_pair_p1[temp_list[3]] = "p1_" + str(index)
            fg_pair_p1[temp_list[2]] = temp_list[0]
            fg_pair_p1[temp_list[3]] = temp_list[1]
            temp_list = parent2_data[index]
            # for parent2's g locality, store p2_index
            my_pair_p2[temp_list[2]] = "p2_" + str(index)
            my_pair_p2[temp_list[3]] = "p2_" + str(index)
            fg_pair_p2[temp_list[2]] = temp_list[0]
            fg_pair_p2[temp_list[3]] = temp_list[1]
        # merge these two dicts
        merged_dict = {}
        for key in my_pair_p1.keys() | my_pair_p2.keys():
            if key in my_pair_p1 and key in my_pair_p2:
                merged_dict[key] = [my_pair_p1[key], my_pair_p2[key]]
            elif key in my_pair_p1:
                merged_dict[key] = [my_pair_p1[key]]
            else:
                merged_dict[key] = [my_pair_p2[key]]
        # get all g list
        print(merged_dict)
        all_g_list = list(merged_dict.keys())
        # randomly shuffle the all_g_list
        random.shuffle(all_g_list)
        g_child1 = [[] for i in range(len(parent1_data))]
        g_child2 = [[] for i in range(len(parent2_data))]
        # here I deep copy the dict from the merge_dict
        multiple_g_dict = {}
        # print(all_g_list)
        for index in range(len(all_g_list)):
            g_item = merged_dict[all_g_list[index]]
            if len(g_item) > 1:  # if this g node have more than one mappings
                # random.shuffle(g_item)
                g_item_idx0 = int(g_item[0].split("_")[-1])
                g_item_idx1 = int(g_item[1].split("_")[-1])
                index_temp = random.choices([1, 2], weights=[0.15, 0.85])[0]
                if index_temp == 1:
                    g_child1[g_item_idx0].append(g_item[0])
                    g_child2[g_item_idx1].append(g_item[1])
                else:
                    g_child1[g_item_idx1].append(g_item[0])
                    g_child2[g_item_idx0].append(g_item[1])
        for index in range(len(all_g_list)):
            g_item = merged_dict[all_g_list[index]]
            if len(g_item) == 1:
                g_item_idx = int(g_item[0].split("_")[-1])
                if len(g_child1[g_item_idx]) == 0 and len(g_child2[g_item_idx]) == 0:
                    index_temp = random.choices([1,2], weights=[0.85, 0.15])[0]
                    if index_temp == 1:
                        g_child1[g_item_idx].append(g_item[0])
                    else:
                        g_child2[g_item_idx].append(g_item[0])
        # print(g_child1)
        # print(g_child2)
        for index in range(len(g_child1)):
            if len(g_child1[index]) == 0 and len(g_child2[index]) != 0:
                parent_name_temp = g_child2[index][0].split("_")[0]
                if "p1" in parent_name_temp:
                    g_child1[index].append("p2_" + str(index))
                    g_child1[index].append("p2_" + str(index))
                else:
                    g_child1[index].append("p1_" + str(index))
                    g_child1[index].append("p1_" + str(index))
            if len(g_child2[index]) == 0 and len(g_child1[index]) != 0:
                parent_name_temp = g_child1[index][0].split("_")[0]
                if "p1" in parent_name_temp:
                    g_child2[index].append("p2_" + str(index))
                    g_child2[index].append("p2_" + str(index))
                else:
                    g_child2[index].append("p1_" + str(index))
                    g_child2[index].append("p1_" + str(index))
            if len(g_child1[index]) == 1:
                g_item = g_child1[index][0]
                g_child1[index].append(g_item)
            if len(g_child2[index]) == 1:
                g_item = g_child2[index][0]
                g_child2[index].append(g_item)
        child1_list = []
        child2_list = []
        for index in range(len(g_child1)):
            p1_index = g_child1[index][0].split("_")[0]
            p2_index = g_child2[index][0].split("_")[0]
            if "p1" in p1_index:
                child1_list.append(parent1_data[index])
            elif "p2" in p1_index:
                child1_list.append(parent2_data[index])
            if "p1" in p2_index:
                child2_list.append(parent1_data[index])
            elif "p2" in p2_index:
                child2_list.append(parent2_data[index])

        child1 = KGSSolution()
        child2 = KGSSolution()
        for index in range(len(child1_list)):
            child1.append_entry(child1_list[index], index)
            child2.append_entry(child1_list[index], index)
        return [child1, child2]


    def execute(self, solutions):
        # return self.generate_children(solutions)
        # crossover parents to create new child
        # check if there is the cycle in the children
        crossover_flag = True
        crossover_flag1 = True
        crossover_flag2 = True
        cross_count = 0
        # Note：here children only change the data value, not the graph value
        # children = self.generate_children_new(solutions)
        children = self.generate_children_dictmethod(solutions)
        while crossover_flag:
            print("keeping checking")
            # children = self.generate_children(solutions)
            # children_updated = children.copy()
            # 1) check if there is repeated element in the crossover data
            # print(children[0].data)
            if (children[0].check_redundancy() == False) and (children[1].check_redundancy() == False):
                crossover_flag2 = False
                print("no redundancy")
                # 2) update the data and graph of the kgss
                if children[0] == solutions[0].data or children[1] == solutions[1].data \
                    or children[1] == solutions[1].data or children[0] == solutions[0].data:
                    print("the children does not get updated")
                children_updated0 = MuxLinkBase.instance().update(self.netlist_str, children[0])
                children_updated1 = MuxLinkBase.instance().update(self.netlist_str, children[1])
                # 3) check if there is cycles in the crossover data
                if (children_updated0.check_cycle() == False) and (children_updated1.check_cycle() == False):
                    crossover_flag1 = False
                    print("no cycles")
                    # print("children_update0 graph")
                    # print(children_updated0.graph)
                    children[0] = children_updated0
                    children[1] = children_updated1  # all satisfy, we will update the graph
                else:
                    # unless we only keep the data to do the crossover next iteration
                    # children = self.generate_children(solutions)
                    cross_count += 1
            else:
                # children = self.generate_children(solutions) # regenerate the children based on the solutions
                cross_count += 1
            crossover_flag = bool(crossover_flag1 + crossover_flag2)
            if crossover_flag:  # if these conditions are not satisfied, regenerate it again
                # children = self.generate_children_new(solutions)
                children = self.generate_children_dictmethod(solutions)
            # if cross_count >= 500:
            #     print("the crossover checking is not successful")
            #     # children[0] = MuxLinkBase.instance().update(self.netlist_str, solutions[0])
            #     # children[1] = MuxLinkBase.instance().update(self.netlist_str, solutions[0])
            #     children = solutions
            #     # we have to see the backup plan :
            #     # change to higher num of cross_count or leave it open
            #     # because no crossover here
            #     # only update the original graph
            #     break
        return children
