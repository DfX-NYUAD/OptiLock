import networkx as nx
from ec.model.solution import Solution
from collections import Counter
import random
import multiprocessing as mp
from functools import partial

# Idea: label each node with a unique ID (0-N)
# Have a system that knows in which layer the ID belongs to (important to select the requirements for new selections)
# Representation length == key budget (S4 1 bit for 2 MUX)
# Concept: just store locations of where the "change" is inserted
# Eg for 4 bits: [[fi1, fj1, gi1, gj1, Ki, TYPE of STRATEGY], [fi2, fj2, gi2, gj2, Ki+1], [X], [X]]
# x-> empty slot
# Ki - correct key

# Key Gate Signature Solution (KGSS)
class KGSSolution(Solution):

    def __init__(self, netlist=None):
        # super().__init__(id) # remove later
        self.data = []

        # reference to bench netlist
        self.netlist = netlist
        # reference to graph
        self.graph = nx.DiGraph()  # initialize an empty graph

    def add_entry(self, f1, f2, g1, g2, key, key_name):
        self.data.append([f1, f2, g1, g2, key, key_name])

    def assign_entry(self, f1, f2, g1, g2, key, key_name):
        key_index = int(key_name)
        self.data[key_index] = [f1, f2, g1, g2, key, key_name]

    def append_entry(self, sol, key_name):
        self.data.append([sol[0], sol[1], sol[2], sol[3], sol[4], key_name])

    def register_graph(self, G):
        # record each graph after loading the graph
        self.graph = G.copy()
    # get the key size
    def get_key_size(self):
        return len(self.data)
    # get the key value string 
    def get_key_value(self):
        key_value_string = ""
        for item in self.data:
            key_value_string += str(item[4])
        return key_value_string
    def check_cycle(self):
        # the original network does not have cycles
        # encode, decode storing
        # return true or false
        if (len(list(nx.simple_cycles(self.graph)))) != 0:
            print("here is the cycles")
            print(list(nx.simple_cycles(self.graph)))
            return True
        else:
            return False

    def check_redundancy(self):
        # here is used to check the redundancy after the crossover
        # [1,1,2] -> [1,2]
        # [f1,f2,g1,g2], [f2,f1,g2,g1] same
        # to avoid the order problem of f1,f2 or f2, f1, we swap each of them based on their value
        data_temp = self.data
        for item in data_temp:
            if "X" in item:
                return True
        for entry in data_temp:
            if int(entry[0]) > int(entry[1]):
                # swap 0 and 1 in entry
                entry[0], entry[1] = entry[1], entry[0]
                entry[2], entry[3] = entry[3], entry[2]
                # here I do not think the key value should be inverted
                # if int(entry[4]) == 0:
                #     entry[4] = "1"
                # else:
                #     entry[4] = "0"
        # print(data_temp)
        # here we do not conside the differnce of the key name
        temp_sol = [list(i) for i in list(set(map(tuple, data_temp)))]
        # print("checking redunc_test")
        # print(len(temp_sol))
        # print(len(self.data))
        g1_g2 = []
        if len(temp_sol) == len(self.data):
            # merge g1 and g2, which could be used to determine
            # if it is okay to keep the new updated kgss data
            for item in self.data:
                g1_g2.append(item[2])
                g1_g2.append(item[3])
            if len(list(set(g1_g2))) == int(len(self.data)*2):
                return False
            else:
                return True

        else:
            return True

    def get_data(self):
        return self.data

    def get_graph(self):
        return self.graph

    def get_netlist_num(self):
        encoding_list = []
        G = self.graph
        for n in G.nodes():
            if 'count' in G.nodes[n]:
                encoding_list.append([n, G.nodes[n]['count']])
        return len(encoding_list)

    def get_size(self):
        return len(self.data)

    def to_string(self):
        return ''.join(str(x) for x in self.data)

    # get the node that is locked
    def get_lock_node(self):
        lock_node_num = []
        lock_data = self.data
        for temp in lock_data:
            lock_node_num += [int(temp[0]), int(temp[1]), int(temp[2]), int(temp[3])]
        return lock_node_num

    # get the fan-in node that is locked (f1, f2)
    def get_lock_fanin(self):
        lock_node_num = []
        lock_data = self.data
        for temp in lock_data:
            lock_node_num += [int(temp[0]), int(temp[1])]
        return lock_node_num

    # get the fan-out node that is locked (g1, g2)
    def get_lock_fanout(self):
        lock_node_num = []
        lock_data = self.data
        for temp in lock_data:
            lock_node_num += [int(temp[2]), int(temp[3])]
        return lock_node_num

    # get all possible edge which could be locked
    # ignore the input and output node
    def get_all_possible_edges(self):
        input_node_list = []
        output_node_list = []
        G= self.graph
        # remove the input node
        for node in G.nodes():
            if 'input' in G.nodes[node]['gate']:
                input_node_list.append(node)
        # remove the output node
        for node in G.nodes:
            if G.nodes[node]['output']:
                output_node_list.append(node)
        # all_removable_nodes = input_node_list + output_node_list
        all_removable_nodes = input_node_list
        possible_edge = []
        for edge in G.edges():
            if edge[0] not in all_removable_nodes and edge[1] not in all_removable_nodes:
                possible_edge.append(edge)
        # numbering each edge in the dictionary
        # possible_edge_old = dict(zip(range(len(possible_edge)), possible_edge))
        # for each edge, we assign an even number to it store them in the dict
        # possible_edge = dict(zip(range(0, len(possible_edge)*2, 2), possible_edge))

        possible_edge = dict(zip(range(len(possible_edge)), possible_edge))
        # for each edge pair, we need to convert them to the count number
        for key in possible_edge:
            possible_edge[key] = [G.nodes[possible_edge[key][0]]['count'], G.nodes[possible_edge[key][1]]['count']]
        return possible_edge
    
    
    def get_all_possible_edges_scope(self):
        input_node_list = []
        output_node_list = []
        G= self.graph
        # remove the input node
        for node in G.nodes():
            if 'input' in G.nodes[node]['gate']:
                input_node_list.append(node)
        # remove the output node
        for node in G.nodes:
            if G.nodes[node]['output']:
                output_node_list.append(node)
        # all_removable_nodes = input_node_list + output_node_list
        all_removable_nodes = input_node_list
        possible_edge = []
        for edge in G.edges():
            if edge[0] not in all_removable_nodes and edge[1] not in all_removable_nodes and len(list(G.successors(edge[0]))) > 1:
                possible_edge.append(edge)
        # numbering each edge in the dictionary
        # possible_edge_old = dict(zip(range(len(possible_edge)), possible_edge))
        # for each edge, we assign an even number to it store them in the dict
        # possible_edge = dict(zip(range(0, len(possible_edge)*2, 2), possible_edge))

        possible_edge = dict(zip(range(len(possible_edge)), possible_edge))
        # for each edge pair, we need to convert them to the count number
        for key in possible_edge:
            possible_edge[key] = [G.nodes[possible_edge[key][0]]['count'], G.nodes[possible_edge[key][1]]['count']]
        return possible_edge
    
    def filter_and_process_edges(self, chunk_G_all_removable_nodes):
        chunk, G, all_removable_nodes = chunk_G_all_removable_nodes
        possible_edge = []
        for edge in chunk:
            if edge[0] not in all_removable_nodes and edge[1] not in all_removable_nodes:
                possible_edge.append(edge)
        return possible_edge
    
    def filter_and_process_edges_scope(self, chunk_G_all_removable_nodes):
        chunk, G, all_removable_nodes = chunk_G_all_removable_nodes
        possible_edge = []
        for edge in chunk:
            if edge[0] not in all_removable_nodes and edge[1] not in all_removable_nodes and len(list(G.successors(edge[0]))) > 1:
                possible_edge.append(edge)
        return possible_edge

    def get_all_possible_edges_large(self):
        input_node_list = []
        output_node_list = []
        G= self.graph
        # remove the input node
        for node in G.nodes():
            if 'input' in G.nodes[node]['gate']:
                input_node_list.append(node)
        # remove the output node
        for node in G.nodes:
            if G.nodes[node]['output']:
                output_node_list.append(node)
        # all_removable_nodes = input_node_list + output_node_list
        all_removable_nodes = input_node_list
        # possible_edge = []
        # for edge in G.edges():
        #     if edge[0] not in all_removable_nodes and edge[1] not in all_removable_nodes:
        #         possible_edge.append(edge)
        edges = list(G.edges())
        num_workers = mp.cpu_count()
        print("num_workers zeng", num_workers, flush=True)
        print("len of edges", len(edges), flush=True)
        chunk_size = len(edges) // num_workers
        print("chunk_size", chunk_size, flush=True)

        # def filter_and_process_edges(chunk_G_all_removable_nodes):
        #     chunk, G, all_removable_nodes = chunk_G_all_removable_nodes
        #     possible_edge = []
        #     for edge in chunk:
        #         if edge[0] not in all_removable_nodes and edge[1] not in all_removable_nodes:
        #             possible_edge.append(edge)
        #     return possible_edge
        
        with mp.Pool(num_workers) as pool:
            chunks = [edges[i:i + chunk_size] for i in range(0, len(edges), chunk_size)]
            results = pool.map(self.filter_and_process_edges, [(chunk, G, all_removable_nodes) for chunk in chunks])
        possible_edge = [edge for result in results for edge in result]
        # numbering each edge in the dictionary
        # possible_edge_old = dict(zip(range(len(possible_edge)), possible_edge))
        # for each edge, we assign an even number to it store them in the dict
        # possible_edge = dict(zip(range(0, len(possible_edge)*2, 2), possible_edge))

        possible_edge = dict(zip(range(len(possible_edge)), possible_edge))
        # for each edge pair, we need to convert them to the count number
        for key in possible_edge:
            possible_edge[key] = [G.nodes[possible_edge[key][0]]['count'], G.nodes[possible_edge[key][1]]['count']]
        print("here is done")
        return possible_edge
    
    def get_all_possible_edges_large_scope(self):
        input_node_list = []
        output_node_list = []
        G= self.graph
        # remove the input node
        for node in G.nodes():
            if 'input' in G.nodes[node]['gate']:
                input_node_list.append(node)
        # remove the output node
        for node in G.nodes:
            if G.nodes[node]['output']:
                output_node_list.append(node)
        # all_removable_nodes = input_node_list + output_node_list
        all_removable_nodes = input_node_list
        # possible_edge = []
        # for edge in G.edges():
        #     if edge[0] not in all_removable_nodes and edge[1] not in all_removable_nodes:
        #         possible_edge.append(edge)
        edges = list(G.edges())
        num_workers = mp.cpu_count()
        print("num_workers", num_workers)
        chunk_size = len(edges) // num_workers

        # def filter_and_process_edges(chunk_G_all_removable_nodes):
        #     chunk, G, all_removable_nodes = chunk_G_all_removable_nodes
        #     possible_edge = []
        #     for edge in chunk:
        #         if edge[0] not in all_removable_nodes and edge[1] not in all_removable_nodes:
        #             possible_edge.append(edge)
        #     return possible_edge
        
        with mp.Pool(num_workers) as pool:
            chunks = [edges[i:i + chunk_size] for i in range(0, len(edges), chunk_size)]
            results = pool.map(self.filter_and_process_edges_scope, [(chunk, G, all_removable_nodes) for chunk in chunks])
        possible_edge = [edge for result in results for edge in result]
        # numbering each edge in the dictionary
        # possible_edge_old = dict(zip(range(len(possible_edge)), possible_edge))
        # for each edge, we assign an even number to it store them in the dict
        # possible_edge = dict(zip(range(0, len(possible_edge)*2, 2), possible_edge))

        possible_edge = dict(zip(range(len(possible_edge)), possible_edge))
        # for each edge pair, we need to convert them to the count number
        for key in possible_edge:
            possible_edge[key] = [G.nodes[possible_edge[key][0]]['count'], G.nodes[possible_edge[key][1]]['count']]
        return possible_edge
    
    
    # get the impacted output in the fan cone
    def get_impacted_output(self):
        possible_edges = self.get_fan_cone_edges()
        # get the fanout list
        output_node_list = []
        G = self.graph
        G_copy = self.graph
        G_info = dict(list(G_copy.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        for node in G.nodes:
            if G.nodes[node]['output']:
                output_node_list.append(node)
        # collect all the nodes in the edges
        selected_nodes = []
        print("G_info_updated", G_info_updated)
        for edge in possible_edges:
            edge_item = possible_edges[edge]
            edge_temp = [G_info_updated[edge_item[0]], G_info_updated[edge_item[1]]]
            selected_nodes += edge_temp
        # find out if there is connection between the node in the selected nodes and the output node
        impacted_output = []
        for node in selected_nodes:
            # check if this node is connecting any nodes in the ourput node list
            for output_node in output_node_list:
                if nx.has_path(G, node, output_node):
                    impacted_output.append(output_node)
        # remove the repeated elements in the impacted nodes
        impacted_output = list(set(impacted_output))
        print("what is the impacted output")
        print(impacted_output)


    def get_fan_cone_edges(self):
        input_node_list = []
        output_node_list = []
        G= self.graph
        # get the fanout node
        # remove the input node
        for node in G.nodes():
            if 'input' in G.nodes[node]['gate']:
                input_node_list.append(node)
        # remove the output node
        for node in G.nodes:
            if G.nodes[node]['output']:
                output_node_list.append(node)
        # print(output_node_list)
        # for node_name in output_node_list:
        #     # print(node_name)
        #     print(G.nodes[node_name]['count'])
        fanin_cones_edges = {node: self.edges_in_fanin_cone(G, node, input_node_list) for node in output_node_list}
        sorted_output_nodes = sorted(fanin_cones_edges, key=lambda node: len(fanin_cones_edges[node]), reverse=True)
        # print each nodes and its fanin_cones_edges size
        # for node in sorted_output_nodes:
        #     print(node, len(fanin_cones_edges[node]))
        # merged_edges = set()
        approx_range = 2000
        # if the maximum number is less than 3000, we could use the maximum number
        if len(fanin_cones_edges[sorted_output_nodes[0]]) < approx_range:
            merged_edges = set()
            for node in sorted_output_nodes:
                 merged_edges.update(fanin_cones_edges[node])
                 if len(merged_edges) >= approx_range:
                     break
            # print("what is the length of merged_edges1")
            # print(len(merged_edges))
            # return the possible edge as a dictionary with index number
            possible_edge = dict(zip(range(len(merged_edges)), merged_edges))
            # for each edge pair, we need to convert them to the count number
            for key in possible_edge:
                possible_edge[key] = [G.nodes[possible_edge[key][0]]['count'], G.nodes[possible_edge[key][1]]['count']]
            # print("what is possible_edge")
            # print(possible_edge)
            return possible_edge
        # else we would choose the closet number to 3000
        else:
            merged_edges = set()
            sorted_edge_num_list = []
            sorted_node_list = []
            for node in sorted_output_nodes:
                sorted_edge_num_list.append(len(fanin_cones_edges[node]))
                sorted_node_list.append(node)
            # min(sorted_edge_num_list, key=lambda x:abs(x-approx_range))
            # find the index of the min(sorted_edge_num_list, key=lambda x:abs(x-approx_range))
            index = sorted_edge_num_list.index(min(sorted_edge_num_list, key=lambda x:abs(x-approx_range)))
            merged_edges.update(fanin_cones_edges[sorted_node_list[index]])
            possible_edge = dict(zip(range(len(merged_edges)), merged_edges))
            # for each edge pair, we need to convert them to the count number
            for key in possible_edge:
                possible_edge[key] = [G.nodes[possible_edge[key][0]]['count'], G.nodes[possible_edge[key][1]]['count']]
            # print("what is possible_edge2")
            # print(possible_edge)
            return possible_edge
    
    def edges_in_fanin_cone(self, G, node, input_node_list):
        fanin_nodes = nx.ancestors(G, node)
        fanin_nodes.add(node)
        edges = [(u, v) for u, v in G.edges() if u in fanin_nodes and v in fanin_nodes and u not in input_node_list and v not in input_node_list]
        return edges

    def edges_in_fanin_cone_scope(self, G, node, input_node_list):
        fanin_nodes = nx.ancestors(G, node)
        fanin_nodes.add(node)
        edges = [(u, v) for u, v in G.edges() if u in fanin_nodes and v in fanin_nodes and u not in input_node_list and v not in input_node_list and len(list(G.successors(u))) > 1]
        return edges
    
    def edges_in_fanin_cone_large(self, args):
        G, node, input_node_list = args
        # edges_list = []
        fanin_nodes = nx.ancestors(G, node)
        fanin_nodes.add(node)
        edges = [
            (u, v) for u, v in G.edges()
            if u in fanin_nodes and v in fanin_nodes and u not in input_node_list and v not in input_node_list
        ]
        return node, edges
    
    def edges_in_fanin_cone_large_scope(self, args):
        G, node, input_node_list = args
        # edges_list = []
        fanin_nodes = nx.ancestors(G, node)
        fanin_nodes.add(node)
        edges = [
            (u, v) for u, v in G.edges()
            if u in fanin_nodes and v in fanin_nodes and u not in input_node_list and v not in input_node_list and len(list(G.successors(u))) > 1
        ]
        return node, edges
    
    # --- worker caches (class-level, one copy per process) ---
    _W_G = None
    _W_INPUTS = None
    _W_SELF = None

    @classmethod
    def _fanin_init(cls, G, input_nodes, self_obj):
        cls._W_G = G
        cls._W_INPUTS = set(input_nodes)
        cls._W_SELF = self_obj

    @classmethod
    def _fanin_task(cls, node):
        # IMPORTANT: pass a single tuple to match the original signature
        # edges_in_fanin_cone_large(self, args)
        return cls._W_SELF.edges_in_fanin_cone_large((cls._W_G, node, cls._W_INPUTS))


    def get_fan_cone_edges_all_large(self):
        # _FANIN_G = None
        # _FANIN_INPUTS = None
        # _FANIN_SELF = None
        from time import time
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None

        input_node_list = []
        output_node_list = []
        G= self.graph
        # get the fanout node
        # remove the input node
        for node in G.nodes():
            if 'input' in G.nodes[node]['gate']:
                input_node_list.append(node)
        # remove the output node
        for node in G.nodes:
            if G.nodes[node]['output']:
                output_node_list.append(node)
        # print(output_node_list)
        # for node_name in output_node_list:
        #     # print(node_name)
        #     print(G.nodes[node_name]['count'])
        # fanin_cones_edges = {node: self.edges_in_fanin_cone(G, node, input_node_list) for node in output_node_list}
        num_workers = mp.cpu_count()
        print("num_workers", num_workers)
        
        # print how may edges here:
        print("len of output_node_list", len(output_node_list))

        # chunk_size = len(output_node_list) // num_workers

        # with mp.Pool(mp.cpu_count()) as pool:
        #     # chunks = [output_node_list[i:i + chunk_size] for i in range(0, len(output_node_list), chunk_size)]
        #     results = pool.map(self.edges_in_fanin_cone_large, [(G, node, input_node_list) for node in output_node_list])
        # fanin_cones_edges = {node: edges for node, edges in results}
        chunksize = max(1, len(output_node_list) // (num_workers * 16))
        
        ctx = mp.get_context('fork') if 'fork' in mp.get_all_start_methods() else mp.get_context('spawn')

        # with ctx.Pool(
        #     processes=num_workers,
        #     initializer=self.__class__._fanin_init,           # <-- classmethod
        #     initargs=(G, input_node_list, self),              # sent once per worker
        # ) as pool:
        #     results_iter = pool.imap_unordered(
        #         self.__class__._fanin_task,                  # <-- classmethod
        #         output_node_list,
        #         chunksize=chunksize
        #     )
        #     fanin_cones_edges = dict(results_iter)
        with ctx.Pool(
            processes=num_workers,
            initializer=self.__class__._fanin_init,        # classmethod you already added
            initargs=(G, input_node_list, self),
        ) as pool:
            results_iter = pool.imap_unordered(self.__class__._fanin_task,
                                            output_node_list,
                                            chunksize=chunksize)

            fanin_cones_edges = {}
            total = len(output_node_list)

            if tqdm:
                # Nice progress bar
                for node, edges in tqdm(results_iter, total=total, desc="fanin cones", dynamic_ncols=True, mininterval=0.5):
                    fanin_cones_edges[node] = edges
            else:
                # Lightweight fallback with periodic ETA
                start = time()
                report_every = max(1, total // 100)  # ~1% steps
                done = 0
                for node, edges in results_iter:
                    fanin_cones_edges[node] = edges
                    done += 1
                    if (done % report_every == 0) or (done == total):
                        elapsed = time() - start
                        rate = done / elapsed if elapsed > 0 else 0.0
                        remain = (total - done) / rate if rate > 0 else float('inf')
                        print(f"[{done}/{total}] {rate:.1f} it/s | ETA {remain/60:.1f} min", flush=True)
        # fanin_cones_edges = {node: edge for chunk, edges in results for edge in edges}
        # fanin_cones_edges = {}
        # for chunk, edges_list in results:
        #     for node, edges in zip(chunk, edges_list):
        #         fanin_cones_edges[node] = edges
        # sort from the smallest to the largest
        sorted_output_nodes = sorted(fanin_cones_edges, key=lambda node: len(fanin_cones_edges[node]), reverse=True)

        sorted_possible_edge_list = [] # here is for recording the edges which containing the fanin_cones_edges
   
        for node in sorted_output_nodes:

            possible_edge = dict(zip(range(len(fanin_cones_edges[node])), fanin_cones_edges[node]))

            for key in possible_edge:
                possible_edge[key] = [G.nodes[possible_edge[key][0]]['count'], G.nodes[possible_edge[key][1]]['count']]
    
            sorted_possible_edge_list.append(possible_edge)
        
        print("here is done 1")

        return sorted_possible_edge_list   


    def get_fan_cone_edges_all_large_scope(self):
        input_node_list = []
        output_node_list = []
        G= self.graph
        # get the fanout node
        # remove the input node
        for node in G.nodes():
            if 'input' in G.nodes[node]['gate']:
                input_node_list.append(node)
        # remove the output node
        for node in G.nodes:
            if G.nodes[node]['output']:
                output_node_list.append(node)
        # print(output_node_list)
        # for node_name in output_node_list:
        #     # print(node_name)
        #     print(G.nodes[node_name]['count'])
        # fanin_cones_edges = {node: self.edges_in_fanin_cone(G, node, input_node_list) for node in output_node_list}
        num_workers = mp.cpu_count()
        print("num_workers", num_workers)
        # chunk_size = len(output_node_list) // num_workers

        with mp.Pool(mp.cpu_count()) as pool:
            # chunks = [output_node_list[i:i + chunk_size] for i in range(0, len(output_node_list), chunk_size)]
            results = pool.map(self.edges_in_fanin_cone_large_scope, [(G, node, input_node_list) for node in output_node_list])
        fanin_cones_edges = {node: edges for node, edges in results}
        # fanin_cones_edges = {node: edge for chunk, edges in results for edge in edges}
        # fanin_cones_edges = {}
        # for chunk, edges_list in results:
        #     for node, edges in zip(chunk, edges_list):
        #         fanin_cones_edges[node] = edges
        # sort from the smallest to the largest
        sorted_output_nodes = sorted(fanin_cones_edges, key=lambda node: len(fanin_cones_edges[node]), reverse=True)

        sorted_possible_edge_list = [] # here is for recording the edges which containing the fanin_cones_edges
   
        for node in sorted_output_nodes:

            possible_edge = dict(zip(range(len(fanin_cones_edges[node])), fanin_cones_edges[node]))

            for key in possible_edge:
                possible_edge[key] = [G.nodes[possible_edge[key][0]]['count'], G.nodes[possible_edge[key][1]]['count']]
    
            sorted_possible_edge_list.append(possible_edge)

        return sorted_possible_edge_list   
    
    def get_fan_cone_edges_all(self):
        input_node_list = []
        output_node_list = []
        G= self.graph
        # get the fanout node
        # remove the input node
        for node in G.nodes():
            if 'input' in G.nodes[node]['gate']:
                input_node_list.append(node)
        # remove the output node
        for node in G.nodes:
            if G.nodes[node]['output']:
                output_node_list.append(node)
        # print("Debug for Zeng Output")
        # print(output_node_list)
        # print(input_node_list)
        # print(G.nodes())
        # for node_name in output_node_list:
        #     # print(node_name)
        #     print(G.nodes[node_name]['count'])
        fanin_cones_edges = {node: self.edges_in_fanin_cone(G, node, input_node_list) for node in output_node_list}
        # sort from the smallest to the largest
        sorted_output_nodes = sorted(fanin_cones_edges, key=lambda node: len(fanin_cones_edges[node]), reverse=True)
        # print each nodes and its fanin_cones_edges size
        # for node in sorted_output_nodes:
        #     print(node, len(fanin_cones_edges[node]))
        # merged_edges = set()
        # without limiting the fan cone size, let us list all the nodes and their fanin_cones_edges 
        # sorted_op_nodes_list = [] # here is for recording the order of output nodes
        sorted_possible_edge_list = [] # here is for recording the edges which containing the fanin_cones_edges
        # occured_edges_list = []
        for node in sorted_output_nodes:
            # sorted_op_nodes_list.append(node)
            # for each element, we store them as indexed list
            possible_edge = dict(zip(range(len(fanin_cones_edges[node])), fanin_cones_edges[node]))
            # possible_edge_copy = {key: value for key, value in possible_edge.items()}
            # make sure the edges we stored in this fan cone is not inclued in the previous one
            for key in possible_edge:
                possible_edge[key] = [G.nodes[possible_edge[key][0]]['count'], G.nodes[possible_edge[key][1]]['count']]
        # def process_node(node,fanin_cones_edges, G):
        #     # Create possible_edge dictionary for the node
        #     possible_edge = dict(zip(range(len(fanin_cones_edges[node])), fanin_cones_edges[node]))
    
        #     # Update possible_edge with node counts
        #     for key in possible_edge:
        #         possible_edge[key] = [G.nodes[possible_edge[key][0]]['count'], G.nodes[possible_edge[key][1]]['count']]
    
        #     return possible_edge
        
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        # # Use partial to pass the other arguments to process_node
        #     from functools import partial
        #     partial_process_node = partial(process_node, fanin_cones_edges=fanin_cones_edges, G=G)
        #     sorted_possible_edge_list = pool.map(partial_process_node, sorted_output_nodes)

            # remove the repeated edge pairs if it already occured in the previous fan cone
            # and also make sure the key is still starting from 0 to len(removed possible_edge)
            # # for key in possible_edge:
            #     if possible_edge[key] in occured_edges_list:
            #         # del possible_edge[key]
            #         pass
            #     else:
            #         occured_edges_list.append(possible_edge[key])
            # and then, resort the reset possible_edge based on the values
            # index = 0
            # possible_edge_temp = dict()
            # for key in possible_edge:
            #     possible_edge_temp[index] = possible_edge[key]
            #     index += 1
            sorted_possible_edge_list.append(possible_edge)

        # reverse the sorted_possible_edge_list
        # sorted_possible_edge_list.reverse()    
        # print("what is the sorted_possible_edge_list")
        # print(sorted_possible_edge_list)
        # print all the length of these edges
        # for item in sorted_possible_edge_list:
        #     print(len(item))
        return sorted_possible_edge_list    
    
    def get_fan_cone_edges_all_scope(self):
        input_node_list = []
        output_node_list = []
        G= self.graph
        # get the fanout node
        # remove the input node
        for node in G.nodes():
            if 'input' in G.nodes[node]['gate']:
                input_node_list.append(node)
        # remove the output node
        for node in G.nodes:
            if G.nodes[node]['output']:
                output_node_list.append(node)
        # print(output_node_list)
        # for node_name in output_node_list:
        #     # print(node_name)
        #     print(G.nodes[node_name]['count'])
        fanin_cones_edges = {node: self.edges_in_fanin_cone_scope(G, node, input_node_list) for node in output_node_list}
        # sort from the smallest to the largest
        sorted_output_nodes = sorted(fanin_cones_edges, key=lambda node: len(fanin_cones_edges[node]), reverse=True)
        # print each nodes and its fanin_cones_edges size
        # for node in sorted_output_nodes:
        #     print(node, len(fanin_cones_edges[node]))
        # merged_edges = set()
        # without limiting the fan cone size, let us list all the nodes and their fanin_cones_edges 
        # sorted_op_nodes_list = [] # here is for recording the order of output nodes
        sorted_possible_edge_list = [] # here is for recording the edges which containing the fanin_cones_edges
        # occured_edges_list = []
        for node in sorted_output_nodes:
            # sorted_op_nodes_list.append(node)
            # for each element, we store them as indexed list
            possible_edge = dict(zip(range(len(fanin_cones_edges[node])), fanin_cones_edges[node]))
            # possible_edge_copy = {key: value for key, value in possible_edge.items()}
            # make sure the edges we stored in this fan cone is not inclued in the previous one
            for key in possible_edge:
                possible_edge[key] = [G.nodes[possible_edge[key][0]]['count'], G.nodes[possible_edge[key][1]]['count']]
        # def process_node(node,fanin_cones_edges, G):
        #     # Create possible_edge dictionary for the node
        #     possible_edge = dict(zip(range(len(fanin_cones_edges[node])), fanin_cones_edges[node]))
    
        #     # Update possible_edge with node counts
        #     for key in possible_edge:
        #         possible_edge[key] = [G.nodes[possible_edge[key][0]]['count'], G.nodes[possible_edge[key][1]]['count']]
    
        #     return possible_edge
        
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        # # Use partial to pass the other arguments to process_node
        #     from functools import partial
        #     partial_process_node = partial(process_node, fanin_cones_edges=fanin_cones_edges, G=G)
        #     sorted_possible_edge_list = pool.map(partial_process_node, sorted_output_nodes)

            # remove the repeated edge pairs if it already occured in the previous fan cone
            # and also make sure the key is still starting from 0 to len(removed possible_edge)
            # # for key in possible_edge:
            #     if possible_edge[key] in occured_edges_list:
            #         # del possible_edge[key]
            #         pass
            #     else:
            #         occured_edges_list.append(possible_edge[key])
            # and then, resort the reset possible_edge based on the values
            # index = 0
            # possible_edge_temp = dict()
            # for key in possible_edge:
            #     possible_edge_temp[index] = possible_edge[key]
            #     index += 1
            sorted_possible_edge_list.append(possible_edge)

        # reverse the sorted_possible_edge_list
        # sorted_possible_edge_list.reverse()    
        # print("what is the sorted_possible_edge_list")
        # print(sorted_possible_edge_list)
        # print all the length of these edges
        # for item in sorted_possible_edge_list:
        #     print(len(item))
        return sorted_possible_edge_list  




    # check if there is same gs in the kgss
    def check_same_gs(self):
        data = self.data
        gs_list = []
        for item in data:
            gs_list.append(item[2])
            gs_list.append(item[3])
        # print("what is the gs")
        # print(gs_list)
        if len(gs_list) == len(set(map(tuple, gs_list))):
            return False
        else:
            return True
        
    # check if there is same gs in the kgss
    def check_same_gs_scope(self):
        data = self.data
        gs_list = []
        for item in data:
            gs_list.append(item[2])
            # gs_list.append(item[3])
        # print("what is the gs")
        # print(gs_list)
        if len(gs_list) == len(set(map(tuple, gs_list))):
            return False
        else:
            return True
        
    # check all the same gs in the kgss
    # return the same gs
    def check_all_same_gs(self):
        data = self.data
        gs_list = []
        for item in data:
            gs_list.append(item[2])
            gs_list.append(item[3])
        gs_repeat = [item for item, count in Counter(gs_list).items() if count > 1]
        return gs_repeat
    # if there is same gs in the kgss, try to find the same gs within the distance of 2
    # random pick one of them to replace the old one
    def change_same_gs(self):
        data = self.data
        G = self.graph
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        gs_list = []
        for item in data:
            gs_list.append(item[2])
            gs_list.append(item[3])
        # find the repeated gs in gs_list
        gs_repeat = [item for item, count in Counter(gs_list).items() if count > 1]
        # also find the index of the repeated gs in data
        gs_repeat_index = {}
        for item in gs_repeat:
            # here we only pick the largest index
            index_temp = [i for i, x in enumerate(gs_list) if x == item]
            gs_repeat_index[item] = [int(index_temp[-1]/2)]
        # and then, based on the graph, we could know the fs to this gs
        # fs_repeated = []
        for item in gs_repeat_index:
            item_idx = gs_repeat_index[item][0]
            data_temp = data[item_idx]
            # find the fs to this gs
            if item == data_temp[2]:
                fs_temp = data_temp[0]
                gs_repeat_index[item].append(0)
            else:
                fs_temp = data_temp[1]
                gs_repeat_index[item].append(1)
            gs_repeat_index[item].append(fs_temp)
            # get the fs_temp name
            fs_temp_name = G_info_updated[int(fs_temp)]
            gs_temp_name = G_info_updated[int(item)]
            # gs_repeat_index[item].append(fs_temp_name)
            # find the fs_temp_name's fanout
            fs_temp_fanout = []
            fs_temp_fanout_count = []
            for edge in G.edges():
                if edge[0] == fs_temp_name:
                    fs_temp_fanout.append(edge[1])
                    fs_temp_fanout_count.append(G.nodes[edge[1]]['count'])
            # if the size of fs_temp_fanout is larger than 1,
            # we could randomly pick one of them except the gs
            print(fs_temp_fanout)
            if len(fs_temp_fanout) > 1:
                fs_temp_fanout.remove(gs_temp_name)
                # remove the fs_temp_fanout_count which has been in gs_list
                for item1 in gs_list:
                    if item1 in fs_temp_fanout_count:
                        fs_temp_fanout.remove(G_info_updated[int(item1)])
                if len(fs_temp_fanout) == 0:
                    print("we could not choose other gs")
                    return False
                fs_temp_fanout_new = random.choice(fs_temp_fanout)
                # gs_repeat_index[item].append(fs_temp_fanout_new)
                # get the count number of the fs_temp_fanout_new
                fs_temp_fanout_new_count = G.nodes[fs_temp_fanout_new]['count']
                # update the data
                if gs_repeat_index[item][1] == 0:
                    data[item_idx][0+2] = fs_temp_fanout_new_count
                else:
                    data[item_idx][1+2] = fs_temp_fanout_new_count
            else:
                print("the fs_temp_fanout is less than 1")
                return False
        # update the data
        self.data = data
        return True

    def modify_same_gs(self, edge_dict):
        data = self.data
        # print("original data")
        # print(data)
        G = self.graph
        found_mark = 0
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        gs_list = []
        for item in data:
            gs_list.append(item[2])
            gs_list.append(item[3])
        # find the repeated gs in gs_list
        gs_repeat = [item for item, count in Counter(gs_list).items() if count > 1]
        
        # also find the index of the repeated gs in data
        gs_repeat_index = {}
        for item in gs_repeat:
            # here we pick up the repeated gs except for the first one
            index_temp = [i for i, x in enumerate(gs_list) if x == item]
            # gs_repeat_index[item] = [int(index_temp[-1]/2)]
            gs_repeat_temp = [int(x/2) for x in index_temp[1:]]
            gs_repeat_index[item] = [[item] for item in gs_repeat_temp]
        # print("here is gs_repeat_index")
        # print(gs_repeat_index)
        # gs_list = list(set(gs_list))
        # and then, based on the graph, we could know the fs to this gs
        # fs_repeated = []
        for item in gs_repeat_index:
            for idx in range(len(gs_repeat_index[item])):
                item_idx = gs_repeat_index[item][idx][0]
                data_temp = data[item_idx]
                # find the fs to this gs
                if item == data_temp[2]:
                    fs_temp = data_temp[0]
                    gs_repeat_index[item][idx].append(0)
                else:
                    fs_temp = data_temp[1]
                    gs_repeat_index[item][idx].append(1)
                gs_repeat_index[item][idx].append(fs_temp)
                # get the fs_temp name
                fs_temp_name = G_info_updated[int(fs_temp)]
                gs_temp_name = G_info_updated[int(item)]
                # find the fs_temp_name's fanout
                fs_temp_fanout = []
                fs_temp_fanout_count = []
                for edge in G.edges():
                    if edge[0] == fs_temp_name:
                        fs_temp_fanout.append(edge[1])
                        fs_temp_fanout_count.append(str(G.nodes[edge[1]]['count']))
                # if the size of fs_temp_fanout is larger than 1,
                # we could randomly pick one of them except the gs
                # print("here is fs_temp_out")
                # print(fs_temp_fanout)
                # print("here is fs_temp_fanout_count")
                # print(fs_temp_fanout_count)
                fs_temp_fanout_count_copy = fs_temp_fanout_count.copy()
                if len(fs_temp_fanout) > 1:
                    # if gs_temp_name in fs_temp_fanout:
                    fs_temp_idx = fs_temp_fanout.index(gs_temp_name)
                    fs_temp_remove = fs_temp_fanout_count_copy[fs_temp_idx]
                    fs_temp_fanout.remove(gs_temp_name)
                    fs_temp_fanout_count.remove(fs_temp_remove)
                    # print("current fs")
                    # print(fs_temp_fanout)
                    # print("current fs_count")
                    # print(fs_temp_fanout_count)
                    fs_temp_fanout_copy = fs_temp_fanout.copy()
                    # remove the fs_temp_fanout_count which has been in gs_list
                    for item1 in fs_temp_fanout_count:
                        if item1 in gs_list:
                            idx_temp = fs_temp_fanout_count.index(item1)
                            fs_temp_remove1 = fs_temp_fanout_copy[idx_temp]
                            fs_temp_fanout.remove(fs_temp_remove1)
                            if len(fs_temp_fanout) == 0:
                                break
                    # for item1 in gs_list:
                    #     if item1 in fs_temp_fanout_count and G_info_updated[int(item1)] in fs_temp_fanout:
                    #         fs_temp_fanout.remove(G_info_updated[int(item1)])
                    #     if len(fs_temp_fanout) == 0:

                    if len(fs_temp_fanout) == 0:
                        # need to get other changes here 
                        # print("we could not choose other gs")
                        found_mark = 1
                        # return False
                    else:
                        fs_temp_fanout_new = random.choice(fs_temp_fanout)
                        fs_temp_fanout_new_count = G.nodes[fs_temp_fanout_new]['count']
                        # once I find a replacable one, update the gs_list
                        # add one more check here 
                        if [int(fs_temp), fs_temp_fanout_new_count] not in list(edge_dict.values()):
                            found_mark = 1
                        else:
                            gs_list.append(str(fs_temp_fanout_new_count))
                            # gs_list = list(set(gs_list))
                            # print("here I add new gs")
                            # print(str(fs_temp_fanout_new_count))
                            # update the data
                            if str(fs_temp_fanout_new_count) in data[item_idx][0:2]:
                                found_mark = 1
                            else:
                                if gs_repeat_index[item][idx][1] == 0:
                                    data[item_idx][0+2] = str(fs_temp_fanout_new_count)
                                else:
                                    data[item_idx][1+2] = str(fs_temp_fanout_new_count)
                else:
                    # print("the fs_temp_fanout is less than 1")
                    found_mark = 1
                    # return False
                # # found_mark =1 : we need to find another nearby f to replace it 
                if found_mark == 1:
                    chose_flag = True
                    false_count = 0 
                    # get the update gs name list 
                    while chose_flag:
                        key, edge_pair = random.choice(list(edge_dict.items()))
                        if str(edge_pair[1]) in gs_list:
                            chose_flag = True
                        else:
                            if gs_repeat_index[item][idx][1] == 0:
                                if str(edge_pair[1]) == data[item_idx][1] or str(edge_pair[0]) == data[item_idx][3]:
                                    chose_flag = True
                                    continue
                                data[item_idx][0] = str(edge_pair[0])
                                data[item_idx][0+2] = str(edge_pair[1])
                                # print("changed the idx:", idx)
                                # print("selected edges:")
                                # print(edge_pair, key)
                            else:
                                if str(edge_pair[1]) == data[item_idx][0] or str(edge_pair[0]) == data[item_idx][2]:
                                    chose_flag = True
                                    continue
                                data[item_idx][1] = str(edge_pair[0])
                                data[item_idx][1+2] = str(edge_pair[1])
                                # print("changed the idx:", idx)
                                # print("selected edges:")
                                # print(edge_pair, key)
                            # update gs_list once we added it
                            # print("here I add new g")
                            # print(str(edge_pair[1]))
                            gs_list.append(str(edge_pair[1]))
                            # gs_list = list(set(gs_list))
                            chose_flag = False
                            found_mark = 0
                        # if chose_flag == True:
                        #     false_count += 1
                        #     if false_count > 500:
                        #         return None # if we could not find the correct one, we return False
        # if we find there is the problem like f in the g
        # we will randomly switch it with the neigbor value 
        idx_list = []
        for idx in range(len(data)):
            [f1, f2,  g1, g2] = data[idx][0:4]
            if g1 in [f1, f2] or g2 in [f1, f2]:
                idx_list.append(idx)
        print("idx list found", idx_list)
        for idx1 in idx_list:
            print("here is correct")
            idx_temp = random.randint(0, 1)
            unchanged_idx = int(1-idx_temp)
            chose_flag1 = True
            false_count1 = 0
            while chose_flag1:
                _, edge_pair1 = random.choice(list(edge_dict.items()))
                if str(edge_pair1[1]) in gs_list:
                    chose_flag1 = True 
                else:
                    if str(edge_pair1[1]) == data[idx1][unchanged_idx] or str(edge_pair1[0]) == data[idx1][unchanged_idx+2]:
                        chose_flag1 = True
                        continue
                    data[idx1][idx_temp] = str(edge_pair1[0])
                    data[idx1][idx_temp+2] = str(edge_pair1[1])
                    gs_list.remove(data[idx1][unchanged_idx+2])
                    gs_list.append(str(edge_pair1[1]))
                    chose_flag1 = False
                # if chose_flag1 == True:
                #     false_count1 += 1
                #     if false_count1 > 500:
                #         return None # if we could not find the correct one, we return False
 
        # based on the data, convert it to sol vector later
        self.data = data
        print("generated")
        print(data)
        return data
    
    def modify_same_gs_unchanged(self, edge_dict, found_pairs):
        data_temp = self.data
        # remove the found_pairs in the data_temp based on the index
        data_temp = data_temp[:len(data_temp)-len(found_pairs)]
        data = data_temp
        # print("original data")
        # print(data)
        G = self.graph
        found_mark = 0
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        gs_list = []
        gs_unchanged_list = []
        gs_list_all = []
        for item in data:
            gs_list.append(item[2])
            gs_list.append(item[3])
            gs_list_all.append(item[2])
            gs_list_all.append(item[3])
        # also add the found_pairs into the gs_list
        for item in found_pairs:
            # gs_list.append(item[2])
            # gs_list.append(item[3])
            # also store the unchanged gs to future usage
            gs_unchanged_list.append(item[2])
            gs_unchanged_list.append(item[3])
            gs_list_all.append(item[2])
            gs_list_all.append(item[3])
        # find the repeated gs in gs_list
        gs_repeat = [item for item, count in Counter(gs_list).items() if count > 1]
        # add the unchanged gs into the gs_repeat
        for g_temp in gs_list:
            if g_temp in gs_unchanged_list:
                # gs_repeat.append(g_temp)
                # add them into gs_repeat
                gs_repeat += [g_temp]
        
        # also find the index of the repeated gs in data
        gs_repeat_index = {}
        for item in gs_repeat:
            if item not in gs_unchanged_list:
                # here we pick up the repeated gs except for the first one
                index_temp = [i for i, x in enumerate(gs_list) if x == item]
                # gs_repeat_index[item] = [int(index_temp[-1]/2)]
                gs_repeat_temp = [int(x/2) for x in index_temp[1:]]
                gs_repeat_index[item] = [[item] for item in gs_repeat_temp]
            else:
                # keep the last repreated gs unchanged
                index_temp = [i for i, x in enumerate(gs_list_all) if x == item]
                gs_repeat_temp = [int(x/2) for x in index_temp[:-1]]
                gs_repeat_index[item] = [[item] for item in gs_repeat_temp]
        # print("here is gs_repeat_index")
        # print(gs_repeat_index)
        gs_list = gs_list_all # update the gs list to the whole gs list
        # gs_list = list(set(gs_list))
        # and then, based on the graph, we could know the fs to this gs
        # fs_repeated = []
        for item in gs_repeat_index:
            for idx in range(len(gs_repeat_index[item])):
                item_idx = gs_repeat_index[item][idx][0]
                data_temp = data[item_idx]
                # find the fs to this gs
                if item == data_temp[2]:
                    fs_temp = data_temp[0]
                    gs_repeat_index[item][idx].append(0)
                else:
                    fs_temp = data_temp[1]
                    gs_repeat_index[item][idx].append(1)
                gs_repeat_index[item][idx].append(fs_temp)
                # get the fs_temp name
                fs_temp_name = G_info_updated[int(fs_temp)]
                gs_temp_name = G_info_updated[int(item)]
                # find the fs_temp_name's fanout
                fs_temp_fanout = []
                fs_temp_fanout_count = []
                for edge in G.edges():
                    if edge[0] == fs_temp_name:
                        fs_temp_fanout.append(edge[1])
                        fs_temp_fanout_count.append(str(G.nodes[edge[1]]['count']))
                # if the size of fs_temp_fanout is larger than 1,
                # we could randomly pick one of them except the gs
                # print("here is fs_temp_out")
                # print(fs_temp_fanout)
                # print("here is fs_temp_fanout_count")
                # print(fs_temp_fanout_count)
                fs_temp_fanout_count_copy = fs_temp_fanout_count.copy()
                if len(fs_temp_fanout) > 1:
                    
                    fs_temp_idx = fs_temp_fanout.index(gs_temp_name)
                    fs_temp_remove = fs_temp_fanout_count_copy[fs_temp_idx]
                    fs_temp_fanout.remove(gs_temp_name)
                    fs_temp_fanout_count.remove(fs_temp_remove)
                    # print("current fs")
                    # print(fs_temp_fanout)
                    # print("current fs_count")
                    # print(fs_temp_fanout_count)
                    fs_temp_fanout_copy = fs_temp_fanout.copy()
                    # remove the fs_temp_fanout_count which has been in gs_list
                    for item1 in fs_temp_fanout_count:
                        if item1 in gs_list:
                            idx_temp = fs_temp_fanout_count.index(item1)
                            fs_temp_remove1 = fs_temp_fanout_copy[idx_temp]
                            fs_temp_fanout.remove(fs_temp_remove1)
                            if len(fs_temp_fanout) == 0:
                                break
                    # for item1 in gs_list:
                    #     if item1 in fs_temp_fanout_count and G_info_updated[int(item1)] in fs_temp_fanout:
                    #         fs_temp_fanout.remove(G_info_updated[int(item1)])
                    #     if len(fs_temp_fanout) == 0:

                    if len(fs_temp_fanout) == 0:
                        # need to get other changes here 
                        # print("we could not choose other gs")
                        found_mark = 1
                        # return False
                    else:
                        fs_temp_fanout_new = random.choice(fs_temp_fanout)
                        fs_temp_fanout_new_count = G.nodes[fs_temp_fanout_new]['count']
                        # once I find a replacable one, update the gs_list
                        # add one more check here 
                        if [int(fs_temp), fs_temp_fanout_new_count] not in list(edge_dict.values()):
                            found_mark = 1
                        else:
                            gs_list.append(str(fs_temp_fanout_new_count))
                            # gs_list = list(set(gs_list))
                            # print("here I add new gs")
                            # print(str(fs_temp_fanout_new_count))
                            # update the data
                            if str(fs_temp_fanout_new_count) in data[item_idx][0:2]:
                                found_mark = 1
                            else:
                                if gs_repeat_index[item][idx][1] == 0:
                                    data[item_idx][0+2] = str(fs_temp_fanout_new_count)
                                else:
                                    data[item_idx][1+2] = str(fs_temp_fanout_new_count)
                else:
                    # print("the fs_temp_fanout is less than 1")
                    found_mark = 1
                    # return False
                # # found_mark =1 : we need to find another nearby f to replace it 
                if found_mark == 1:
                    chose_flag = True
                    false_count = 0 
                    # get the update gs name list 
                    while chose_flag:
                        key, edge_pair = random.choice(list(edge_dict.items()))
                        if str(edge_pair[1]) in gs_list:
                            chose_flag = True
                        else:
                            if gs_repeat_index[item][idx][1] == 0:
                                if str(edge_pair[1]) == data[item_idx][1] or str(edge_pair[0]) == data[item_idx][3]:
                                    chose_flag = True
                                    continue
                                data[item_idx][0] = str(edge_pair[0])
                                data[item_idx][0+2] = str(edge_pair[1])
                                # print("changed the idx:", idx)
                                # print("selected edges:")
                                # print(edge_pair, key)
                            else:
                                if str(edge_pair[1]) == data[item_idx][0] or str(edge_pair[0]) == data[item_idx][2]:
                                    chose_flag = True
                                    continue
                                data[item_idx][1] = str(edge_pair[0])
                                data[item_idx][1+2] = str(edge_pair[1])
                                # print("changed the idx:", idx)
                                # print("selected edges:")
                                # print(edge_pair, key)
                            # update gs_list once we added it
                            # print("here I add new g")
                            # print(str(edge_pair[1]))
                            gs_list.append(str(edge_pair[1]))
                            # gs_list = list(set(gs_list))
                            chose_flag = False
                            found_mark = 0
                        if chose_flag == True:
                            false_count += 1
                            if false_count > 500:
                                return None # if we could not find the correct one, we return False
        # if we find there is the problem like f in the g
        # we will randomly switch it with the neigbor value 
        idx_list = []
        for idx in range(len(data)):
            [f1, f2,  g1, g2] = data[idx][0:4]
            if g1 in [f1, f2] or g2 in [f1, f2]:
                idx_list.append(idx)
        print("idx list found", idx_list)
        for idx1 in idx_list:
            print("here is correct")
            idx_temp = random.randint(0, 1)
            unchanged_idx = int(1-idx_temp)
            chose_flag1 = True
            false_count1 = 0
            while chose_flag1:
                _, edge_pair1 = random.choice(list(edge_dict.items()))
                if str(edge_pair1[1]) in gs_list:
                    chose_flag1 = True 
                else:
                    if str(edge_pair1[1]) == data[idx1][unchanged_idx] or str(edge_pair1[0]) == data[idx1][unchanged_idx+2]:
                        chose_flag1 = True
                        continue
                    data[idx1][idx_temp] = str(edge_pair1[0])
                    data[idx1][idx_temp+2] = str(edge_pair1[1])
                    gs_list.remove(data[idx1][unchanged_idx+2])
                    gs_list.append(str(edge_pair1[1]))
                    chose_flag1 = False
                if chose_flag1 == True:
                    false_count1 += 1
                    if false_count1 > 500:
                        return None # if we could not find the correct one, we return False
 
        # based on the data, convert it to sol vector later
        # after the modification, append the found pairs
        for item in found_pairs:
            data.append(item)
        self.data = data
        print("generated")
        print(data)
        
        return data
    
    def modify_same_gs_unchanged_omla(self, edge_dict, found_pairs):
        data_temp = self.data
        # remove the found_pairs in the data_temp based on the index
        data_temp = data_temp[:len(data_temp)-len(found_pairs)]
        data = data_temp
        # print("original data")
        # print(data)
        G = self.graph
        found_mark = 0
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        gs_list = []
        gs_unchanged_list = []
        gs_list_all = []
        for item in data:
            gs_list.append(item[1])
            # gs_list.append(item[3])
            gs_list_all.append(item[1])
            # gs_list_all.append(item[3])
        # also add the found_pairs into the gs_list
        for item in found_pairs:
            # gs_list.append(item[2])
            # gs_list.append(item[3])
            # also store the unchanged gs to future usage
            gs_unchanged_list.append(item[1])
            # gs_unchanged_list.append(item[3])
            gs_list_all.append(item[1])
            # gs_list_all.append(item[3])
        # find the repeated gs in gs_list
        # here is to save +/-
        # gs_list_saved = gs_list.copy()
        # gs_list_all_saved = gs_list_all.copy()
        # gs_unchanged_list_saved = gs_unchanged_list.copy()

        # gs_list = [abs(x) for x in gs_list]
        # gs_list_all = [abs(x) for x in gs_list_all]
        # gs_unchanged_list = [abs(x) for x in gs_unchanged_list]

        gs_repeat = [item for item, count in Counter(gs_list).items() if count > 1]
        # add the unchanged gs into the gs_repeat
        for g_temp in gs_list:
            if g_temp in gs_unchanged_list:
                # gs_repeat.append(g_temp)
                # add them into gs_repeat
                gs_repeat += [g_temp]
        
        # also find the index of the repeated gs in data
        gs_repeat_index = {}
        for item in gs_repeat:
            if item not in gs_unchanged_list:
                # here we pick up the repeated gs except for the first one
                index_temp = [i for i, x in enumerate(gs_list) if x == item]
                # gs_repeat_index[item] = [int(index_temp[-1]/2)]
                gs_repeat_temp = [int(x) for x in index_temp[1:]]
                gs_repeat_index[item] = [[item] for item in gs_repeat_temp]
            else:
                # keep the last repreated gs unchanged
                index_temp = [i for i, x in enumerate(gs_list_all) if x == item]
                gs_repeat_temp = [int(x) for x in index_temp[:-1]]
                gs_repeat_index[item] = [[item] for item in gs_repeat_temp]
        # print("here is gs_repeat_index")
        # print(gs_repeat_index)
        gs_list = gs_list_all # update the gs list to the whole gs list
        # gs_list = list(set(gs_list))
        # and then, based on the graph, we could know the fs to this gs
        # fs_repeated = []
        for item in gs_repeat_index:
            for idx in range(len(gs_repeat_index[item])):
                item_idx = gs_repeat_index[item][idx][0]
                data_temp = data[item_idx]
                # find the fs to this gs
                # if item == data_temp[1]:
                fs_temp = data_temp[0]
                gs_repeat_index[item][idx].append(0)
                # else:
                #     fs_temp = data_temp[1]
                #     gs_repeat_index[item][idx].append(1)
                gs_repeat_index[item][idx].append(fs_temp)
                # get the fs_temp name
                fs_temp_name = G_info_updated[int(fs_temp)]
                gs_temp_name = G_info_updated[int(item)]
                # find the fs_temp_name's fanout
                fs_temp_fanout = []
                fs_temp_fanout_count = []
                for edge in G.edges():
                    if edge[0] == fs_temp_name:
                        fs_temp_fanout.append(edge[1])
                        fs_temp_fanout_count.append(str(G.nodes[edge[1]]['count']))
                # if the size of fs_temp_fanout is larger than 1,
                # we could randomly pick one of them except the gs
                # print("here is fs_temp_out")
                # print(fs_temp_fanout)
                # print("here is fs_temp_fanout_count")
                # print(fs_temp_fanout_count)
                fs_temp_fanout_count_copy = fs_temp_fanout_count.copy()
                if len(fs_temp_fanout) > 1:
                    
                    fs_temp_idx = fs_temp_fanout.index(gs_temp_name)
                    fs_temp_remove = fs_temp_fanout_count_copy[fs_temp_idx]
                    fs_temp_fanout.remove(gs_temp_name)
                    fs_temp_fanout_count.remove(fs_temp_remove)
                    # print("current fs")
                    # print(fs_temp_fanout)
                    # print("current fs_count")
                    # print(fs_temp_fanout_count)
                    fs_temp_fanout_copy = fs_temp_fanout.copy()
                    # remove the fs_temp_fanout_count which has been in gs_list
                    for item1 in fs_temp_fanout_count:
                        if item1 in gs_list:
                            idx_temp = fs_temp_fanout_count.index(item1)
                            fs_temp_remove1 = fs_temp_fanout_copy[idx_temp]
                            fs_temp_fanout.remove(fs_temp_remove1)
                            if len(fs_temp_fanout) == 0:
                                break
                    # for item1 in gs_list:
                    #     if item1 in fs_temp_fanout_count and G_info_updated[int(item1)] in fs_temp_fanout:
                    #         fs_temp_fanout.remove(G_info_updated[int(item1)])
                    #     if len(fs_temp_fanout) == 0:

                    if len(fs_temp_fanout) == 0:
                        # need to get other changes here 
                        # print("we could not choose other gs")
                        found_mark = 1
                        # return False
                    else:
                        fs_temp_fanout_new = random.choice(fs_temp_fanout)
                        fs_temp_fanout_new_count = G.nodes[fs_temp_fanout_new]['count']
                        # once I find a replacable one, update the gs_list
                        # add one more check here 
                        if [int(fs_temp), fs_temp_fanout_new_count] not in list(edge_dict.values()):
                            found_mark = 1
                        else:
                            gs_list.append(str(fs_temp_fanout_new_count))
                            # gs_list = list(set(gs_list))
                            # print("here I add new gs")
                            # print(str(fs_temp_fanout_new_count))
                            # update the data
                            if str(fs_temp_fanout_new_count) == data[item_idx][0]:
                                found_mark = 1
                            else:
                                if gs_repeat_index[item][idx][1] == 0:
                                    # if gs_list_all_saved[item_idx] < 0:
                                    data[item_idx][1] = str(fs_temp_fanout_new_count)
                                    # else:
                                    #     data[item_idx][1] = str(fs_temp_fanout_new_count)
                                else:
                                    found_mark = 1
                else:
                    # print("the fs_temp_fanout is less than 1")
                    found_mark = 1
                    # return False
                # # found_mark =1 : we need to find another nearby f to replace it 
                if found_mark == 1:
                    chose_flag = True
                    false_count = 0 
                    # get the update gs name list 
                    while chose_flag:
                        key, edge_pair = random.choice(list(edge_dict.items()))
                        if str(edge_pair[1]) in gs_list:
                            chose_flag = True
                        else:
                            if str(edge_pair[1]) == data[item_idx][0] or str(edge_pair[0]) == data[item_idx][1]:
                                chose_flag = True
                                continue
                            data[item_idx][0] = str(edge_pair[0])
                            data[item_idx][1] = str(edge_pair[1])
                                # print("changed the idx:", idx)
                                # print("selected edges:")
                                # print(edge_pair, key)
                            # else:
                            #     if str(edge_pair[1]) == data[item_idx][0] or str(edge_pair[0]) == data[item_idx][2]:
                            #         chose_flag = True
                            #         continue
                            #     data[item_idx][1] = str(edge_pair[0])
                            #     data[item_idx][1+2] = str(edge_pair[1])
                            #     # print("changed the idx:", idx)
                            #     # print("selected edges:")
                            #     # print(edge_pair, key)
                            # update gs_list once we added it
                            # print("here I add new g")
                            # print(str(edge_pair[1]))
                            gs_list.append(str(edge_pair[1]))
                            # gs_list = list(set(gs_list))
                            chose_flag = False
                            found_mark = 0
                        if chose_flag == True:
                            false_count += 1
                            if false_count > 500:
                                return None # if we could not find the correct one, we return False
        # if we find there is the problem like f in the g
        # we will randomly switch it with the neigbor value 
        # idx_list = []
        # for idx in range(len(data)):
        #     [f1, g1] = data[idx][0:2]
        #     if g1 in [f1, f2] or g2 in [f1, f2]:
        #         idx_list.append(idx)
        # print("idx list found", idx_list)
        # for idx1 in idx_list:
        #     print("here is correct")
        #     idx_temp = random.randint(0, 1)
        #     unchanged_idx = int(1-idx_temp)
        #     chose_flag1 = True
        #     false_count1 = 0
        #     while chose_flag1:
        #         _, edge_pair1 = random.choice(list(edge_dict.items()))
        #         if str(edge_pair1[1]) in gs_list:
        #             chose_flag1 = True 
        #         else:
        #             if str(edge_pair1[1]) == data[idx1][unchanged_idx] or str(edge_pair1[0]) == data[idx1][unchanged_idx+2]:
        #                 chose_flag1 = True
        #                 continue
        #             data[idx1][idx_temp] = str(edge_pair1[0])
        #             data[idx1][idx_temp+2] = str(edge_pair1[1])
        #             gs_list.remove(data[idx1][unchanged_idx+2])
        #             gs_list.append(str(edge_pair1[1]))
        #             chose_flag1 = False
        #         if chose_flag1 == True:
        #             false_count1 += 1
        #             if false_count1 > 500:
        #                 return None # if we could not find the correct one, we return False
 
        # based on the data, convert it to sol vector later
        # after the modification, append the found pairs
        for item in found_pairs:
            data.append(item)
        self.data = data
        print("generated")
        print(data)
        
        return data

    def modify_same_gs_unchanged_scope(self, edge_dict, found_pairs):
        data_temp = self.data
        # remove the found_pairs in the data_temp based on the index
        data_temp = data_temp[:len(data_temp)-len(found_pairs)]
        data = data_temp
        # print("original data")
        # print(data)
        G = self.graph
        found_mark = 0
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        gs_list = []
        gs_unchanged_list = []
        gs_list_all = []
        for item in data:
            gs_list.append(item[2])
            # gs_list.append(item[3])
            gs_list_all.append(item[2])
            # gs_list_all.append(item[3])
        # also add the found_pairs into the gs_list
        for item in found_pairs:
            # gs_list.append(item[2])
            # gs_list.append(item[3])
            # also store the unchanged gs to future usage
            gs_unchanged_list.append(item[2])
            # gs_unchanged_list.append(item[3])
            gs_list_all.append(item[2])
            # gs_list_all.append(item[3])
        # find the repeated gs in gs_list
        gs_repeat = [item for item, count in Counter(gs_list).items() if count > 1]
        # add the unchanged gs into the gs_repeat
        for g_temp in gs_list:
            if g_temp in gs_unchanged_list:
                # gs_repeat.append(g_temp)
                # add them into gs_repeat
                gs_repeat += [g_temp]
        
        # also find the index of the repeated gs in data
        gs_repeat_index = {}
        for item in gs_repeat:
            if item not in gs_unchanged_list:
                # here we pick up the repeated gs except for the first one
                index_temp = [i for i, x in enumerate(gs_list) if x == item]
                # gs_repeat_index[item] = [int(index_temp[-1]/2)]
                gs_repeat_temp = [int(x) for x in index_temp[1:]]
                gs_repeat_index[item] = [[item] for item in gs_repeat_temp]
            else:
                # keep the last repreated gs unchanged
                index_temp = [i for i, x in enumerate(gs_list_all) if x == item]
                gs_repeat_temp = [int(x) for x in index_temp[:-1]]
                gs_repeat_index[item] = [[item] for item in gs_repeat_temp]
        # print("here is gs_repeat_index")
        # print(gs_repeat_index)
        gs_list = gs_list_all # update the gs list to the whole gs list
        # gs_list = list(set(gs_list))
        # and then, based on the graph, we could know the fs to this gs
        # fs_repeated = []
        # print("here is gs_list before")
        # print(data_temp)
        # print(gs_repeat)
        # print(gs_repeat_index)
        for item in gs_repeat_index:
            for idx in range(len(gs_repeat_index[item])):
                item_idx = gs_repeat_index[item][idx][0]
                data_temp = data[item_idx]
                # find the fs to this gs
                if item == data_temp[2]:
                    fs_temp = data_temp[0]
                    gs_repeat_index[item][idx].append(0)
                else:
                    continue
                    # fs_temp = data_temp[1]
                    # gs_repeat_index[item][idx].append(1)
                gs_repeat_index[item][idx].append(fs_temp)
                # get the fs_temp name
                fs_temp_name = G_info_updated[int(fs_temp)]
                gs_temp_name = G_info_updated[int(item)]
                # find the fs_temp_name's fanout
                fs_temp_fanout = []
                fs_temp_fanout_count = []
                for edge in G.edges():
                    if edge[0] == fs_temp_name:
                        fs_temp_fanout.append(edge[1])
                        fs_temp_fanout_count.append(str(G.nodes[edge[1]]['count']))
                # if the size of fs_temp_fanout is larger than 1,
                # we could randomly pick one of them except the gs
                # print("here is fs_temp_out")
                # print(fs_temp_fanout)
                # print("here is fs_temp_fanout_count")
                # print(fs_temp_fanout_count)
                fs_temp_fanout_count_copy = fs_temp_fanout_count.copy()
                if len(fs_temp_fanout) > 1:
                    if gs_temp_name in fs_temp_fanout:
                        fs_temp_idx = fs_temp_fanout.index(gs_temp_name)
                        fs_temp_remove = fs_temp_fanout_count_copy[fs_temp_idx]
                        fs_temp_fanout.remove(gs_temp_name)
                        fs_temp_fanout_count.remove(fs_temp_remove)
                    # print("current fs")
                    # print(fs_temp_fanout)
                    # print("current fs_count")
                    # print(fs_temp_fanout_count)
                    fs_temp_fanout_copy = fs_temp_fanout.copy()
                    # remove the fs_temp_fanout_count which has been in gs_list
                    for item1 in fs_temp_fanout_count:
                        if item1 in gs_list:
                            idx_temp = fs_temp_fanout_count.index(item1)
                            fs_temp_remove1 = fs_temp_fanout_copy[idx_temp]
                            fs_temp_fanout.remove(fs_temp_remove1)
                            if len(fs_temp_fanout) == 0:
                                print("here we break")
                                break
                    # for item1 in gs_list:
                    #     if item1 in fs_temp_fanout_count and G_info_updated[int(item1)] in fs_temp_fanout:
                    #         fs_temp_fanout.remove(G_info_updated[int(item1)])
                    #     if len(fs_temp_fanout) == 0:

                    if len(fs_temp_fanout) == 0:
                        # need to get other changes here 
                        # print("we could not choose other gs")
                        found_mark = 1
                        # return False
                    else:
                        fs_temp_fanout_new = random.choice(fs_temp_fanout)
                        fs_temp_fanout_new_count = G.nodes[fs_temp_fanout_new]['count']
                        # once I find a replacable one, update the gs_list
                        # add one more check here 
                        if [int(fs_temp), fs_temp_fanout_new_count] not in list(edge_dict.values()):
                            found_mark = 1
                        else:
                            gs_list.append(str(fs_temp_fanout_new_count))
                            # gs_list = list(set(gs_list))
                            # print("here I add new gs")
                            # print(str(fs_temp_fanout_new_count))
                            # update the data
                            if str(fs_temp_fanout_new_count) in data[item_idx][0:2]:
                                found_mark = 1
                            else:
                                if gs_repeat_index[item][idx][1] == 0:
                                    # print("we enter here:", item_idx)
                                    data[item_idx][0+2] = str(fs_temp_fanout_new_count)
                                else:
                                    data[item_idx][1+2] = str(fs_temp_fanout_new_count)
                else:
                    # print("the fs_temp_fanout is less than 1")
                    found_mark = 1
                    # return False
                # # found_mark =1 : we need to find another nearby f to replace it 
                if found_mark == 1:
                    chose_flag = True
                    false_count = 0 
                    # get the update gs name list 
                    while chose_flag:
                        key, edge_pair = random.choice(list(edge_dict.items()))
                        if str(edge_pair[1]) in gs_list:
                            chose_flag = True
                        else:
                            if gs_repeat_index[item][idx][1] == 0:
                                # print("here is changed")
                                if str(edge_pair[1]) == data[item_idx][1] or str(edge_pair[0]) == data[item_idx][3]:
                                    chose_flag = True
                                    continue
                                data[item_idx][0] = str(edge_pair[0])
                                data[item_idx][0+2] = str(edge_pair[1])
                                # print("changed the idx:", idx)
                                # print("selected edges:")
                                # print(edge_pair, key)
                            else:
                                # print("here is changed2")
                                if str(edge_pair[1]) == data[item_idx][0] or str(edge_pair[0]) == data[item_idx][2]:
                                    chose_flag = True
                                    continue
                                data[item_idx][1] = str(edge_pair[0])
                                data[item_idx][1+2] = str(edge_pair[1])
                                # print("changed the idx:", idx)
                                # print("selected edges:")
                                # print(edge_pair, key)
                            # update gs_list once we added it
                            # print("here I add new g")
                            # print(str(edge_pair[1]))
                            gs_list.append(str(edge_pair[1]))
                            # gs_list = list(set(gs_list))
                            chose_flag = False
                            found_mark = 0
                        if chose_flag == True:
                            false_count += 1
                            if false_count > 500:
                                return None # if we could not find the correct one, we return False
        # if we find there is the problem like f in the g
        # we will randomly switch it with the neigbor value 
        # print("here is temp data")
        # print(data)
        idx_list = []
        for idx in range(len(data)):
            [f1, f2,  g1, g2] = data[idx][0:4]
            if g1 in [f1, f2]:# or g2 in [f1, f2]:
                idx_list.append(idx)
        print("idx list found", idx_list)
        for idx1 in idx_list:
            print("here is correct")
            idx_temp = 0
            unchanged_idx = int(1-idx_temp)
            chose_flag1 = True
            false_count1 = 0
            while chose_flag1:
                _, edge_pair1 = random.choice(list(edge_dict.items()))
                if str(edge_pair1[1]) in gs_list:
                    chose_flag1 = True 
                else:
                    if str(edge_pair1[1]) == data[idx1][unchanged_idx] or str(edge_pair1[0]) == data[idx1][unchanged_idx+2]:
                        chose_flag1 = True
                        continue
                    data[idx1][idx_temp] = str(edge_pair1[0])
                    data[idx1][idx_temp+2] = str(edge_pair1[1])
                    # gs_list.remove(data[idx1][unchanged_idx+2])
                    gs_list.append(str(edge_pair1[1]))
                    chose_flag1 = False
                if chose_flag1 == True:
                    false_count1 += 1
                    if false_count1 > 500:
                        return None # if we could not find the correct one, we return False
 
        # based on the data, convert it to sol vector later
        # after the modification, append the found pairs
        for item in found_pairs:
            data.append(item)
        self.data = data
        print("generated")
        print(data)
        # check same gs
        # gs_list = []
        # for item in data:
        #     gs_list.append(item[2])
        #     # gs_list.append(item[3])
        # # find the repeated gs in gs_list
        # gs_repeat = [item for item, count in Counter(gs_list).items() if count > 1]
        # print("here is gs_repeat")
        # print(gs_repeat)
        return data
    

    def modify_same_gs_selected(self, edge_dict):
        data = self.data
        # print("original data")
        # print(data)
        G = self.graph
        found_mark = 0
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        gs_list = []
        for item in data:
            gs_list.append(item[2])
            gs_list.append(item[3])
        # find the repeated gs in gs_list
        gs_repeat = [item for item, count in Counter(gs_list).items() if count > 1]
        
        # also find the index of the repeated gs in data
        gs_repeat_index = {}
        for item in gs_repeat:
            # here we pick up the repeated gs except for the first one
            index_temp = [i for i, x in enumerate(gs_list) if x == item]
            # gs_repeat_index[item] = [int(index_temp[-1]/2)]
            gs_repeat_temp = [int(x/2) for x in index_temp[1:]]
            gs_repeat_index[item] = [[item] for item in gs_repeat_temp]
        # print("here is gs_repeat_index")
        # print(gs_repeat_index)
        # gs_list = list(set(gs_list))
        # and then, based on the graph, we could know the fs to this gs
        # fs_repeated = []
        for item in gs_repeat_index:
            for idx in range(len(gs_repeat_index[item])):
                item_idx = gs_repeat_index[item][idx][0]
                data_temp = data[item_idx]
                # find the fs to this gs
                if item == data_temp[2]:
                    fs_temp = data_temp[0]
                    gs_repeat_index[item][idx].append(0)
                else:
                    fs_temp = data_temp[1]
                    gs_repeat_index[item][idx].append(1)
                gs_repeat_index[item][idx].append(fs_temp)
                # get the fs_temp name
                fs_temp_name = G_info_updated[int(fs_temp)]
                gs_temp_name = G_info_updated[int(item)]
                # find the fs_temp_name's fanout
                fs_temp_fanout = []
                fs_temp_fanout_count = []
                for edge in G.edges():
                    if edge[0] == fs_temp_name:
                        if [G.nodes[edge[0]]['count'], G.nodes[edge[1]]['count']] not in list(edge_dict.values()):
                            continue
                        fs_temp_fanout.append(edge[1])
                        fs_temp_fanout_count.append(str(G.nodes[edge[1]]['count']))
                # if the size of fs_temp_fanout is larger than 1,
                # we could randomly pick one of them except the gs
                # print("here is fs_temp_out")
                # print(fs_temp_fanout)
                # print("here is fs_temp_fanout_count")
                # print(fs_temp_fanout_count)
                fs_temp_fanout_count_copy = fs_temp_fanout_count.copy()
                if len(fs_temp_fanout) > 1:
                    
                    fs_temp_idx = fs_temp_fanout.index(gs_temp_name)
                    fs_temp_remove = fs_temp_fanout_count_copy[fs_temp_idx]
                    fs_temp_fanout.remove(gs_temp_name)
                    fs_temp_fanout_count.remove(fs_temp_remove)
                    # print("current fs")
                    # print(fs_temp_fanout)
                    # print("current fs_count")
                    # print(fs_temp_fanout_count)
                    fs_temp_fanout_copy = fs_temp_fanout.copy()
                    # remove the fs_temp_fanout_count which has been in gs_list
                    for item1 in fs_temp_fanout_count:
                        if item1 in gs_list:
                            idx_temp = fs_temp_fanout_count.index(item1)
                            fs_temp_remove1 = fs_temp_fanout_copy[idx_temp]
                            fs_temp_fanout.remove(fs_temp_remove1)
                            if len(fs_temp_fanout) == 0:
                                break
                    # for item1 in gs_list:
                    #     if item1 in fs_temp_fanout_count and G_info_updated[int(item1)] in fs_temp_fanout:
                    #         fs_temp_fanout.remove(G_info_updated[int(item1)])
                    #     if len(fs_temp_fanout) == 0:

                    if len(fs_temp_fanout) == 0:
                        # need to get other changes here 
                        # print("we could not choose other gs")
                        found_mark = 1
                        # return False
                    else:
                        fs_temp_fanout_new = random.choice(fs_temp_fanout)
                        fs_temp_fanout_new_count = G.nodes[fs_temp_fanout_new]['count']
                        # add one more check here 
                        if [int(fs_temp), fs_temp_fanout_new_count] not in list(edge_dict.values()):
                            found_mark = 1
                        else:
                            # once I find a replacable one, update the gs_list
                            gs_list.append(str(fs_temp_fanout_new_count))
                            # gs_list = list(set(gs_list))
                            print("here I add new gs")
                            print(str(fs_temp_fanout_new_count))
                            # update the data
                            # found_mark = 1 #debug
                            if str(fs_temp_fanout_new_count) in data[item_idx][0:2]:
                                found_mark = 1
                            else:
                                # found_mark = 1
                                if gs_repeat_index[item][idx][1] == 0:
                                    data[item_idx][0+2] = str(fs_temp_fanout_new_count)
                                else:
                                    data[item_idx][1+2] = str(fs_temp_fanout_new_count)
                else:
                    # print("the fs_temp_fanout is less than 1")
                    found_mark = 1
                    # return False
                # # found_mark =1 : we need to find another nearby f to replace it 
                if found_mark == 1:
                    chose_flag = True
                    # get the update gs name list 
                    while chose_flag:
                        key, edge_pair = random.choice(list(edge_dict.items()))
                        if str(edge_pair[1]) in gs_list:
                            chose_flag = True
                        else:
                            if gs_repeat_index[item][idx][1] == 0:
                                if str(edge_pair[1]) == data[item_idx][1] or str(edge_pair[0]) == data[item_idx][3]:
                                    chose_flag = True
                                    continue
                                data[item_idx][0] = str(edge_pair[0])
                                data[item_idx][0+2] = str(edge_pair[1])
                                print("changed the idx:", idx)
                                print("selected edges:")
                                print(edge_pair, key)
                            else:
                                if str(edge_pair[1]) == data[item_idx][0] or str(edge_pair[0]) == data[item_idx][2]:
                                    chose_flag = True
                                    continue
                                data[item_idx][1] = str(edge_pair[0])
                                data[item_idx][1+2] = str(edge_pair[1])
                                print("changed the idx:", idx)
                                print("selected edges:")
                                print(edge_pair, key)
                            # update gs_list once we added it
                            # print("here I add new g")
                            # print(str(edge_pair[1]))
                            gs_list.append(str(edge_pair[1]))
                            # gs_list = list(set(gs_list))
                            chose_flag = False
                            found_mark = 0
        # if we find there is the problem like f in the g
        # we will randomly switch it with the neigbor value 
        idx_list = []
        for idx in range(len(data)):
            [f1, f2,  g1, g2] = data[idx][0:4]
            if g1 in [f1, f2] or g2 in [f1, f2]:
                idx_list.append(idx)
        print("idx list found", idx_list)
        for idx1 in idx_list:
            print("here is correct")
            idx_temp = random.randint(0, 1)
            unchanged_idx = int(1-idx_temp)
            chose_flag1 = True
            while chose_flag1:
                _, edge_pair1 = random.choice(list(edge_dict.items()))
                if str(edge_pair1[1]) in gs_list:
                    chose_flag1 = True 
                    continue
                else:
                    if str(edge_pair1[1]) == data[idx1][unchanged_idx] or str(edge_pair1[0]) == data[item_idx][unchanged_idx+2]:
                        chose_flag1 = True
                        continue
                    data[idx1][idx_temp] = str(edge_pair1[0])
                    data[idx1][idx_temp+2] = str(edge_pair1[1])
                    gs_list.remove(data[idx1][unchanged_idx+2])
                    gs_list.append(str(edge_pair[1]))
                    chose_flag1 = False
 
        # based on the data, convert it to sol vector later
        self.data = data
        print("generated")
        print(data)
        return data



    # check if there is same fs in the kgss
    def check_same_fs(self):
        data = self.data
        fs_list = []
        for item in data:
            fs_list.append(item[0])
            fs_list.append(item[1])
        if len(fs_list) == len(set(map(tuple, fs_list))):
            return False
        else:
            return True

    # check if each f node is in the g nodes
    def check_f_in_g(self):
        data = self.data
        G = self.graph
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        check_result = []
        for item in data:
            [f1, f2, g1, g2] = [int(i) for i in item[0:4]]
            # print(f1, f2, g1, g2)
            # based on the count number, find the node name
            f1 = G_info_updated[f1]
            f2 = G_info_updated[f2]
            g1 = G_info_updated[g1]
            g2 = G_info_updated[g2]
            # print(f1, f2, g1, g2)
            if g1 not in [f1, f2] and g2 not in [f1, f2]:
                check_result.append(0)
            else:
                check_result.append(1)
        if sum(check_result) > 0:
            return True
        else:
            return False
    
    # check if each f node is in the g nodes
    def check_f_in_g_scope(self):
        data = self.data
        G = self.graph
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        check_result = []
        for item in data:
            [f1, f2, g1, g2] = [int(i) for i in item[0:4]]
            # print(f1, f2, g1, g2)
            # based on the count number, find the node name
            f1 = G_info_updated[f1]
            f2 = G_info_updated[f2]
            g1 = G_info_updated[g1]
            g2 = G_info_updated[g2]
            # print(f1, f2, g1, g2)
            if g1 not in [f1, f2]:# and g2 not in [f1, f2]:
                check_result.append(0)
            else:
                check_result.append(1)
        if sum(check_result) > 0:
            return True
        else:
            return False
    
    def check_cycles(self):
        # G = self.graph
        # here we provide the graph with the updated edge
        G = self.graph
        data = self.data
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        check_result = []
        for item in data:
            [f1, f2, g1, g2] = [int(i) for i in item[0:4]]
            # print(f1, f2, g1, g2)
            # based on the count number, find the node name
            f1 = G_info_updated[f1]
            f2 = G_info_updated[f2]
            g1 = G_info_updated[g1]
            g2 = G_info_updated[g2]
            # print(f1, f2, g1, g2)
            # check if there is a cycle after adding the edge
            R1 = nx.has_path(G, g1, f2)
            # print("what the value of R1:", R1)
            R2 = nx.has_path(G, g2, f1)
            # print("what the value of R2:", R2)
            if (g1 != g2) and (f1 != f2) and not R1 and not R2:
                check_result.append(0)
                G.add_edge(f2,g1)
                G.add_edge(f1,g2)
            else:
                check_result.append(1)
        # print("check_result:")
        # print(check_result)
        if sum(check_result) == 0: #correct
            return False
        else:
            return True
    
    def check_cycles_scope(self):
        # G = self.graph
        # here we provide the graph with the updated edge
        G = self.graph
        data = self.data
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        check_result = []
        for item in data:
            [f1, f2, g1, g2] = [int(i) for i in item[0:4]]
            # print(f1, f2, g1, g2)
            # based on the count number, find the node name
            f1 = G_info_updated[f1]
            f2 = G_info_updated[f2]
            g1 = G_info_updated[g1]
            g2 = G_info_updated[g2]
            # print(f1, f2, g1, g2)
            # check if there is a cycle after adding the edge
            R1 = nx.has_path(G, g1, f2)
            # print("what the value of R1:", R1)
            R2 = False
            # print("what the value of R2:", R2)
            if (g1 != g2) and (f1 != f2) and not R1 and not R2:
                check_result.append(0)
                G.add_edge(f2,g1)
                # G.add_edge(f1,g2)
            else:
                check_result.append(1)
        # print("check_result:")
        # print(check_result)
        if sum(check_result) == 0: #correct
            return False
        else:
            return True

    # check if there will introduce cycle after adding 2 pairs of edges
    def check_cycle_pair(self, edge_dict):
        # G = self.graph
        # here we provide the graph with the updated edge
        data = self.data
        G = self.graph
        G_copy = self.graph
        G_info = dict(list(G_copy.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        check_result = []
        all_gs = []
        # for item in data:
        #     [f1, f2, g1, g2] = [int(i) for i in item[0:4]]
        #     g1 = G_info_updated[g1]
        #     g2 = G_info_updated[g2]
        #     all_gs.append(g1)
        #     all_gs.append(g2)
        # I only want to keep the used gs in order to give more space to use
        for item in data:
            [f1, f2, g1, g2] = [int(i) for i in item[0:4]]
            # change_mark = 0
            # print(f1, f2, g1, g2)
            # based on the count number, find the node name
            f1 = G_info_updated[f1]
            f2 = G_info_updated[f2]
            g1 = G_info_updated[g1]
            g2 = G_info_updated[g2]
            # print(f1, f2, g1, g2)
            # check if there is a cycle after adding the edge
            R1 = nx.has_path(G, g1, f2)
            # print("what the value of R1:", R1)
            R2 = nx.has_path(G, g2, f1)
            # print("what the value of R2:", R2)
            if (g1 != g2) and (f1 != f2) and not R1 and not R2:
                check_result.append(0)
                G.add_edge(f2, g1)
                G.add_edge(f1, g2)
                all_gs.append(g1)
                all_gs.append(g2)
            else:
                if R1 and not R2:
                    check_result.append(1)
                    all_gs.append(g2)
                elif not R1 and R2:
                    check_result.append(2)
                    all_gs.append(g1)
                else:
                    check_result.append(3)
        if sum(check_result) == 0: #correct
            # no need to change return the data
            self.data = data
            return data
        else:
            # current the graph has been added edges which we need for correct localities
            # we need to change to make sure there is no loop here
            idx_list = []
            # print(check_result)
            for idx in range(len(check_result)):
                # here means that R1 has a path, we need to replace it with f1 successors
                # find the f1, f2, g1, g2
                if check_result[idx] == 1:
                    [f1_num, f2_num, g1_num, g2_num] = data[idx][0:4]
                    f1 = G_info_updated[int(f1_num)]
                    f2 = G_info_updated[int(f2_num)]
                    g1 = G_info_updated[int(g1_num)]
                    g2 = G_info_updated[int(g2_num)]
                    random_sel = 0
                    f1_fanout = list(G_copy.successors(f1))
                    f1_fanout_copy = list(G_copy.successors(f1))
                    for item in f1_fanout_copy:
                        if [G.nodes[f1]['count'], G.nodes[item]['count']] not in list(edge_dict.values()):
                            f1_fanout.remove(item)
                    # here we need to change to find other find out of f1
                    if len(f1_fanout) == 1: # here means of g1
                        random_sel = 1
                        all_gs.remove(g2)
                    else:
                        # remove the f1's current fanout and see if there is unrepeated gs
                        if g1 in f1_fanout:
                            f1_fanout.remove(g1)
                        for item in all_gs:
                            if item in f1_fanout:
                                f1_fanout.remove(item)
                        if len(f1_fanout) > 1: # check if there is loop for R1
                            g1_new_temp = random.choice(f1_fanout)
                            R1_new = nx.has_path(G, g1_new_temp, f2)
                            if not R1_new:
                                # change the g1 to g1_new
                                # add the edge to G
                                g1_new_num = G_copy.nodes[g1_new_temp]['count']
                                data[idx][2] = str(g1_new_num)
                                all_gs.append(g1_new_temp)
                                G.add_edge(f2, g1_new_temp)
                                G.add_edge(f1, g2)
                                # random_sel = 1
                                # all_gs.append(g2)
                            else:
                                # need to random selected
                                random_sel = 1
                                all_gs.remove(g2)
                        else:
                            random_sel = 1
                            all_gs.remove(g2)
                    # random select also need to be constricted
                    # 1) not f in gs; 2) no repeated gs; 3) no path in the G
                    if random_sel == 1:
                        idx_list.append(idx)
                elif check_result[idx] == 2:
                    [f1_num, f2_num, g1_num, g2_num] = data[idx][0:4]
                    f1 = G_info_updated[int(f1_num)]
                    f2 = G_info_updated[int(f2_num)]
                    g1 = G_info_updated[int(g1_num)]
                    g2 = G_info_updated[int(g2_num)]
                    random_sel = 0
                    f2_fanout = list(G_copy.successors(f2))
                    f2_fanout_copy = list(G_copy.successors(f2))
                    for item in f2_fanout_copy:
                        if [G.nodes[f2]['count'], G.nodes[item]['count']] not in list(edge_dict.values()):
                            f2_fanout.remove(item)
                    if len(f2_fanout) == 1: # here means of g1
                        random_sel = 1
                        all_gs.remove(g1)
                    else:
                        # remove the f1's current fanout and see if there is unrepeated gs
                        if g2 in f2_fanout:
                           f2_fanout.remove(g2)
                        for item in all_gs:
                            if item in f2_fanout:
                               f2_fanout.remove(item)
                        if len(f2_fanout) > 1: # check if there is loop for R1
                            g2_new_temp = random.choice(f2_fanout)
                            R2_new = nx.has_path(G, g2_new_temp, f1)
                            if not R2_new:
                                # change the g1 to g1_new
                                # add the edge to G
                                g2_new_num = G_copy.nodes[g2_new_temp]['count']
                                data[idx][3] = str(g2_new_num)
                                all_gs.append(g2_new_temp)
                                G.add_edge(f2, g1)
                                G.add_edge(f1, g2_new_temp)
                                # random_sel = 1
                                # all_gs.append(g2)
                            else:
                                # need to random selected
                                random_sel = 1
                                all_gs.remove(g1)
                        else:
                            random_sel = 1
                            all_gs.remove(g1)
                    if random_sel == 1:
                        idx_list.append(idx)
                elif check_result[idx] == 3:
                # check_result[idx] != 0:
                    idx_list.append(idx)
            print("idx_list", idx_list)
            for rand_idx in idx_list:
                # random select
                sel_mark = True
                fasle_count_sel = 0
                while sel_mark:
                    key1, edge_pair1 = random.choice(list(edge_dict.items()))
                    key2, edge_pair2 = random.choice(list(edge_dict.items()))
                    # rand_edge_2 = random.sample(list(edge_dict.values()), 2)
                    # edge_pair1 = rand_edge_2[0]
                    # edge_pair2 = rand_edge_2[1]
                    # print(edge_pair1)
                    # print(edge_pair2)
                    R1_temp = nx.has_path(G, G_info_updated[edge_pair1[1]], G_info_updated[edge_pair2[0]])
                    R2_temp = nx.has_path(G, G_info_updated[edge_pair2[1]], G_info_updated[edge_pair1[0]])
                    # if there are repeated gs
                    if G_info_updated[edge_pair1[1]] in all_gs or G_info_updated[edge_pair2[1]] in all_gs:
                        sel_mark = True
                        continue
                    if G_info_updated[edge_pair1[1]] == G_info_updated[edge_pair2[1]]:
                        sel_mark = True
                        continue
                    # if there f in gs
                    if (edge_pair1[1] in [edge_pair1[0], edge_pair2[0]]) or (edge_pair2[1] in [edge_pair1[0], edge_pair2[0]]):
                        sel_mark = True
                        continue
                            
                    if not R1_temp and not R2_temp:
                        sel_mark = False
                        G.add_edge(G_info_updated[edge_pair2[0]], G_info_updated[edge_pair1[1]])
                        G.add_edge(G_info_updated[edge_pair1[0]], G_info_updated[edge_pair2[1]])
                        all_gs.append(G_info_updated[edge_pair1[1]])
                        all_gs.append(G_info_updated[edge_pair2[1]])
                        
                        data[rand_idx][0] = str(edge_pair1[0])
                        data[rand_idx][1] = str(edge_pair2[0])
                        data[rand_idx][2] = str(edge_pair1[1])
                        data[rand_idx][3] = str(edge_pair2[1])
                        # print("selected edges:")
                        # print(edge_pair1, key1)
                        # print(edge_pair2, key2)
                    else:
                        sel_mark = True
                        # continue
                    # if sel_mark == True:
                    #     fasle_count_sel += 1
                    #     if fasle_count_sel > 500:
                    #         # print
                    #         return None # if we could not find the correct one, we return False
            # print("all_gs:", all_gs)
            # print("all_gs_len:", len(all_gs))
            # print("unrepeated gs:", len(list(set(all_gs))))
            self.data = data
            print("cyle data", data)
            return data
    
    def check_cycle_pair_unchanged_scope(self, edge_dict, found_pairs):
        # G = self.graph
        # here we provide the graph with the updated edge
        data_temp = self.data
        # remove the found pairs
        found_pair_length = len(found_pairs)
        data = data_temp[:len(data_temp)-found_pair_length]

        G = self.graph
        G_copy = self.graph
        G_info = dict(list(G_copy.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        check_result = []
        all_gs = []
        # for item in data:
        #     [f1, f2, g1, g2] = [int(i) for i in item[0:4]]
        #     g1 = G_info_updated[g1]
        #     g2 = G_info_updated[g2]
        #     all_gs.append(g1)
        #     all_gs.append(g2)
        # before check each node, we update the graph with new edges from found pairs
        for item in found_pairs:
            [f1, f2, g1, g2] = [int(i) for i in item[0:4]]
            # based on the count number, find the node name
            f1 = G_info_updated[f1]
            f2 = G_info_updated[f2]
            g1 = G_info_updated[g1]
            g2 = G_info_updated[g2]
            # add the edge
            G.add_edge(f2, g1)
            # G.add_edge(f1, g2)
            all_gs.append(g1)
            # all_gs.append(g2)
        # I only want to keep the used gs in order to give more space to use
        for item in data:
            [f1, f2, g1, g2] = [int(i) for i in item[0:4]]
            # change_mark = 0
            # print(f1, f2, g1, g2)
            # based on the count number, find the node name
            f1 = G_info_updated[f1]
            f2 = G_info_updated[f2]
            g1 = G_info_updated[g1]
            g2 = G_info_updated[g2]
            # print(f1, f2, g1, g2)
            # check if there is a cycle after adding the edge
            R1 = nx.has_path(G, g1, f2)
            # print("what the value of R1:", R1)
            # R2 = nx.has_path(G, g2, f1)
            R2 = False
            # print("what the value of R2:", R2)
            if (g1 != g2) and (f1 != f2) and not R1 and not R2:
                check_result.append(0)
                G.add_edge(f2, g1)
                # G.add_edge(f1, g2)
                all_gs.append(g1)
                # all_gs.append(g2)
            else:
                if R1 and not R2:
                    check_result.append(1)
                    # all_gs.append(g2)
                elif not R1 and R2:
                    check_result.append(2)
                    all_gs.append(g1)
                else:
                    check_result.append(3)
        if sum(check_result) == 0: #correct
            # no need to change return the data
            for item in found_pairs:
                data.append(item)
            self.data = data
            return data
        else:
            # current the graph has been added edges which we need for correct localities
            # we need to change to make sure there is no loop here
            idx_list = []
            # print(check_result)
            for idx in range(len(check_result)):
                # here means that R1 has a path, we need to replace it with f1 successors
                # find the f1, f2, g1, g2
                if check_result[idx] == 1:
                    [f1_num, f2_num, g1_num, g2_num] = data[idx][0:4]
                    f1 = G_info_updated[int(f1_num)]
                    f2 = G_info_updated[int(f2_num)]
                    g1 = G_info_updated[int(g1_num)]
                    g2 = G_info_updated[int(g2_num)]
                    random_sel = 0
                    f1_fanout = list(G_copy.successors(f1))
                    f1_fanout_copy = list(G_copy.successors(f1))
                    for item in f1_fanout_copy:
                        if [G.nodes[f1]['count'], G.nodes[item]['count']] not in list(edge_dict.values()):
                            f1_fanout.remove(item)
                    # here we need to change to find other find out of f1
                    if len(f1_fanout) == 1: # here means of g1
                        random_sel = 1
                        # all_gs.remove(g2)
                    else:
                        # remove the f1's current fanout and see if there is unrepeated gs
                        if g1 in f1_fanout:
                            f1_fanout.remove(g1)
                        for item in all_gs:
                            if item in f1_fanout:
                                f1_fanout.remove(item)
                        if len(f1_fanout) > 1: # check if there is loop for R1
                            g1_new_temp = random.choice(f1_fanout)
                            R1_new = nx.has_path(G, g1_new_temp, f2)
                            if not R1_new:
                                # change the g1 to g1_new
                                # add the edge to G
                                g1_new_num = G_copy.nodes[g1_new_temp]['count']
                                data[idx][2] = str(g1_new_num)
                                all_gs.append(g1_new_temp)
                                G.add_edge(f2, g1_new_temp)
                                # G.add_edge(f1, g2)
                                # random_sel = 1
                                # all_gs.append(g2)
                            else:
                                # need to random selected
                                random_sel = 1
                                # all_gs.remove(g2)
                        else:
                            random_sel = 1
                            # all_gs.remove(g2)
                    # random select also need to be constricted
                    # 1) not f in gs; 2) no repeated gs; 3) no path in the G
                    if random_sel == 1:
                        idx_list.append(idx)
                elif check_result[idx] == 2:
                    [f1_num, f2_num, g1_num, g2_num] = data[idx][0:4]
                    f1 = G_info_updated[int(f1_num)]
                    f2 = G_info_updated[int(f2_num)]
                    g1 = G_info_updated[int(g1_num)]
                    g2 = G_info_updated[int(g2_num)]
                    random_sel = 0
                    f2_fanout = list(G_copy.successors(f2))
                    f2_fanout_copy = list(G_copy.successors(f2))
                    for item in f2_fanout_copy:
                        if [G.nodes[f2]['count'], G.nodes[item]['count']] not in list(edge_dict.values()):
                            f2_fanout.remove(item)
                    if len(f2_fanout) == 1: # here means of g1
                        random_sel = 1
                        all_gs.remove(g1)
                    else:
                        # remove the f1's current fanout and see if there is unrepeated gs
                        if g2 in f2_fanout:
                           f2_fanout.remove(g2)
                        for item in all_gs:
                            if item in f2_fanout:
                               f2_fanout.remove(item)
                        if len(f2_fanout) > 1: # check if there is loop for R1
                            g2_new_temp = random.choice(f2_fanout)
                            R2_new = nx.has_path(G, g2_new_temp, f1)
                            if not R2_new:
                                # change the g1 to g1_new
                                # add the edge to G
                                g2_new_num = G_copy.nodes[g2_new_temp]['count']
                                data[idx][3] = str(g2_new_num)
                                all_gs.append(g2_new_temp)
                                G.add_edge(f2, g1)
                                G.add_edge(f1, g2_new_temp)
                                # random_sel = 1
                                # all_gs.append(g2)
                            else:
                                # need to random selected
                                random_sel = 1
                                all_gs.remove(g1)
                        else:
                            random_sel = 1
                            all_gs.remove(g1)
                    if random_sel == 1:
                        idx_list.append(idx)
                elif check_result[idx] == 3:
                # check_result[idx] != 0:
                    idx_list.append(idx)
            print("idx_list", idx_list)
            for rand_idx in idx_list:
                # random select
                sel_mark = True
                fasle_count_sel = 0
                while sel_mark:
                    key1, edge_pair1 = random.choice(list(edge_dict.items()))
                    key2, edge_pair2 = random.choice(list(edge_dict.items()))
                    # rand_edge_2 = random.sample(list(edge_dict.values()), 2)
                    # edge_pair1 = rand_edge_2[0]
                    # edge_pair2 = rand_edge_2[1]
                    # print(edge_pair1)
                    # print(edge_pair2)
                    R1_temp = nx.has_path(G, G_info_updated[edge_pair1[1]], G_info_updated[edge_pair2[0]])
                    # R2_temp = nx.has_path(G, G_info_updated[edge_pair2[1]], G_info_updated[edge_pair1[0]])
                    R2_temp = False
                    # if there are repeated gs
                    if G_info_updated[edge_pair1[1]] in all_gs:# or G_info_updated[edge_pair2[1]] in all_gs:
                        sel_mark = True
                        continue
                    # if G_info_updated[edge_pair1[1]] == G_info_updated[edge_pair2[1]]:
                    #     sel_mark = True
                    #     continue
                    # if there f in gs
                    if (edge_pair1[1] in [edge_pair1[0], edge_pair2[0]]):# or (edge_pair2[1] in [edge_pair1[0], edge_pair2[0]]):
                        sel_mark = True
                        continue
                    # if f1 = f2
                    if edge_pair1[0] == edge_pair2[0]:
                        sel_mark = True
                        continue        
                    if not R1_temp and not R2_temp:
                        sel_mark = False
                        G.add_edge(G_info_updated[edge_pair2[0]], G_info_updated[edge_pair1[1]])
                        # G.add_edge(G_info_updated[edge_pair1[0]], G_info_updated[edge_pair2[1]])
                        all_gs.append(G_info_updated[edge_pair1[1]])
                        # all_gs.append(G_info_updated[edge_pair2[1]])
                        
                        data[rand_idx][0] = str(edge_pair1[0])
                        data[rand_idx][1] = str(edge_pair2[0])
                        data[rand_idx][2] = str(edge_pair1[1])
                        data[rand_idx][3] = str(edge_pair2[1])
                        # print("selected edges:")
                        # print(edge_pair1, key1)
                        # print(edge_pair2, key2)
                    else:
                        sel_mark = True
                        # continue
                    if sel_mark == True:
                        fasle_count_sel += 1
                        if fasle_count_sel > 500:
                            # print
                            return None # if we could not find the correct one, we return False
            # print("all_gs:", all_gs)
            # print("all_gs_len:", len(all_gs))
            # print("unrepeated gs:", len(list(set(all_gs))))
            for item in found_pairs:
                data.append(item)
            self.data = data
            print("cyle data", data)
            # after the checking, append the found pairs
            return data


     # check if there will introduce cycle after adding 2 pairs of edges
    def check_cycle_pair_unchanged(self, edge_dict, found_pairs):
        # G = self.graph
        # here we provide the graph with the updated edge
        data_temp = self.data
        # remove the found pairs
        found_pair_length = len(found_pairs)
        data = data_temp[:len(data_temp)-found_pair_length]

        G = self.graph
        G_copy = self.graph
        G_info = dict(list(G_copy.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        check_result = []
        all_gs = []
        # for item in data:
        #     [f1, f2, g1, g2] = [int(i) for i in item[0:4]]
        #     g1 = G_info_updated[g1]
        #     g2 = G_info_updated[g2]
        #     all_gs.append(g1)
        #     all_gs.append(g2)
        # before check each node, we update the graph with new edges from found pairs
        for item in found_pairs:
            [f1, f2, g1, g2] = [int(i) for i in item[0:4]]
            # based on the count number, find the node name
            f1 = G_info_updated[f1]
            f2 = G_info_updated[f2]
            g1 = G_info_updated[g1]
            g2 = G_info_updated[g2]
            # add the edge
            G.add_edge(f2, g1)
            G.add_edge(f1, g2)
            all_gs.append(g1)
            all_gs.append(g2)
        # I only want to keep the used gs in order to give more space to use
        for item in data:
            [f1, f2, g1, g2] = [int(i) for i in item[0:4]]
            # change_mark = 0
            # print(f1, f2, g1, g2)
            # based on the count number, find the node name
            f1 = G_info_updated[f1]
            f2 = G_info_updated[f2]
            g1 = G_info_updated[g1]
            g2 = G_info_updated[g2]
            # print(f1, f2, g1, g2)
            # check if there is a cycle after adding the edge
            R1 = nx.has_path(G, g1, f2)
            # print("what the value of R1:", R1)
            R2 = nx.has_path(G, g2, f1)
            # print("what the value of R2:", R2)
            if (g1 != g2) and (f1 != f2) and not R1 and not R2:
                check_result.append(0)
                G.add_edge(f2, g1)
                G.add_edge(f1, g2)
                all_gs.append(g1)
                all_gs.append(g2)
            else:
                if R1 and not R2:
                    check_result.append(1)
                    all_gs.append(g2)
                elif not R1 and R2:
                    check_result.append(2)
                    all_gs.append(g1)
                else:
                    check_result.append(3)
        if sum(check_result) == 0: #correct
            # no need to change return the data
            for item in found_pairs:
                data.append(item)
            self.data = data
            return data
        else:
            # current the graph has been added edges which we need for correct localities
            # we need to change to make sure there is no loop here
            idx_list = []
            # print(check_result)
            for idx in range(len(check_result)):
                # here means that R1 has a path, we need to replace it with f1 successors
                # find the f1, f2, g1, g2
                if check_result[idx] == 1:
                    [f1_num, f2_num, g1_num, g2_num] = data[idx][0:4]
                    f1 = G_info_updated[int(f1_num)]
                    f2 = G_info_updated[int(f2_num)]
                    g1 = G_info_updated[int(g1_num)]
                    g2 = G_info_updated[int(g2_num)]
                    random_sel = 0
                    f1_fanout = list(G_copy.successors(f1))
                    f1_fanout_copy = list(G_copy.successors(f1))
                    for item in f1_fanout_copy:
                        if [G.nodes[f1]['count'], G.nodes[item]['count']] not in list(edge_dict.values()):
                            f1_fanout.remove(item)
                    # here we need to change to find other find out of f1
                    if len(f1_fanout) == 1: # here means of g1
                        random_sel = 1
                        all_gs.remove(g2)
                    else:
                        # remove the f1's current fanout and see if there is unrepeated gs
                        if g1 in f1_fanout:
                            f1_fanout.remove(g1)
                        for item in all_gs:
                            if item in f1_fanout:
                                f1_fanout.remove(item)
                        if len(f1_fanout) > 1: # check if there is loop for R1
                            g1_new_temp = random.choice(f1_fanout)
                            R1_new = nx.has_path(G, g1_new_temp, f2)
                            if not R1_new:
                                # change the g1 to g1_new
                                # add the edge to G
                                g1_new_num = G_copy.nodes[g1_new_temp]['count']
                                data[idx][2] = str(g1_new_num)
                                all_gs.append(g1_new_temp)
                                G.add_edge(f2, g1_new_temp)
                                G.add_edge(f1, g2)
                                # random_sel = 1
                                # all_gs.append(g2)
                            else:
                                # need to random selected
                                random_sel = 1
                                all_gs.remove(g2)
                        else:
                            random_sel = 1
                            all_gs.remove(g2)
                    # random select also need to be constricted
                    # 1) not f in gs; 2) no repeated gs; 3) no path in the G
                    if random_sel == 1:
                        idx_list.append(idx)
                elif check_result[idx] == 2:
                    [f1_num, f2_num, g1_num, g2_num] = data[idx][0:4]
                    f1 = G_info_updated[int(f1_num)]
                    f2 = G_info_updated[int(f2_num)]
                    g1 = G_info_updated[int(g1_num)]
                    g2 = G_info_updated[int(g2_num)]
                    random_sel = 0
                    f2_fanout = list(G_copy.successors(f2))
                    f2_fanout_copy = list(G_copy.successors(f2))
                    for item in f2_fanout_copy:
                        if [G.nodes[f2]['count'], G.nodes[item]['count']] not in list(edge_dict.values()):
                            f2_fanout.remove(item)
                    if len(f2_fanout) == 1: # here means of g1
                        random_sel = 1
                        all_gs.remove(g1)
                    else:
                        # remove the f1's current fanout and see if there is unrepeated gs
                        if g2 in f2_fanout:
                           f2_fanout.remove(g2)
                        for item in all_gs:
                            if item in f2_fanout:
                               f2_fanout.remove(item)
                        if len(f2_fanout) > 1: # check if there is loop for R1
                            g2_new_temp = random.choice(f2_fanout)
                            R2_new = nx.has_path(G, g2_new_temp, f1)
                            if not R2_new:
                                # change the g1 to g1_new
                                # add the edge to G
                                g2_new_num = G_copy.nodes[g2_new_temp]['count']
                                data[idx][3] = str(g2_new_num)
                                all_gs.append(g2_new_temp)
                                G.add_edge(f2, g1)
                                G.add_edge(f1, g2_new_temp)
                                # random_sel = 1
                                # all_gs.append(g2)
                            else:
                                # need to random selected
                                random_sel = 1
                                all_gs.remove(g1)
                        else:
                            random_sel = 1
                            all_gs.remove(g1)
                    if random_sel == 1:
                        idx_list.append(idx)
                elif check_result[idx] == 3:
                # check_result[idx] != 0:
                    idx_list.append(idx)
            print("idx_list", idx_list)
            for rand_idx in idx_list:
                # random select
                sel_mark = True
                fasle_count_sel = 0
                while sel_mark:
                    key1, edge_pair1 = random.choice(list(edge_dict.items()))
                    key2, edge_pair2 = random.choice(list(edge_dict.items()))
                    # rand_edge_2 = random.sample(list(edge_dict.values()), 2)
                    # edge_pair1 = rand_edge_2[0]
                    # edge_pair2 = rand_edge_2[1]
                    # print(edge_pair1)
                    # print(edge_pair2)
                    R1_temp = nx.has_path(G, G_info_updated[edge_pair1[1]], G_info_updated[edge_pair2[0]])
                    R2_temp = nx.has_path(G, G_info_updated[edge_pair2[1]], G_info_updated[edge_pair1[0]])
                    # if there are repeated gs
                    if G_info_updated[edge_pair1[1]] in all_gs or G_info_updated[edge_pair2[1]] in all_gs:
                        sel_mark = True
                        continue
                    if G_info_updated[edge_pair1[1]] == G_info_updated[edge_pair2[1]]:
                        sel_mark = True
                        continue
                    # if there f in gs
                    if (edge_pair1[1] in [edge_pair1[0], edge_pair2[0]]) or (edge_pair2[1] in [edge_pair1[0], edge_pair2[0]]):
                        sel_mark = True
                        continue
                    # if f1 = f2
                    if edge_pair1[0] == edge_pair2[0]:
                        sel_mark = True
                        continue        
                    if not R1_temp and not R2_temp:
                        sel_mark = False
                        G.add_edge(G_info_updated[edge_pair2[0]], G_info_updated[edge_pair1[1]])
                        G.add_edge(G_info_updated[edge_pair1[0]], G_info_updated[edge_pair2[1]])
                        all_gs.append(G_info_updated[edge_pair1[1]])
                        all_gs.append(G_info_updated[edge_pair2[1]])
                        
                        data[rand_idx][0] = str(edge_pair1[0])
                        data[rand_idx][1] = str(edge_pair2[0])
                        data[rand_idx][2] = str(edge_pair1[1])
                        data[rand_idx][3] = str(edge_pair2[1])
                        # print("selected edges:")
                        # print(edge_pair1, key1)
                        # print(edge_pair2, key2)
                    else:
                        sel_mark = True
                        # continue
                    if sel_mark == True:
                        fasle_count_sel += 1
                        if fasle_count_sel > 500:
                            # print
                            return None # if we could not find the correct one, we return False
            # print("all_gs:", all_gs)
            # print("all_gs_len:", len(all_gs))
            # print("unrepeated gs:", len(list(set(all_gs))))
            for item in found_pairs:
                data.append(item)
            self.data = data
            print("cyle data", data)
            # after the checking, append the found pairs
            return data





    # update the graph
    # consider to use it in the future
    def update_graph(self, G):
        # G = self.graph
        data = self.data
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        for item in data:
            [f1, f2, g1, g2] = [int(i) for i in item[0:4]]
            # based on the count number, find the node name
            f1 = G_info_updated[f1]
            f2 = G_info_updated[f2]
            g1 = G_info_updated[g1]
            g2 = G_info_updated[g2]
            # add the edge
            G.add_edge(f1, g2)
            G.add_edge(f2, g1)
        # update the graph
        return G
    

    


    # new added by 01/08
    # def init_kgss(self, key_size):
    #     for i in range(key_size):
    #         self.data.append([None, None, None, None, None, None])
    #     return self.data
