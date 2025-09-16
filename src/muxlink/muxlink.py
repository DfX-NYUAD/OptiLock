#!/usr/bin/python
import torch
import numpy as np
import sys, copy
import os.path
import random

from ec.impl.kgs_solution import KGSSolution

sys.path.append('%s/../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
import os
import re
import shutil
import networkx as nx
from muxlink.original import *
# import threading
import time
from multiprocessing import Process
# from src.muxlink.original import *


def GenerateKey(K):
    nums = np.ones(K)
    nums[0:(K // 2)] = 0
    np.random.shuffle(nums)
    return nums


def ExtractSingleMultiOutputNodes(G):
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
    return F_multi, F_single


def verilog2gates(verilog):
    Dict_gates = {'xor': [0, 1, 0, 0, 0, 0, 0, 0],
                  'XOR': [0, 1, 0, 0, 0, 0, 0, 0],
                  'OR': [0, 0, 1, 0, 0, 0, 0, 0],
                  'or': [0, 0, 1, 0, 0, 0, 0, 0],
                  'XNOR': [0, 0, 0, 1, 0, 0, 0, 0],
                  'xnor': [0, 0, 0, 1, 0, 0, 0, 0],
                  'and': [0, 0, 0, 0, 1, 0, 0, 0],
                  'AND': [0, 0, 0, 0, 1, 0, 0, 0],
                  'nand': [0, 0, 0, 0, 0, 1, 0, 0],
                  'buf': [0, 0, 0, 0, 0, 0, 0, 1],
                  'BUF': [0, 0, 0, 0, 0, 0, 0, 1],
                  'NAND': [0, 0, 0, 0, 0, 1, 0, 0],
                  'not': [0, 0, 0, 0, 0, 0, 1, 0],
                  'NOT': [0, 0, 0, 0, 0, 0, 1, 0],
                  'nor': [1, 0, 0, 0, 0, 0, 0, 0],
                  'NOR': [1, 0, 0, 0, 0, 0, 0, 0],
                  }
    G = nx.DiGraph()
    ML_count = 0
    regex = "\s*(\S+)\s*=\s*(BUF|NOT)\((\S+)\)\s*"
    for output, function, net_str in re.findall(regex, verilog, flags=re.I | re.DOTALL):
        input = net_str.replace(" ", "")
        G.add_edge(input, output)
        G.nodes[output]['gate'] = function
        G.nodes[output]['count'] = ML_count
        ML_count += 1
    regex = "(\S+)\s*=\s*(OR|XOR|AND|NAND|XNOR|NOR|AOI\d*|OAI\d*)\((.+?)\)\s*"
    for output, function, net_str in re.findall(regex, verilog, flags=re.I | re.DOTALL):
        nets = net_str.replace(" ", "").replace("\n", "").replace("\t", "").split(",")
        inputs = nets
        G.add_edges_from((net, output) for net in inputs)
        G.nodes[output]['gate'] = function
        G.nodes[output]['count'] = ML_count
        ML_count += 1
    for n in G.nodes():
        if 'gate' not in G.nodes[n]:
            G.nodes[n]['gate'] = 'input'
    for n in G.nodes:
        G.nodes[n]['output'] = False
    out_regex = "OUTPUT\((.+?)\)\n"
    for net_str in re.findall(out_regex, verilog, flags=re.I | re.DOTALL):
        nets = net_str.replace(" ", "").replace("\n", "").replace("\t", "").split(",")
        for net in nets:
            if net in G:
                G.nodes[net]['output'] = True
    # print("G nodes numer", G.nodes())
    return G


def read_string(path):
    with open(path, 'r') as f:
        data = f.read()
    return data


def verilog2gates_train(f_link_train, f_link_train_f, verilog, f_feat, f_cell, f_count, dump):
    Dict_gates = {'xor': [0, 1, 0, 0, 0, 0, 0, 0],
                  'XOR': [0, 1, 0, 0, 0, 0, 0, 0],
                  'OR': [0, 0, 1, 0, 0, 0, 0, 0],
                  'or': [0, 0, 1, 0, 0, 0, 0, 0],
                  'XNOR': [0, 0, 0, 1, 0, 0, 0, 0],
                  'xnor': [0, 0, 0, 1, 0, 0, 0, 0],
                  'and': [0, 0, 0, 0, 1, 0, 0, 0],
                  'AND': [0, 0, 0, 0, 1, 0, 0, 0],
                  'nand': [0, 0, 0, 0, 0, 1, 0, 0],
                  'buf': [0, 0, 0, 0, 0, 0, 0, 1],
                  'BUF': [0, 0, 0, 0, 0, 0, 0, 1],
                  'NAND': [0, 0, 0, 0, 0, 1, 0, 0],
                  'not': [0, 0, 0, 0, 0, 0, 1, 0],
                  'NOT': [0, 0, 0, 0, 0, 0, 1, 0],
                  'nor': [1, 0, 0, 0, 0, 0, 0, 0],
                  'NOR': [1, 0, 0, 0, 0, 0, 0, 0],
                  }
    G = nx.DiGraph()
    ML_count = 0
    regex = "\s*(\S+)\s*=\s*(BUF|NOT)\((\S+)\)\s*"
    for output, function, net_str in re.findall(regex, verilog, flags=re.I | re.DOTALL):
        input = net_str.replace(" ", "")

        G.add_edge(input, output)
        G.nodes[output]['gate'] = function
        G.nodes[output]['count'] = ML_count
        if dump:

            feat = Dict_gates[function]
            for item in feat:
                f_feat.write(str(item) + " ")
            f_feat.write("\n")
            f_cell.write(str(ML_count) + " assign for output " + output + "\n")
            f_count.write(str(ML_count) + "\n")
        ML_count += 1
    regex = "(\S+)\s*=\s*(OR|XOR|AND|NAND|XNOR|NOR)\((.+?)\)\s*"
    for output, function, net_str in re.findall(regex, verilog, flags=re.I | re.DOTALL):
        nets = net_str.replace(" ", "").replace("\n", "").replace("\t", "").split(",")
        inputs = nets
        G.add_edges_from((net, output) for net in inputs)
        G.nodes[output]['gate'] = function
        G.nodes[output]['count'] = ML_count
        if dump:
            feat = Dict_gates[function]
            for item in feat:
                f_feat.write(str(item) + " ")
            f_feat.write("\n")
            f_cell.write(str(ML_count) + " assign for output " + output + "\n")
            f_count.write(str(ML_count) + "\n")
        ML_count += 1
    for n in G.nodes():
        if 'gate' not in G.nodes[n]:
            G.nodes[n]['gate'] = 'input'
    for n in G.nodes:
        G.nodes[n]['output'] = False
    out_regex = "OUTPUT\((.+?)\)\n"
    for net_str in re.findall(out_regex, verilog, flags=re.I | re.DOTALL):
        nets = net_str.replace(" ", "").replace("\n", "").replace("\t", "").split(",")
        for net in nets:
            if net not in G:
                print("Output " + net + " is Float")
            else:
                G.nodes[net]['output'] = True
    if dump:
        for u, v in G.edges:
            if 'count' in G.nodes[u].keys() and 'count' in G.nodes[v].keys():
                f_link_train.write(str(G.nodes[u]['count']) + " " + str(G.nodes[v]['count']) + "\n")
    return G


def verilog2gates_locked(f_link_train, f_link_train_f, f_link_test, f_link_test_neg, verilog, f_feat, f_cell, f_count,
                         dump):
    key_size = 0
    Dict_gates = {'xor': [0, 1, 0, 0, 0, 0, 0, 0],
                  'XOR': [0, 1, 0, 0, 0, 0, 0, 0],
                  'OR': [0, 0, 1, 0, 0, 0, 0, 0],
                  'or': [0, 0, 1, 0, 0, 0, 0, 0],
                  'XNOR': [0, 0, 0, 1, 0, 0, 0, 0],
                  'xnor': [0, 0, 0, 1, 0, 0, 0, 0],
                  'and': [0, 0, 0, 0, 1, 0, 0, 0],
                  'AND': [0, 0, 0, 0, 1, 0, 0, 0],
                  'nand': [0, 0, 0, 0, 0, 1, 0, 0],
                  'buf': [0, 0, 0, 0, 0, 0, 0, 1],
                  'BUF': [0, 0, 0, 0, 0, 0, 0, 1],
                  'BUFF': [0, 0, 0, 0, 0, 0, 0, 1],
                  'NAND': [0, 0, 0, 0, 0, 1, 0, 0],
                  'not': [0, 0, 0, 0, 0, 0, 1, 0],
                  'NOT': [0, 0, 0, 0, 0, 0, 1, 0],
                  'nor': [1, 0, 0, 0, 0, 0, 0, 0],
                  'NOR': [1, 0, 0, 0, 0, 0, 0, 0],
                  }
    G = nx.DiGraph()
    ML_count = 0
    regex = "\s*(\S+)\s*=\s*(BUF|BUFF|NOT)\((\S+)\)\s*"
    for output, function, net_str in re.findall(regex, verilog, flags=re.I | re.DOTALL):
        input = net_str.replace(" ", "")

        G.add_edge(input, output)
        G.nodes[output]['gate'] = function
        G.nodes[output]['count'] = ML_count
        if dump:

            feat = Dict_gates[function]
            for item in feat:
                f_feat.write(str(item) + " ")
            f_feat.write("\n")
            f_cell.write(str(ML_count) + " assign for output " + output + "\n")
            f_count.write(str(ML_count) + "\n")
        ML_count += 1
    regex = "(\S+)\s*=\s*(OR|XOR|AND|NAND|XNOR|NOR)\((.+?)\)\s*"
    for output, function, net_str in re.findall(regex, verilog, flags=re.I | re.DOTALL):
        nets = net_str.replace(" ", "").replace("\n", "").replace("\t", "").split(",")
        inputs = nets
        G.add_edges_from((net, output) for net in inputs)
        G.nodes[output]['gate'] = function
        G.nodes[output]['count'] = ML_count
        if dump:
            feat = Dict_gates[function]
            for item in feat:
                f_feat.write(str(item) + " ")
            f_feat.write("\n")
            f_cell.write(str(ML_count) + " assign for output " + output + "\n")
            f_count.write(str(ML_count) + "\n")
        ML_count += 1
    for n in G.nodes():
        if 'gate' not in G.nodes[n]:
            G.nodes[n]['gate'] = 'input'
    for n in G.nodes:
        G.nodes[n]['output'] = False
    out_regex = "OUTPUT\((.+?)\)\n"
    for net_str in re.findall(out_regex, verilog, flags=re.I | re.DOTALL):
        nets = net_str.replace(" ", "").replace("\n", "").replace("\t", "").split(",")
        for net in nets:
            if net not in G:
                print("Output " + net + " is Float")
            else:
                G.nodes[net]['output'] = True
    regex = "#key=(\d+)\s*"
    for key_bits in re.findall(regex, verilog, flags=re.I | re.DOTALL):
        key_size = len(key_bits)
        # print(key_bits)
        # print("Key size is "+str(key_size))
        i = 0
        K_list = np.ones(key_size)
        for bit in key_bits:
            K_list[i] = int(bit)
            i = i + 1
        # print(K_list)
    # debug - Zeng
    # print("G nodes list: ", G.nodes())
    # print("G nodes list has count: ", dict(G.nodes(data='count')))
    regex = "(\S+)\s*=\s*(MUX)\((.+?)\)\s*"
    for output, function, net_str in re.findall(regex, verilog, flags=re.I | re.DOTALL):
        nets = net_str.replace(" ", "").replace("\n", "").replace("\t", "").split(",")
        inputs = nets
        output_x = output.replace('_from_mux', '')
        regex_key = "keyinput(\d+)"
        for key_bit in re.findall(regex_key, inputs[0], flags=re.I | re.DOTALL):
            key_bit_value = K_list[int(key_bit)]
            correct = ""
            false = ""
            # debug - Zeng
            # print("inputs: ", inputs)
            if key_bit_value == 0:
                correct = inputs[1]
                false = inputs[2]
                f_link_test.write(str(G.nodes[correct]['count']) + " " + str(G.nodes[output_x]['count']) + "\n")

                f_link_test_neg.write(str(G.nodes[false]['count']) + " " + str(G.nodes[output_x]['count']) + "\n")
            else:

                correct = inputs[2]
                false = inputs[1]

                f_link_test.write(str(G.nodes[correct]['count']) + " " + str(G.nodes[output_x]['count']) + "\n")

                f_link_test_neg.write(str(G.nodes[false]['count']) + " " + str(G.nodes[output_x]['count']) + "\n")
    i = 0

    if dump:
        for u, v in G.edges:
            if 'count' in G.nodes[u].keys() and 'count' in G.nodes[v].keys():
                f_link_train.write(str(G.nodes[u]['count']) + " " + str(G.nodes[v]['count']) + "\n")
    return G, key_size


class MuxLink:

    def __init__(self, circuit_name):
        self.model_path = "../ml_data_" + circuit_name + "/"
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            perl_source_file = os.path.join("../ml_data_muxlink", "break_DMUX.pl")
            perl_dest_file = os.path.join(self.model_path, "break_DMUX.pl")
            shutil.copy(perl_source_file, perl_dest_file)



    def FindPairs(self, S_sel, F_single, F_multi, I_max, O_max, G, selected_g):
        F1 = []
        F2 = []
        if S_sel == "s1" or S_sel == "s2":
            F1 = F_multi
            F2 = F_multi
        elif S_sel == "s3":
            F1 = F_multi
            F2 = F_single
        else:  ##here might design for S4
            F1 = F_multi + F_single
            F2 = F_multi + F_single
        done = False
        i = 0
        f1 = ""
        f2 = ""
        g1 = ""
        g2 = ""
        while i < I_max:
            f1 = random.choice(F1)
            f2 = f1
            while f2 == f1:
                f2 = random.choice(F2)
            ## here is to make sure f1 != f2

            j = 0
            while j < O_max:
                g1 = random.choice(list(G.successors(f1)))
                g2 = random.choice(list(G.successors(f2)))
                R1 = nx.has_path(G, g1, f2)
                R2 = nx.has_path(G, g2, f1)
                if (g1 != g2) and not R1 and not R2:
                    if g1 not in selected_g and g2 not in selected_g:
                        done = True
                        break
                j = j + 1
            if done:
                break
            i = i + 1
        if done:
            if S_sel == "s1" or S_sel == "s4":
                G.add_edge(f2, g1)
                G.add_edge(f1, g2)
            else:
                G.add_edge(f2, g1)

        return f1, f2, g1, g2, done, G
    
    def train_only(self, bench_as_a_string, lock_mark, key=2, epochs=100):
        location = self.model_path + "trained_model"
        
        ##Now it is the time to train

        train_or_test_original_impl(self.model_path, "trained_model", "links_train.txt", "links_test.txt",
                                    "link_test_n.txt", 3, epochs, True)

        # os.system(
        #     "python ./muxlink/original.py --file-name trained_model --train-name links_train.txt --test-name links_test.txt --testneg-name link_test_n.txt --hop 3 --save-model > log_train_model.txt")



    def train(self, bench_as_a_string, lock_mark, key=2, epochs=100):
        location = self.model_path + "trained_model"
        if not os.path.exists(location):
            os.mkdir(location)
        f_feat = open(location + "/feat.txt", "w")
        f_cell = open(location + "/cell.txt", "w")
        f_count = open(location + "/count.txt", "w")
        f_link_test_neg = open(location + "/link_test_n.txt", "w")
        f_link_test = open(location + "/links_test.txt", "w")
        f_link_train = open(location + "/links_train_temp.txt", "w")
        f_link_train_f = open(location + "/links_train.txt", "w")
        if lock_mark == False:
            c = verilog2gates_train(f_link_train, f_link_train_f, bench_as_a_string, f_feat, f_cell, f_count, True)
            new_c = verilog2gates_train(f_link_train, f_link_train_f, bench_as_a_string, f_feat, f_cell, f_count, False)
        else:
            c, _ = verilog2gates_locked(f_link_train, f_link_train_f, f_link_test, f_link_test_neg, bench_as_a_string, f_feat, f_cell, f_count, True)
            new_c = c
        self.F_multi, self.F_single = ExtractSingleMultiOutputNodes(c)
        # Generate the key (small key, only for training purposes)
        key_size = key # 2
        K_list = GenerateKey(key_size)
        counter = key_size - 1
        myDict = {}
        # choices for locking. s4 is avilable always. So it is only used when needed
        L_s = np.array(["s1", "s2", "s3"])
        selected_f1_gates = []
        selected_f2_gates = []
        selected_g2_gates = []
        selected_g1_gates = []
        selected_g = []
        break_out = 0
        while counter >= 0:
            np.random.shuffle(L_s)
            fallback = True
            S_sel = ""
            for s in L_s:
                if s == "s1" and counter < 2:
                    continue
                elif s == "s3" and len(self.F_multi) > 1 and len(self.F_single) > 1:
                    S_sel = s
                    fallback = False
                    break
                elif len(self.F_multi) < 2:

                    continue
                S_sel = s
                fallback = False
                break
            if fallback:
                S_sel = "s4"
            to_be_new_c = nx.DiGraph()
            done = False
            f1, f2, g1, g2, done, to_be_new_c = self.FindPairs(S_sel, self.F_single, self.F_multi, 10, 10, new_c,
                                                               selected_g)
            if not done:
                break_out += 1
                if (break_out >= 10):
                    break_out = 0
                    S_sel = "s4"
                    while not done:
                        f1, f2, g1, g2, done, to_be_new_c = self.FindPairs("s4", self.F_single, self.F_multi, 10, 10,
                                                                           new_c,
                                                                           selected_g)
                else:
                    continue
            if len(list(nx.simple_cycles(to_be_new_c))) != 0:
                continue
            new_c = copy.deepcopy(to_be_new_c)
            selected_f1_gates.append(f1)
            selected_f2_gates.append(f2)
            if f1 in self.F_multi:
                self.F_multi.remove(f1)
            elif f1 in self.F_single:
                self.F_single.remove(f1)
            if f2 in self.F_multi:
                self.F_multi.remove(f2)
            elif f2 in self.F_single:
                self.F_single.remove(f2)
            if S_sel == "s1":
                myDict[g1] = [f1, f2, counter]
                myDict[g2] = [f2, f1, counter - 1]
                counter = counter - 2
                selected_g1_gates.append(g1)
                selected_g2_gates.append(g2)
                selected_g.append(g1)
                selected_g.append(g2)
                f_link_test_neg.write(str(c.nodes[f2]['count']) + " " + str(c.nodes[g1]['count']) + "\n")
                f_link_test_neg.write(str(c.nodes[f1]['count']) + " " + str(c.nodes[g2]['count']) + "\n")
                f_link_test.write(str(c.nodes[f1]['count']) + " " + str(c.nodes[g1]['count']) + "\n")
                f_link_test.write(str(c.nodes[f2]['count']) + " " + str(c.nodes[g2]['count']) + "\n")
            else:
                if S_sel == "s4":
                    selected_g1_gates.append(g1)
                    selected_g2_gates.append(g2)
                    selected_g.append(g1)
                    selected_g.append(g2)
                    myDict[g1] = [f1, f2, counter]

                    myDict[g2] = [f2, f1, counter]

                    f_link_test_neg.write(str(c.nodes[f2]['count']) + " " + str(c.nodes[g1]['count']) + "\n")
                    f_link_test_neg.write(str(c.nodes[f1]['count']) + " " + str(c.nodes[g2]['count']) + "\n")
                    f_link_test.write(str(c.nodes[f1]['count']) + " " + str(c.nodes[g1]['count']) + "\n")
                    f_link_test.write(str(c.nodes[f2]['count']) + " " + str(c.nodes[g2]['count']) + "\n")
                else:

                    f_link_test_neg.write(str(c.nodes[f2]['count']) + " " + str(c.nodes[g1]['count']) + "\n")
                    f_link_test.write(str(c.nodes[f1]['count']) + " " + str(c.nodes[g1]['count']) + "\n")
                    selected_g1_gates.append(g1)
                    selected_g.append(g1)
                    myDict[g1] = [f1, f2, counter]
                counter = counter - 1
        f_feat.close()
        f_cell.close()
        f_count.close()
        f_link_test.close()
        f_link_test_neg.close()
        f_link_train.close()

        with open(location + "/links_test.txt") as f_a, open(location + "/links_train_temp.txt") as f_b:
            a_lines = set(f_a.read().splitlines())
            b_lines = set(f_b.read().splitlines())
            for line in b_lines:
                if line not in a_lines:
                    f_link_train_f.write(line + "\n")
        f_link_train_f.close()
        os.remove(location + "/links_train_temp.txt")
        ##Now it is the time to train

        train_or_test_original_impl(self.model_path, "trained_model", "links_train.txt", "links_test.txt",
                                    "link_test_n.txt", 3, epochs, True)

        # os.system(
        #     "python ./muxlink/original.py --file-name trained_model --train-name links_train.txt --test-name links_test.txt --testneg-name link_test_n.txt --hop 3 --save-model > log_train_model.txt")

        #        Main.py --file-name trained_model --train-name links_train.txt  --test-name links_test.txt --testneg-name link_test_n.txt --hop 3  --save-model > log_train_model.txt')

    def lock(self, bench_as_a_string, key_len, d_mux_type):
        c = verilog2gates(bench_as_a_string)
        new_c = verilog2gates(bench_as_a_string)
        self.F_multi, self.F_single = ExtractSingleMultiOutputNodes(c)  # TODO: this could be done once
        # Generate the key
        K_list = GenerateKey(key_len)
        counter = key_len - 1
        myDict = {}
        if d_mux_type == "eD-MUX":
            L_s = np.array(["s1", "s2", "s3"])
        else:
            L_s = np.array([])

        selected_f1_gates = []
        selected_f2_gates = []
        selected_g2_gates = []
        selected_g1_gates = []
        selected_g = []
        break_out = 0
        while counter >= 0:
            np.random.shuffle(L_s)
            fallback = True
            S_sel = ""
            for s in L_s:
                if s == "s1" and counter < 2:
                    continue  ## no operation???
                elif s == "s3" and len(self.F_multi) > 1 and len(self.F_single) > 1:
                    S_sel = s
                    fallback = False
                    break  ## no operation??
                elif len(self.F_multi) < 2:

                    continue  ## no operations??
                S_sel = s
                fallback = False
                break
            if fallback:
                S_sel = "s4"
            to_be_new_c = nx.DiGraph()
            done = False
            f1, f2, g1, g2, done, to_be_new_c = self.FindPairs(S_sel, self.F_single, self.F_multi, 10, 10, new_c,
                                                               selected_g)
            if not done:
                break_out += 1
                if (break_out >= 10):
                    break_out = 0
                    S_sel = "s4"
                    while not done:
                        # each time find one pair(f1, f2, g1, g2)
                        f1, f2, g1, g2, done, to_be_new_c = self.FindPairs("s4", self.F_single, self.F_multi, 10, 10,
                                                                           new_c,
                                                                           selected_g)
                else:
                    continue
            if len(list(nx.simple_cycles(to_be_new_c))) != 0:
                continue  # Zeng added : avoid generate the loop in the circuit
            new_c = copy.deepcopy(to_be_new_c)
            selected_f1_gates.append(f1)
            selected_f2_gates.append(f2)
            if f1 in self.F_multi:
                self.F_multi.remove(f1)
            elif f1 in self.F_single:
                self.F_single.remove(f1)
            if f2 in self.F_multi:
                self.F_multi.remove(f2)
            elif f2 in self.F_single:
                self.F_single.remove(f2)
            if S_sel == "s1":
                myDict[g1] = [f1, f2, counter]
                myDict[g2] = [f2, f1, counter - 1]
                counter = counter - 2
                selected_g1_gates.append(g1)
                selected_g2_gates.append(g2)
                selected_g.append(g1)
                selected_g.append(g2)
            else:
                if S_sel == "s4":
                    selected_g1_gates.append(g1)
                    selected_g2_gates.append(g2)
                    selected_g.append(g1)
                    selected_g.append(g2)
                    myDict[g1] = [f1, f2, counter]

                    myDict[g2] = [f2, f1, counter]

                else:

                    selected_g1_gates.append(g1)
                    selected_g.append(g1)
                    myDict[g1] = [f1, f2, counter]
                counter = counter - 1
        if len(list(nx.simple_cycles(new_c))) != 0:
            sys.exit("There is a loop in the circuit!")
        locked_file = ""
        i = 0
        locked_file = locked_file + "#key="
        while i < key_len:
            locked_file = locked_file + str(int(K_list[i]))
            i = i + 1
        locked_file = locked_file + ("\n")
        i = 0
        while i < key_len+0:
            locked_file = locked_file + "INPUT(keyinput" + str(i) + ")\n"
            i = i + 1
        count = 0
        detected = 0

        for line in bench_as_a_string.split("\n"):
            count += 1
            line = line.strip()
            if any(ext + " =" in line for ext in selected_g):  # id -> gate_name -> search
                detected = detected + 1
                regex = "(\S+)\s*=\s*(NOT|BUF|OR|XOR|AND|NAND|XNOR|NOR)\((.+?)\)\s*"
                for output, function, net_str in re.findall(regex, line, flags=re.I | re.DOTALL):
                    if output in myDict.keys():
                        my_f1 = myDict[output][0]
                        my_f2 = myDict[output][1]
                        my_key = myDict[output][2]
                        line = line.replace(my_f1 + ",", output + "_from_mux,")
                        line = line.replace(my_f1 + ")", output + "_from_mux)")
                        locked_file = locked_file + (line + "\n")
                        if K_list[my_key] == 0:
                            locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
                                my_key) + ", " + my_f1 + ", " + my_f2 + ")\n"
                        else:

                            locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
                                my_key) + ", " + my_f2 + ", " + my_f1 + ")\n"
                    else:
                        locked_file = locked_file + line + "\n"
            else:
                locked_file = locked_file + line + "\n"

        # Done-lilas: this will be overwritten for each solution. Must be fixed (add solution id!)
        # Question: where do we read this check file again?
        # no need to keep solution id
        # text_file = open(self.model_path + "check_" + "original.txt", "w")
        # text_file.write(locked_file)
        # text_file.close()

        return locked_file

    # new lock method by adding new condition for g
    def new_lock(self, bench_as_a_string, key_len, d_mux_type, selected_g):
        c = verilog2gates(bench_as_a_string)
        new_c = verilog2gates(bench_as_a_string)
        self.F_multi, self.F_single = ExtractSingleMultiOutputNodes(c)  # TODO: this could be done once
        # Generate the key
        K_list = GenerateKey(key_len)
        counter = key_len - 1
        myDict = {}
        if d_mux_type == "eD-MUX":
            L_s = np.array(["s1", "s2", "s3"])
        else:
            L_s = np.array([])

        selected_f1_gates = []
        selected_f2_gates = []
        selected_g2_gates = []
        selected_g1_gates = []
        # selected_g = []
        break_out = 0
        while counter >= 0:
            np.random.shuffle(L_s)
            fallback = True
            S_sel = ""
            for s in L_s:
                if s == "s1" and counter < 2:
                    continue  ## no operation???
                elif s == "s3" and len(self.F_multi) > 1 and len(self.F_single) > 1:
                    S_sel = s
                    fallback = False
                    break  ## no operation??
                elif len(self.F_multi) < 2:

                    continue  ## no operations??
                S_sel = s
                fallback = False
                break
            if fallback:
                S_sel = "s4"
            to_be_new_c = nx.DiGraph()
            done = False
            f1, f2, g1, g2, done, to_be_new_c = self.FindPairs(S_sel, self.F_single, self.F_multi, 10, 10, new_c,
                                                               selected_g)
            if not done:
                break_out += 1
                if (break_out >= 10):
                    break_out = 0
                    S_sel = "s4"
                    while not done:
                        # each time find one pair(f1, f2, g1, g2)
                        f1, f2, g1, g2, done, to_be_new_c = self.FindPairs("s4", self.F_single, self.F_multi, 10, 10,
                                                                           new_c,
                                                                           selected_g)
                else:
                    continue
            if len(list(nx.simple_cycles(to_be_new_c))) != 0:
                continue  # Zeng added : avoid generate the loop in the circuit
            new_c = copy.deepcopy(to_be_new_c)
            selected_f1_gates.append(f1)
            selected_f2_gates.append(f2)
            if f1 in self.F_multi:
                self.F_multi.remove(f1)
            elif f1 in self.F_single:
                self.F_single.remove(f1)
            if f2 in self.F_multi:
                self.F_multi.remove(f2)
            elif f2 in self.F_single:
                self.F_single.remove(f2)
            if S_sel == "s1":
                myDict[g1] = [f1, f2, counter]
                myDict[g2] = [f2, f1, counter - 1]
                counter = counter - 2
                selected_g1_gates.append(g1)
                selected_g2_gates.append(g2)
                selected_g.append(g1)
                selected_g.append(g2)
            else:
                if S_sel == "s4":
                    selected_g1_gates.append(g1)
                    selected_g2_gates.append(g2)
                    selected_g.append(g1)
                    selected_g.append(g2)
                    myDict[g1] = [f1, f2, counter]

                    myDict[g2] = [f2, f1, counter]

                else:

                    selected_g1_gates.append(g1)
                    selected_g.append(g1)
                    myDict[g1] = [f1, f2, counter]
                counter = counter - 1
        if len(list(nx.simple_cycles(new_c))) != 0:
            sys.exit("There is a loop in the circuit!")
        locked_file = ""
        i = 0
        locked_file = locked_file + "#key="
        while i < key_len:
            locked_file = locked_file + str(int(K_list[i]))
            i = i + 1
        locked_file = locked_file + ("\n")
        i = 0
        while i < key_len:
            locked_file = locked_file + "INPUT(keyinput" + str(i) + ")\n"
            i = i + 1
        count = 0
        detected = 0

        for line in bench_as_a_string.split("\n"):
            count += 1
            line = line.strip()
            if any(ext + " =" in line for ext in selected_g):  # id -> gate_name -> search
                detected = detected + 1
                regex = "(\S+)\s*=\s*(NOT|BUF|OR|XOR|AND|NAND|XNOR|NOR)\((.+?)\)\s*"
                for output, function, net_str in re.findall(regex, line, flags=re.I | re.DOTALL):
                    if output in myDict.keys():
                        my_f1 = myDict[output][0]
                        my_f2 = myDict[output][1]
                        my_key = myDict[output][2]
                        line = line.replace(my_f1 + ",", output + "_from_mux,")
                        line = line.replace(my_f1 + ")", output + "_from_mux)")
                        locked_file = locked_file + (line + "\n")
                        if K_list[my_key] == 0:
                            locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
                                my_key) + ", " + my_f1 + ", " + my_f2 + ")\n"
                        else:

                            locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
                                my_key) + ", " + my_f2 + ", " + my_f1 + ")\n"
                    else:
                        locked_file = locked_file + line + "\n"
            else:
                locked_file = locked_file + line + "\n"

        # Done-lilas: this will be overwritten for each solution. Must be fixed (add solution id!)
        # Question: where do we read this check file again?
        # no need to keep solution id
        # text_file = open(self.model_path + "check_" + "original.txt", "w")
        # text_file.write(locked_file)
        # text_file.close()
        print("what is the selected_g:", selected_g)
        return locked_file, selected_g

    
    def attack(self, bench_as_a_string, circuit_name, h_hop): # add a circuit name
        location = self.model_path + "test_model"

        if not os.path.exists(location):
            os.mkdir(location)
        
        start_time = time.time()
        f_feat = open(location + "/feat.txt", "w")
        f_cell = open(location + "/cell.txt", "w")
        f_count = open(location + "/count.txt", "w")
        f_link_test_neg = open(location + "/link_test_n.txt", "w")
        f_link_test = open(location + "/links_test.txt", "w")
        f_link_train = open(location + "/links_train_temp.txt", "w")
        f_link_train_f = open(location + "/links_train.txt", "w")
        c, key_size = verilog2gates_locked(f_link_train, f_link_train_f, f_link_test, f_link_test_neg,
                                           bench_as_a_string,
                                           f_feat, f_cell, f_count, True)
        f_feat.close()
        f_cell.close()
        f_count.close()
        f_link_test.close()
        f_link_test_neg.close()
        f_link_train.close()
        end_time = time.time()
        print("the time for verilog2gates_locked is:", end_time - start_time)

        with open(location + "/links_test.txt") as f_a, open(location + "/links_train_temp.txt") as f_b:
            a_lines = set(f_a.read().splitlines())
            b_lines = set(f_b.read().splitlines())
            for line in b_lines:
                if line not in a_lines:
                    f_link_train_f.write(line + "\n")
        f_link_train_f.close()
        os.remove(location + "/links_train_temp.txt")
        ##Now it is the time to test

        # here to divid all the link in the links_test.txt and links_test_n.txt into several files
        # and then use the perl script to break the DMUX
        # the number of files is decided by each line of the links_test.txt and links_test_n.txt
        # divide the links_test.txt
        start_time = time.time()
        train_or_test_original_impl(self.model_path, location, "links_train.txt", "links_test.txt", "link_test_n.txt",
                                    int(h_hop), 0, False, True)
        train_or_test_original_impl(self.model_path, location, "links_train.txt", "link_test_n.txt", "link_test_n.txt",
                                    int(h_hop), 0, False, True)
        end_time = time.time()
        print("the time for train_or_test_original_impl is:", end_time - start_time)
        #     "python ./muxlink/original.py --file-name trained_model --train-name links_train.txt --test-name links_test.txt --testneg-name link_test_n.txt --hop 3 --save-model > log_train_model.txt")

        # os.system(
        #     "python Main.py  --file-name " + location + " --train-name links_train.txt  --test-name links_test.txt --hop 3 --only-predict > log_test_pos.txt")
        #
        # os.system(
        #     "python Main.py  --file-name " + location + " --train-name links_train.txt  --test-name  link_test_n.txt --hop 3 --only-predict > log_test_neg.txt")
        
        start_time = time.time()
        break_cmd = "cd ../ml_data_" + circuit_name + "/ && perl break_DMUX.pl ./test_model/" + " 0.01 " + str(h_hop) + " " + str(key_size) + " " + circuit_name
        results = os.popen(break_cmd).read()
        x = results.split(" ")
        end_time = time.time()
        print("the time for break_DMUX.pl is:", end_time - start_time)

        print("results:", results)
        # sometime the prediction on 2 links is same; therefore, if the result is empty
        # we will random guess it to be 0 or 1 or directly to be 1 means that wrongly predict it to be 1
        if len(x) == 1:
            x = [1, 1, 1]
        return x[0], x[1], x[2]
    
    def attack_exact(self, bench_as_a_string, circuit_name, h_hop): # add a circuit name
        location = self.model_path + "test_model"

        if not os.path.exists(location):
            os.mkdir(location)

        f_feat = open(location + "/feat.txt", "w")
        f_cell = open(location + "/cell.txt", "w")
        f_count = open(location + "/count.txt", "w")
        f_link_test_neg = open(location + "/link_test_n.txt", "w")
        f_link_test = open(location + "/links_test.txt", "w")
        f_link_train = open(location + "/links_train_temp.txt", "w")
        f_link_train_f = open(location + "/links_train.txt", "w")
        c, key_size = verilog2gates_locked(f_link_train, f_link_train_f, f_link_test, f_link_test_neg,
                                           bench_as_a_string,
                                           f_feat, f_cell, f_count, True)
        f_feat.close()
        f_cell.close()
        f_count.close()
        f_link_test.close()
        f_link_test_neg.close()
        f_link_train.close()

        with open(location + "/links_test.txt") as f_a, open(location + "/links_train_temp.txt") as f_b:
            a_lines = set(f_a.read().splitlines())
            b_lines = set(f_b.read().splitlines())
            for line in b_lines:
                if line not in a_lines:
                    f_link_train_f.write(line + "\n")
        f_link_train_f.close()
        os.remove(location + "/links_train_temp.txt")
        ##Now it is the time to test

        # here to divid all the link in the links_test.txt and links_test_n.txt into several files
        # and then use the perl script to break the DMUX
        # the number of files is decided by each line of the links_test.txt and links_test_n.txt
        # divide the links_test.txt
       
        train_or_test_original_impl(self.model_path, location, "links_train.txt", "links_test.txt", "link_test_n.txt",
                                    int(h_hop), 0, False, True)
        train_or_test_original_impl(self.model_path, location, "links_train.txt", "link_test_n.txt", "link_test_n.txt",
                                    int(h_hop), 0, False, True)

        #     "python ./muxlink/original.py --file-name trained_model --train-name links_train.txt --test-name links_test.txt --testneg-name link_test_n.txt --hop 3 --save-model > log_train_model.txt")

        # os.system(
        #     "python Main.py  --file-name " + location + " --train-name links_train.txt  --test-name links_test.txt --hop 3 --only-predict > log_test_pos.txt")
        #
        # os.system(
        #     "python Main.py  --file-name " + location + " --train-name links_train.txt  --test-name  link_test_n.txt --hop 3 --only-predict > log_test_neg.txt")
        
        break_cmd = "cd ../ml_data_" + circuit_name + "/ && perl break_DMUX_exact.pl ./test_model/" + " 0.01 " + str(h_hop) + " " + str(key_size) + " " + circuit_name
        results = os.popen(break_cmd).read()
        x_temp = results.split(",")
        print("x_temp:", x_temp)
        kpa_temp = x_temp[0]
        uni_temp = [int(num) for num in x_temp[1].split()]
        wrong_temp = [int(num) for num in x_temp[2].split()]
        x = [kpa_temp, uni_temp, wrong_temp]
        print("results:", results)
        # sometime the prediction on 2 links is same; therefore, if the result is empty
        # we will random guess it to be 0 or 1 or directly to be 1 means that wrongly predict it to be 1
        if len(x) == 1:
            x = [1, None, None]
        return x[0], x[1], x[2]


    # design for the multi-muxlink attack
    # 1) extract the train files to get the link_trains.txt
    def attack_get_train_links(self, bench_as_a_string):
        location = self.model_path + "test_model"

        if not os.path.exists(location):
            os.mkdir(location)
        
        f_feat = open(location + "/feat.txt", "w")
        f_cell = open(location + "/cell.txt", "w")
        f_count = open(location + "/count.txt", "w")
        f_link_test_neg = open(location + "/link_test_n.txt", "w")
        f_link_test = open(location + "/links_test.txt", "w")
        f_link_train = open(location + "/links_train_temp.txt", "w")
        f_link_train_f = open(location + "/links_train.txt", "w")
        c, key_size = verilog2gates_locked(f_link_train, f_link_train_f, f_link_test, f_link_test_neg,
                                           bench_as_a_string,
                                           f_feat, f_cell, f_count, True)
        f_feat.close()
        f_cell.close()
        f_count.close()
        f_link_test.close()
        f_link_test_neg.close()
        f_link_train.close()

        with open(location + "/links_test.txt") as f_a, open(location + "/links_train_temp.txt") as f_b:
            a_lines = set(f_a.read().splitlines())
            b_lines = set(f_b.read().splitlines())
            for line in b_lines:
                if line not in a_lines:
                    f_link_train_f.write(line + "\n")
        f_link_train_f.close()
        os.remove(location + "/links_train_temp.txt")
        os.remove(location + "/link_test_n.txt")
        os.remove(location + "/links_test.txt")
    
    #2） based on the chosen solution, we get the link_test_n.txt and links_test.txt
    def attack_get_test_links(self, chosen_sol, sol_index, h_hop):
        location = self.model_path + "test_model"
        
        with open(location + "/links_test_" + str(sol_index) + ".txt", "w") as f_a, open(location + "/link_test_n_" + str(sol_index) + ".txt", "w") as f_b:
            for i in range(len(chosen_sol)):
                f_a.write(chosen_sol[i][0] + " " + chosen_sol[i][2] + "\n")
                f_a.write(chosen_sol[i][1] + " " + chosen_sol[i][3] + "\n")
                f_b.write(chosen_sol[i][0] + " " + chosen_sol[i][3] + "\n")
                f_b.write(chosen_sol[i][1] + " " + chosen_sol[i][2] + "\n")
        
        train_or_test_original_impl(self.model_path, location, "links_train.txt", "links_test_" + str(sol_index) + ".txt", "link_test_n_" + str(sol_index) + ".txt",
                                    int(h_hop), 0, False, True)
        train_or_test_original_impl(self.model_path, location, "links_train.txt", "link_test_n_" + str(sol_index) + ".txt", "link_test_n_" + str(sol_index) + ".txt",
                                    int(h_hop), 0, False, True)
        
        # after all the process, it will provide us all the files we need to break the DMUX
    
    # 3) merge all the results to two files: link_test_n_h__pred.txt and links_test_h__pred.txt
    def attack_merge_results(self, circuit_name, h_hop, key_size):
        location = self.model_path + "test_model"
        # find out all the file name at current location containing the "links_test" 
        # List all files in the specified directory
        all_files = os.listdir(location)
        all_test_pos_files = [f for f in all_files if "links_test" in f and "pred" in f]
        all_test_neg_files = [f for f in all_files if "link_test_n" in f and "pred" in f]
        print("all_test_pos_files:", all_test_pos_files)
        print("all_test_neg_files:", all_test_neg_files)
        f_link_test_neg = open(location + "/all_test_n_" + str(h_hop) + "__pred.txt", "w")
        f_link_test = open(location + "/all_test_" + str(h_hop) + "__pred.txt", "w")
        for pos_file in all_test_pos_files:
            pos_file_location = location + "/" + pos_file
            with open (pos_file_location, 'r') as f:
                f_link_test.write(f.read())
            # os.remove(pos_file_location)
            # print("what is the pos_file_location:", pos_file_location)
            # pos_file_location_no_pred = pos_file_location.replace("_" + str(h_hop) + "__pred", "")
            # print("what is the pos_file_location_no_pred:", pos_file_location_no_pred)
            # os.remove(pos_file_location_no_pred)
        for neg_file in all_test_neg_files:
            neg_file_location = location + "/" + neg_file
            with open (neg_file_location, 'r') as f:
                f_link_test_neg.write(f.read())
            # os.remove(neg_file_location)
            # print("what is the neg_file_location:", neg_file_location)
            # neg_file_location_no_pred = neg_file_location.replace("_" + str(h_hop) + "__pred", "")
            # print("what is the neg_file_location_no_pred:", neg_file_location_no_pred)
            # os.remove(neg_file_location_no_pred)
        f_link_test_neg.close()
        f_link_test.close()
        # change the name all_test to links_test and all_test_n to link_test_n
        os.system("mv " + location + "/all_test_n_" + str(h_hop) + "__pred.txt " + location + "/link_test_n_" + str(h_hop) + "__pred.txt")
        os.system("mv " + location + "/all_test_" + str(h_hop) + "__pred.txt " + location + "/links_test_" + str(h_hop) + "__pred.txt")
        break_cmd = "cd ../ml_data_" + circuit_name + "/ && perl break_DMUX.pl ./test_model/" + " 0.01 " + str(h_hop) + " " + str(key_size) + " " + circuit_name
        results = os.popen(break_cmd).read()
        x = results.split(" ")

        print("results:", results)
        # sometime the prediction on 2 links is same; therefore, if the result is empty
        # we will random guess it to be 0 or 1 or directly to be 1 means that wrongly predict it to be 1
        if len(x) == 1:
            x = [1, 1, 1]
        return x[0], x[1], x[2]





    # design for thread attack for muxlink
    def attack_thread(self, bench_as_a_string, circuit_name, file_num): # add a circuit name
        location = self.model_path + "test_model_" + str(file_num)

        if not os.path.exists(location):
            os.mkdir(location)

        f_feat = open(location + "/feat.txt", "w")
        f_cell = open(location + "/cell.txt", "w")
        f_count = open(location + "/count.txt", "w")
        f_link_test_neg = open(location + "/link_test_n.txt", "w")
        f_link_test = open(location + "/links_test.txt", "w")
        f_link_train = open(location + "/links_train_temp.txt", "w")
        f_link_train_f = open(location + "/links_train.txt", "w")
        c, key_size = verilog2gates_locked(f_link_train, f_link_train_f, f_link_test, f_link_test_neg,
                                           bench_as_a_string,
                                           f_feat, f_cell, f_count, True)
        f_feat.close()
        f_cell.close()
        f_count.close()
        f_link_test.close()
        f_link_test_neg.close()
        f_link_train.close()

        with open(location + "/links_test.txt") as f_a, open(location + "/links_train_temp.txt") as f_b:
            a_lines = set(f_a.read().splitlines())
            b_lines = set(f_b.read().splitlines())
            for line in b_lines:
                if line not in a_lines:
                    f_link_train_f.write(line + "\n")
        f_link_train_f.close()
        os.remove(location + "/links_train_temp.txt")
        ##Now it is the time to test

        train_or_test_original_impl(self.model_path, location, "links_train.txt", "links_test.txt", "link_test_n.txt",
                                    3, 0, False, True)
        train_or_test_original_impl(self.model_path, location, "links_train.txt", "link_test_n.txt", "link_test_n.txt",
                                    3, 0, False, True)

        # generated the new break_DMUX perl file to run the code
        # added by ZengWang
        print("what is self model path", self.model_path)
        created_perl = self.model_path + "break_DMUX_" + str(file_num) + ".pl"
        source_perl = self.model_path + "break_DMUX.pl"
        if not os.path.exists(created_perl):
            shutil.copy2(source_perl, created_perl)

        break_cmd = "cd ../ml_data_" + circuit_name + "/ && perl break_DMUX_" + str(file_num) + ".pl ./test_model/" + " 0.01 3 " + str(key_size) + " " + circuit_name + "_"+ str(file_num)
        results = os.popen(break_cmd).read()
        x = results.split(" ")

        print(str(file_num) + "_results:", results)
        # sometime the prediction on 2 links is same; therefore, if the result is empty
        # we will random guess it to be 0 or 1 or directly to be 1 means that wrongly predict it to be 1
        if len(x) == 1:
            x = [1, 1, 1]
        return x[0], x[1], x[2]
    
    def attack_large(self, bench_as_a_string, circuit_name, h_hop):
        import os, subprocess
        from pathlib import Path
        from time import perf_counter

        # ---- set up paths (prefer RAM disk if available) ----
        root = Path(self.model_path).resolve()
        on_ram = Path("/dev/shm")
        use_ram = on_ram.is_dir() and os.access("/dev/shm", os.W_OK)
        model_dir_disk = root / "test_model"

        if use_ram:
            model_dir_ram = on_ram / f"test_model_{os.getpid()}"
            model_dir = model_dir_ram
            model_dir.mkdir(parents=True, exist_ok=True)
            # ensure downstream tools see it at the expected disk path
            model_dir_disk.parent.mkdir(parents=True, exist_ok=True)
            if model_dir_disk.is_symlink() or model_dir_disk.exists():
                try:
                    model_dir_disk.unlink()
                except IsADirectoryError:
                    import shutil; shutil.rmtree(model_dir_disk)
            model_dir_disk.symlink_to(model_dir, target_is_directory=True)
        else:
            model_dir_disk.mkdir(parents=True, exist_ok=True)
            model_dir = model_dir_disk

        # Larger buffer for fewer syscalls
        BUF = 1 << 20  # 1 MiB

        # ---- generate files via verilog2gates_locked ----
        t0 = perf_counter()
        feat_p   = model_dir / "feat.txt"
        cell_p   = model_dir / "cell.txt"
        count_p  = model_dir / "count.txt"
        test_p   = model_dir / "links_test.txt"
        testn_p  = model_dir / "link_test_n.txt"
        train_tmp_p = model_dir / "links_train_temp.txt"
        train_p  = model_dir / "links_train.txt"

        with open(feat_p,  "w", buffering=BUF) as f_feat, \
            open(cell_p,  "w", buffering=BUF) as f_cell, \
            open(count_p, "w", buffering=BUF) as f_count, \
            open(test_p,  "w", buffering=BUF) as f_link_test, \
            open(testn_p, "w", buffering=BUF) as f_link_test_neg, \
            open(train_tmp_p, "w", buffering=BUF) as f_link_train_tmp, \
            open(train_p, "w", buffering=BUF) as f_link_train:

            c, key_size = verilog2gates_locked(
                f_link_train_tmp,  # temp train links
                f_link_train,      # final train links (base)
                f_link_test,
                f_link_test_neg,
                bench_as_a_string,
                f_feat, f_cell, f_count, True
            )
        print("verilog2gates_locked:", f"{perf_counter()-t0:.2f}s", flush=True)

        # ---- append only NEW train links (not in test) - streaming, no extra set for temp ----
        t1 = perf_counter()
        with open(test_p, "r", buffering=BUF) as f_a:
            test_set = {ln.rstrip("\n") for ln in f_a}
        with open(train_tmp_p, "r", buffering=BUF) as f_tmp, \
            open(train_p, "a", buffering=BUF) as f_train:
            wr = f_train.write
            for line in f_tmp:
                s = line.rstrip("\n")
                if s not in test_set:
                    wr(s + "\n")
        try:
            os.remove(train_tmp_p)
        except FileNotFoundError:
            pass
        print("dedupe train vs test:", f"{perf_counter()-t1:.2f}s", flush=True)

        # ---- train once, then predict twice (avoid retraining) ----
        # Adjust the two boolean flags if your signature differs:
        # (..., hop, seed, save_model, only_predict)
        t2 = perf_counter()
        train_or_test_original_impl(
            self.model_path, str(model_dir), "links_train.txt", "links_test.txt", "link_test_n.txt",
            int(h_hop), 0, True, False    # train & save model
        )
        # predict on negatives (re-use saved model)
        train_or_test_original_impl(
            self.model_path, str(model_dir), "links_train.txt", "link_test_n.txt", "link_test_n.txt",
            int(h_hop), 0, False, True    # only predict
        )
        print("train/test total:", f"{perf_counter()-t2:.2f}s", flush=True)

        # ---- run Perl step without shell, no extra 'cd' ----
        t3 = perf_counter()
        perl_cwd = Path("..") / f"ml_data_{circuit_name}"
        # break_DMUX.pl expects the test_model path relative to cwd; we pass the symlink path
        cp = subprocess.run(
            ["perl", "break_DMUX.pl", "./test_model/", "0.01", str(h_hop), str(key_size), circuit_name],
            cwd=str(perl_cwd), capture_output=True, text=True, check=True
        )
        results = cp.stdout.strip()
        print("break_DMUX.pl:", f"{perf_counter()-t3:.2f}s", flush=True)
        print("results:", results, flush=True)

        x = results.split()
        if len(x) == 1:
            x = [1, 1, 1]
        return x[0], x[1], x[2]



    def encode(self, locked_bench_str) -> KGSSolution:
        # Debug-lilas: implement the encoding to genotype from locked benchmark
        # why do we need the solution ID? If we are passing the locked bench_str.
        key_size = 0
        Dict_gates = {'xor': [0, 1, 0, 0, 0, 0, 0, 0],
                      'XOR': [0, 1, 0, 0, 0, 0, 0, 0],
                      'OR': [0, 0, 1, 0, 0, 0, 0, 0],
                      'or': [0, 0, 1, 0, 0, 0, 0, 0],
                      'XNOR': [0, 0, 0, 1, 0, 0, 0, 0],
                      'xnor': [0, 0, 0, 1, 0, 0, 0, 0],
                      'and': [0, 0, 0, 0, 1, 0, 0, 0],
                      'AND': [0, 0, 0, 0, 1, 0, 0, 0],
                      'nand': [0, 0, 0, 0, 0, 1, 0, 0],
                      'buf': [0, 0, 0, 0, 0, 0, 0, 1],
                      'BUF': [0, 0, 0, 0, 0, 0, 0, 1],
                      'NAND': [0, 0, 0, 0, 0, 1, 0, 0],
                      'not': [0, 0, 0, 0, 0, 0, 1, 0],
                      'NOT': [0, 0, 0, 0, 0, 0, 1, 0],
                      'nor': [1, 0, 0, 0, 0, 0, 0, 0],
                      'NOR': [1, 0, 0, 0, 0, 0, 0, 0],
                      }
        G = nx.DiGraph()
        ML_count = 0
        regex = "\s*(\S+)\s*=\s*(BUF|NOT)\((\S+)\)\s*"
        for output, function, net_str in re.findall(regex, locked_bench_str, flags=re.I | re.DOTALL):
            input = net_str.replace(" ", "")
            ## Zeng added: debug testing
            # print(output, ML_count)
            G.add_edge(input, output)
            G.nodes[output]['gate'] = function
            G.nodes[output]['count'] = ML_count
            ML_count += 1
        regex = "(\S+)\s*=\s*(OR|XOR|AND|NAND|XNOR|NOR|AOI\d*|OAI\d*)\((.+?)\)\s*"
        for output, function, net_str in re.findall(regex, locked_bench_str, flags=re.I | re.DOTALL):
            nets = net_str.replace(" ", "").replace("\n", "").replace("\t", "").split(",")
            inputs = nets
            G.add_edges_from((net, output) for net in inputs)
            ## Zeng added: debug testing
            # print(output, ML_count)
            G.nodes[output]['gate'] = function
            G.nodes[output]['count'] = ML_count
            ML_count += 1
        # print(list(G.nodes(data="count")))
        for n in G.nodes():
            if 'gate' not in G.nodes[n]:
                G.nodes[n]['gate'] = 'input'
                # why is there no count for the keyinput and original inputs
        for n in G.nodes:
            G.nodes[n]['output'] = False
        out_regex = "OUTPUT\((.+?)\)\n"
        for net_str in re.findall(out_regex, locked_bench_str, flags=re.I | re.DOTALL):
            nets = net_str.replace(" ", "").replace("\n", "").replace("\t", "").split(",")
            for net in nets:
                if net not in G:
                    print("Output " + net + " is Float")
                else:
                    G.nodes[net]['output'] = True
        regex = "#key=(\d+)\s*"
        # K_list = np.ones(key_size)
        # K_list = []
        for key_bits in re.findall(regex, locked_bench_str, flags=re.I | re.DOTALL):
            key_size = len(key_bits)
            # print("key_size", key_size)
            # print(key_size)
            # print("Key size is "+str(key_size))
            i = 0
            K_list = np.ones(key_size)
            for bit in key_bits:
                K_list[i] = int(bit)
                i = i + 1
        # print("G node number:", G.nodes())
        regex = "(\S+)\s*=\s*(MUX)\((.+?)\)\s*"
        listoflists = []
        for i in range(key_size):
            listoflists.append(['X', 'X', 'X', 'X', 'X', i])
        kgss = KGSSolution()  # create kgs object 12/06
                              # new updated -> remove the solution id
        for output, function, net_str in re.findall(regex, locked_bench_str, flags=re.I | re.DOTALL):
            nets = net_str.replace(" ", "").replace("\n", "").replace("\t", "").split(",")
            inputs = nets
            output_x = output.replace('_from_mux', '')
            regex_key = "keyinput(\d+)"
            for key_bit in re.findall(regex_key, inputs[0], flags=re.I | re.DOTALL):
                key_bit_value = K_list[int(key_bit)]
                correct = ""
                false = ""
                a_list = listoflists[int(key_bit)]  # get the encoding of this key_bit
                # added by 12/08
                # add new element to show the keyinput value
                # no need to shuffle it based on the 0/1
                # correct = inputs[1]
                # false = inputs[2]
                if key_bit_value == 0:
                    correct = inputs[1]
                    false = inputs[2]
                    # g1 assignment, X values
                    if a_list[0] == 'X':
                        a_list[0] = str(G.nodes[correct]['count'])
                        a_list[2] = str(G.nodes[output_x]['count'])
                        a_list[1] = str(G.nodes[false]['count'])
                        a_list[4] = str(int(key_bit_value))
                    else:
                        a_list[3] = str(G.nodes[output_x]['count'])
                    listoflists[int(key_bit)] = a_list
                else:
                    # here keyinput value is 1
                    correct = inputs[2]
                    false = inputs[1]
                    if a_list[0] == 'X':
                        a_list[0] = str(G.nodes[correct]['count'])
                        a_list[2] = str(G.nodes[output_x]['count'])
                        a_list[1] = str(G.nodes[false]['count'])
                        a_list[4] = str(int(key_bit_value))
                    else:
                        a_list[3] = str(G.nodes[output_x]['count'])
                    listoflists[int(key_bit)] = a_list

        encoding = ""

        for x in listoflists:
            kgss.add_entry(x[0], x[1], x[2], x[3], x[4], x[5])

            encoding += str(x) + ", "
        kgss.register_graph(G) ## added by 01/11
        
        return kgss

    def get_FMulti_FSingle(self):
        # TODO-lilas: return the list of Fmulti and Fsingle based on solution_id
        return self.F_multi, self.F_single

    def decode(self, bench_as_a_string, kgss):
        # get the graph from the original benchmark
        G = verilog2gates(bench_as_a_string)
        # print(G)
        encoding_list = kgss.data
        key_len = len(encoding_list)
        K_list = []  # store the key value
        for index in range(key_len):
            K_list.append(encoding_list[index][4])
        ###################################################################
        # generate the key string and key input in the beginning of the file
        ###################################################################
        locked_file = ""
        i = 0
        locked_file = locked_file + "#key="
        while i < key_len:
            locked_file = locked_file + str(int(K_list[i]))
            i = i + 1
        locked_file = locked_file + ("\n")
        i = 0
        while i < key_len:
            locked_file = locked_file + "INPUT(keyinput" + str(i) + ")\n"
            i = i + 1
        ###################################################################
        # get the information from the kgss object about the f and g
        # reconstruct myDict, selected_g
        ###################################################################
        myDict = {}
        selected_g = []
        # format:[('Gxx1', index1), ('Gxx2', index2)]
        # start_time = time.time()
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        # end_time = time.time()
        # print("time for dict:", end_time - start_time)

        start_time = time.time()
        for item in encoding_list:
            f1_name = G_info_updated[int(item[0])]
            f2_name = G_info_updated[int(item[1])]
            g1_name = G_info_updated[int(item[2])]
            g2_name = G_info_updated[int(item[3])]
            key_index = int(item[5])
            myDict[g2_name] = [f2_name, f1_name, key_index]  # f1-> first input f2-> second input
            myDict[g1_name] = [f1_name, f2_name, key_index]
            selected_g.append(g1_name)
            selected_g.append(g2_name)
        end_time = time.time()
        print("time for part1:", end_time - start_time)
        K_list = [int(i) for i in K_list]
        # print(myDict)
        # print(selected_g)
        # ###################################################################
        # # continue to construct the locked circuits based on the same idea
        # # like locking function
        # ###################################################################
        count = 0
        detected = 0
        start_time = time.time()
        # print("what happend:", bench_as_a_string.split("\n"))
        for line in bench_as_a_string.split("\n"):
            count += 1
            line = line.strip()
            # print("line:", line)
            if any(ext + " =" in line for ext in selected_g):  # id -> gate_name -> search
                detected = detected + 1
                regex = "(\S+)\s*=\s*(NOT|BUF|OR|XOR|AND|NAND|XNOR|NOR)\((.+?)\)\s*"
                for output, function, net_str in re.findall(regex, line, flags=re.I | re.DOTALL):
                    if output in myDict.keys():
                        my_f1 = myDict[output][0]
                        my_f2 = myDict[output][1]
                        my_key = myDict[output][2]
                        line = line.replace(my_f1 + ",", output + "_from_mux,")
                        line = line.replace(my_f1 + ")", output + "_from_mux)")
                        locked_file = locked_file + (line + "\n")
                        if K_list[my_key] == 0:
                            locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
                                my_key) + ", " + my_f1 + ", " + my_f2 + ")\n"
                        else:

                            locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
                                my_key) + ", " + my_f2 + ", " + my_f1 + ")\n"
                    else:
                        locked_file = locked_file + line + "\n"
            else:
                locked_file = locked_file + line + "\n"
        end_time = time.time()
        print("time for part2:", end_time - start_time)
        print("detected:", detected)
        # text_file = open(self.model_path + "check_locked_" + "test1.txt", "w")
        # text_file.write(locked_file)
        # text_file.close()

        return locked_file
    
    def decode_spe(self, bench_as_a_string, kgss):
        # get the graph from the original benchmark
        G = verilog2gates(bench_as_a_string)
        # print(G)
        encoding_list = kgss.data
        key_len = len(encoding_list)
        K_list = []  # store the key value
        for index in range(key_len):
            K_list.append(encoding_list[index][4])
        ###################################################################
        # generate the key string and key input in the beginning of the file
        ###################################################################
        locked_file = ""
        i = 0
        locked_file = locked_file + "#key="
        while i < key_len:
            locked_file = locked_file + str(int(K_list[i]))
            i = i + 1
        locked_file = locked_file + ("\n")
        i = 0
        while i < key_len:
            locked_file = locked_file + "INPUT(keyinput" + str(i) + ")\n"
            i = i + 1
        ###################################################################
        # get the information from the kgss object about the f and g
        # reconstruct myDict, selected_g
        ###################################################################
        myDict = {}
        selected_g = []
        # format:[('Gxx1', index1), ('Gxx2', index2)]
        # start_time = time.time()
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        # end_time = time.time()
        # print("time for dict:", end_time - start_time)

        start_time = time.time()
        for item in encoding_list:
            f1_name = G_info_updated[int(item[0])]
            f2_name = G_info_updated[int(item[1])]
            g1_name = G_info_updated[int(item[2])]
            g2_name = G_info_updated[int(item[3])]
            key_index = int(item[5])
            myDict[g2_name] = [f2_name, f1_name, key_index]  # f1-> first input f2-> second input
            myDict[g1_name] = [f1_name, f2_name, key_index]
            selected_g.append(g1_name)
            selected_g.append(g2_name)
        end_time = time.time()
        print("time for part1:", end_time - start_time)
        K_list = [int(i) for i in K_list]
        # print(myDict)
        print(selected_g)
        print(len(selected_g))
        print(len(list(set(selected_g))))
        # ###################################################################
        # # continue to construct the locked circuits based on the same idea
        # # like locking function
        # ###################################################################
        count = 0
        detected = 0
        passed_lines= []
        start_time = time.time()
        # print("what happend:", bench_as_a_string.split("\n"))
        for line in bench_as_a_string.split("\n"):
            count += 1
            line = line.strip()
            # print("line:", line)
            if any(re.search(rf"{re.escape(ext)}\s*=", line) for ext in selected_g):  # id -> gate_name -> search
                detected = detected + 1
                passed_lines.append(line)
                regex = "(\S+)\s*=\s*(NOT|BUF|OR|XOR|AND|NAND|XNOR|NOR)\((.+?)\)\s*"
                for output, function, net_str in re.findall(regex, line, flags=re.I | re.DOTALL):
                    if output in myDict.keys():
                        my_f1 = myDict[output][0]
                        my_f2 = myDict[output][1]
                        my_key = myDict[output][2]
                        line = line.replace(my_f1 + ",", output + "_from_mux,")
                        line = line.replace(my_f1 + ")", output + "_from_mux)")
                        locked_file = locked_file + (line + "\n")
                        if K_list[my_key] == 0:
                            locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
                                my_key) + ", " + my_f1 + ", " + my_f2 + ")\n"
                        else:

                            locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
                                my_key) + ", " + my_f2 + ", " + my_f1 + ")\n"
                    else:
                        locked_file = locked_file + line + "\n"
            else:
                # for ext in selected_g:
                #     if ext in line:
                #         passed_selected_g.append(ext)
                locked_file = locked_file + line + "\n"
        end_time = time.time()
        print("time for part2:", end_time - start_time)
        print("detected:", detected)
        # print("passed_selected_g:", passed_selected_g)
        # print("len(passed_selected_g):", len(passed_selected_g))
        # text_file = open(self.model_path + "check_locked_" + "test1.txt", "w")
        # text_file.write(locked_file)
        # text_file.close()
        passed_gs = []
        for line in passed_lines:
            # print(line)
            g = line.split("=")[0].strip(" ")
            passed_gs.append(g)
        # find the difference of passed_gs and selected_g
        # print("passed_gs:", passed_gs)
        # print("selected_g:", selected_g)
        diff = list(set(selected_g).difference(set(passed_gs)))
        print("diff:", diff)


        return locked_file
    
    def decode_omla(self, bench_as_a_string, kgss, key_start=0):
        # get the graph from the original benchmark
        G = verilog2gates(bench_as_a_string)
        # print(G)
        encoding_list = kgss.data
        key_len = len(encoding_list)
        K_list = []  # store the key value
        for index in range(key_len):
            K_list.append(encoding_list[index][4])
            K_list.append(str(1-int(encoding_list[index][4])))
        ###################################################################
        # generate the key string and key input in the beginning of the file
        ###################################################################
        locked_file = ""
        i = 0
        locked_file = locked_file + "#key="
        while i < key_len*2:
            locked_file = locked_file + str(int(K_list[i]))
            i = i + 1
        locked_file = locked_file + ("\n")
        i = 0
        while i < key_len*2:
            locked_file = locked_file + "INPUT(KEYINPUT" + str(i+key_start) + ")\n"
            i = i + 1
        ###################################################################
        # get the information from the kgss object about the f and g
        # reconstruct myDict, selected_g
        ###################################################################
        myDict = {}
        selected_g = []
        # format:[('Gxx1', index1), ('Gxx2', index2)]
        # start_time = time.time()
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        # end_time = time.time()
        # print("time for dict:", end_time - start_time)

        start_time = time.time()
        for item in encoding_list:
            f1_name = G_info_updated[int(item[0])]
            f2_name = G_info_updated[int(item[1])]
            g1_name = G_info_updated[int(item[2])]
            g2_name = G_info_updated[int(item[3])]
            key_index = int(item[5])*2 
            myDict[g2_name] = [f2_name, f1_name, key_index]  # f1-> first input f2-> second input
            myDict[g1_name] = [f1_name, f2_name, key_index+1]
            selected_g.append(g1_name)
            selected_g.append(g2_name)
        end_time = time.time()
        print("time for part1:", end_time - start_time)
        K_list = [int(i) for i in K_list]
        print("myDict", myDict)
        # ###################################################################
        # # continue to construct the locked circuits based on the same idea
        # # like locking function
        # ###################################################################
        count = 0
        detected = 0
        start_time = time.time()
        for line in bench_as_a_string.split("\n"):
            count += 1
            line = line.strip()
            if any(ext + " =" in line for ext in selected_g):  # id -> gate_name -> search
                detected = detected + 1
                regex = "(\S+)\s*=\s*(NOT|BUF|OR|XOR|AND|NAND|XNOR|NOR|AOI\d*|OAI\d*)\((.+?)\)\s*"
                for output, function, net_str in re.findall(regex, line, flags=re.I | re.DOTALL):
                    if output in myDict.keys():
                        my_f1 = myDict[output][0]
                        # my_f2 = myDict[output][1]
                        my_key = myDict[output][2]
                        flag = True
                        if flag:
                        # X_rand = random.randint(0, 1)
                        # if X_rand == 0:
                            if K_list[my_key] == 0:
                                line = line.replace(my_f1 + ",", output + "_from_XOR,")
                                line = line.replace(my_f1 + ")", output + "_from_XOR)")
                            else:
                                line = line.replace(my_f1 + ",", output + "_from_XNOR,")
                                line = line.replace(my_f1 + ")", output + "_from_XNOR)")
                            locked_file = locked_file + (line + "\n")
                        
                        # if X_rand == 0:
                            if K_list[my_key] == 0:
                                locked_file = locked_file + output + "_from_XOR = XOR(KEYINPUT" + str(
                                    my_key+key_start) + ", " + my_f1 + ")\n"
                            else:

                                locked_file = locked_file + output + "_from_XNOR = XNOR(KEYINPUT" + str(
                                    my_key+key_start) + ", " + my_f1 + ")\n"
                        else:
                            if K_list[my_key] == 0:
                                line = line.replace(my_f1 + ",", output + "_from_XNOR,")
                                line = line.replace(my_f1 + ")", output + "_from_XNOR)")
                            else:
                                line = line.replace(my_f1 + ",", output + "_from_XOR,")
                                line = line.replace(my_f1 + ")", output + "_from_XOR)")
                            locked_file = locked_file + (line + "\n")
                            if K_list[my_key] == 1:
                                locked_file = locked_file + output + "_from_XOR = XOR(KEYINPUT" + str(
                                    my_key+key_start) + "_temp, " + my_f1 + ")\n"
                                locked_file = locked_file + "KEYINPUT" + str(my_key+key_start) + "_temp = NOT(KEYINPUT" + str(my_key+key_start) + ")\n"
                            else:

                                locked_file = locked_file + output + "_from_XNOR = XNOR(KEYINPUT" + str(
                                    my_key+key_start) + "_temp, "  +  my_f1 + ")\n"
                                locked_file = locked_file + "KEYINPUT" + str(my_key+key_start) + "_temp = NOT(KEYINPUT" + str(my_key+key_start) + ")\n"
                    else:
                        locked_file = locked_file + line + "\n"
            else:
                locked_file = locked_file + line + "\n"
        end_time = time.time()
        print("time for part2:", end_time - start_time)
        # text_file = open(self.model_path + "check_locked_" + "test1.txt", "w")
        # text_file.write(locked_file)
        # text_file.close()

        return locked_file
    
    def decode_omla_large(self, bench_as_a_string, kgss, key_start=0):
        # get the graph from the original benchmark
        G = verilog2gates(bench_as_a_string)
        # print(G)
        encoding_list = kgss.data
        key_len = len(encoding_list)
        K_list = []  # store the key value
        for index in range(key_len):
            K_list.append(encoding_list[index][4])
            K_list.append(str(1-int(encoding_list[index][4])))
        ###################################################################
        # generate the key string and key input in the beginning of the file
        ###################################################################
        locked_file = ""
        i = 0
        locked_file = locked_file + "#key="
        while i < key_len*2:
            locked_file = locked_file + str(int(K_list[i]))
            i = i + 1
        locked_file = locked_file + ("\n")
        i = 0
        while i < key_len*2:
            locked_file = locked_file + "INPUT(KEYINPUT" + str(i+key_start) + ")\n"
            i = i + 1
        ###################################################################
        # get the information from the kgss object about the f and g
        # reconstruct myDict, selected_g
        ###################################################################
        myDict = {}
        selected_g = []
        # format:[('Gxx1', index1), ('Gxx2', index2)]
        # start_time = time.time()
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        # end_time = time.time()
        # print("time for dict:", end_time - start_time)

        start_time = time.time()
        for item in encoding_list:
            f1_name = G_info_updated[int(item[0])]
            f2_name = G_info_updated[int(item[1])]
            g1_name = G_info_updated[int(item[2])]
            g2_name = G_info_updated[int(item[3])]
            key_index = int(item[5])*2 
            myDict[g2_name] = [f2_name, f1_name, key_index]  # f1-> first input f2-> second input
            myDict[g1_name] = [f1_name, f2_name, key_index+1]
            selected_g.append(g1_name)
            selected_g.append(g2_name)
        end_time = time.time()
        print("time for part1:", end_time - start_time)
        K_list = [int(i) for i in K_list]
        # print("myDict", myDict)
        # print(myDict)
        # ###################################################################
        # # continue to construct the locked circuits based on the same idea
        # # like locking function
        # ###################################################################
        count = 0
        detected = 0
        start_time = time.time()

        locked_file_rest = self.parallel_process_omla(bench_as_a_string, selected_g, myDict, K_list, key_start)
        locked_file = locked_file + locked_file_rest
        end_time = time.time()
        print("time for part2:", end_time - start_time)
        # text_file = open(self.model_path + "check_locked_" + "test1.txt", "w")
        # text_file.write(locked_file)
        # text_file.close()

        return locked_file
    
    def decode_omla_update(self, bench_as_a_string, kgss, key_start=0):
        # get the graph from the original benchmark
        G = verilog2gates(bench_as_a_string)
        # print(G)
        encoding_list = kgss.data
        key_len = len(encoding_list)
        K_list = []  # store the key value
        for index in range(key_len):
            K_list.append(encoding_list[index][4])
            K_list.append(str(1-int(encoding_list[index][4])))
        ###################################################################
        # generate the key string and key input in the beginning of the file
        ###################################################################
        locked_file = ""
        i = 0
        locked_file = locked_file + "#key="
        while i < key_len*2:
            locked_file = locked_file + str(int(K_list[i]))
            i = i + 1
        locked_file = locked_file + ("\n")
        i = 0
        while i < key_len*2:
            locked_file = locked_file + "INPUT(KEYINPUT" + str(i+key_start) + ")\n"
            i = i + 1
        ###################################################################
        # get the information from the kgss object about the f and g
        # reconstruct myDict, selected_g
        ###################################################################
        myDict = {}
        selected_g = []
        # format:[('Gxx1', index1), ('Gxx2', index2)]
        # start_time = time.time()
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        # end_time = time.time()
        # print("time for dict:", end_time - start_time)

        start_time = time.time()
        for item in encoding_list:
            f1_name = G_info_updated[int(item[0])]
            f2_name = G_info_updated[int(item[1])]
            g1_name = G_info_updated[int(item[2])]
            g2_name = G_info_updated[int(item[3])]
            key_index = int(item[5])*2 
            myDict[g2_name] = [f2_name, f1_name, key_index]  # f1-> first input f2-> second input
            myDict[g1_name] = [f1_name, f2_name, key_index+1]
            selected_g.append(g1_name)
            selected_g.append(g2_name)
        end_time = time.time()
        print("time for part1:", end_time - start_time)
        K_list = [int(i) for i in K_list]
        print("myDict", myDict)
        # ###################################################################
        # # continue to construct the locked circuits based on the same idea
        # # like locking function
        # ###################################################################
        count = 0
        detected = 0
        start_time = time.time()
        for line in bench_as_a_string.split("\n"):
            count += 1
            line = line.strip()
            if any(ext + " =" in line for ext in selected_g):  # id -> gate_name -> search
                detected = detected + 1
                regex = "(\S+)\s*=\s*(NOT|BUF|OR|XOR|AND|NAND|XNOR|NOR|AOI\d*|OAI\d*)\((.+?)\)\s*"
                for output, function, net_str in re.findall(regex, line, flags=re.I | re.DOTALL):
                    if output in myDict.keys():
                        my_f1 = myDict[output][0]
                        # my_f2 = myDict[output][1]
                        my_key = myDict[output][2]
                        flag = True
                        if flag:
                        # X_rand = random.randint(0, 1)
                        # if X_rand == 0:
                            if K_list[my_key] == 0:
                                line = line.replace(my_f1 + ",", output + "_from_XOR,")
                                line = line.replace(my_f1 + ")", output + "_from_XOR)")
                            else:
                                line = line.replace(my_f1 + ",", output + "_from_XNOR,")
                                line = line.replace(my_f1 + ")", output + "_from_XNOR)")
                            locked_file = locked_file + (line + "\n")
                        
                        # if X_rand == 0:
                            if K_list[my_key] == 0:
                                locked_file = locked_file + output + "_from_XOR = XOR(KEYINPUT" + str(
                                    my_key+key_start) + ", " + my_f1 + ")\n"
                            else:

                                locked_file = locked_file + output + "_from_XNOR = XNOR(KEYINPUT" + str(
                                    my_key+key_start) + ", " + my_f1 + ")\n"
                        else:
                            if K_list[my_key] == 0:
                                line = line.replace(my_f1 + ",", output + "_from_XNOR,")
                                line = line.replace(my_f1 + ")", output + "_from_XNOR)")
                            else:
                                line = line.replace(my_f1 + ",", output + "_from_XOR,")
                                line = line.replace(my_f1 + ")", output + "_from_XOR)")
                            locked_file = locked_file + (line + "\n")
                            if K_list[my_key] == 1:
                                locked_file = locked_file + output + "_from_XOR = XOR(KEYINPUT" + str(
                                    my_key+key_start) + "_temp, " + my_f1 + ")\n"
                                locked_file = locked_file + "KEYINPUT" + str(my_key+key_start) + "_temp = NOT(KEYINPUT" + str(my_key+key_start) + ")\n"
                            else:

                                locked_file = locked_file + output + "_from_XNOR = XNOR(KEYINPUT" + str(
                                    my_key+key_start) + "_temp, "  +  my_f1 + ")\n"
                                locked_file = locked_file + "KEYINPUT" + str(my_key+key_start) + "_temp = NOT(KEYINPUT" + str(my_key+key_start) + ")\n"
                    else:
                        locked_file = locked_file + line + "\n"
            else:
                locked_file = locked_file + line + "\n"
        end_time = time.time()
        print("time for part2:", end_time - start_time)
        # text_file = open(self.model_path + "check_locked_" + "test1.txt", "w")
        # text_file.write(locked_file)
        # text_file.close()

        return locked_file
    
    def process_chunk(self, args):
        chunk, selected_g, myDict, K_list = args
        processed_lines = []
        # selected_g_set = {'example_gate1', 'example_gate2'}  # Your set of gates
        regex = r"(\S+)\s*=\s*(NOT|BUF|OR|XOR|AND|NAND|XNOR|NOR)\((.+?)\)\s*"
        pattern = re.compile(regex, flags=re.I | re.DOTALL)
        
        for line in chunk.split("\n"):
            line = line.strip()
            if any(ext + " =" in line for ext in selected_g):
                matches = pattern.findall(line)
                if matches:
                    for output, function, net_str in matches:
                        if output in myDict:
                            my_f1, my_f2, my_key = myDict[output]
                            replacement = output + "_from_mux"
                            line = line.replace(my_f1 + ",", replacement + ",")
                            line = line.replace(my_f1 + ")", replacement + ")")
                            mux_line = f"{replacement} = MUX(keyinput{my_key}, {my_f2 if K_list[my_key] else my_f1}, {my_f1 if K_list[my_key] else my_f2})"
                            processed_lines.append(line)
                            processed_lines.append(mux_line)
                        else:
                            processed_lines.append(line)
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)
        
        return "\n".join(processed_lines)
    
    def process_chunk_omla(self, args):
        chunk, selected_g, myDict, K_list, key_start = args
        processed_lines = []
        # selected_g_set = {'example_gate1', 'example_gate2'}  # Your set of gates
        regex = "(\S+)\s*=\s*(NOT|BUF|OR|XOR|AND|NAND|XNOR|NOR|AOI\d*|OAI\d*)\((.+?)\)\s*"
        pattern = re.compile(regex, flags=re.I | re.DOTALL)
        
        for line in chunk.split("\n"):
            line = line.strip()
            if any(ext + " =" in line for ext in selected_g):
                matches = pattern.findall(line)
                if matches:
                    for output, function, net_str in matches:
                        if output in myDict:
                            my_f1, my_f2, my_key = myDict[output]
                            # replacement = output + "_from_mux"
                            # line = line.replace(my_f1 + ",", replacement + ",")
                            # line = line.replace(my_f1 + ")", replacement + ")")
                            # mux_line = f"{replacement} = MUX(keyinput{my_key}, {my_f2 if K_list[my_key] else my_f1}, {my_f1 if K_list[my_key] else my_f2})"
                            # processed_lines.append(line)
                            # processed_lines.append(mux_line)
                            if K_list[my_key] == 0:
                                line = line.replace(my_f1 + ",", output + "_from_XOR,")
                                line = line.replace(my_f1 + ")", output + "_from_XOR)")
                            else:
                                line = line.replace(my_f1 + ",", output + "_from_XNOR,")
                                line = line.replace(my_f1 + ")", output + "_from_XNOR)")
                            # locked_file = locked_file + (line + "\n")
                            processed_lines.append(line)
                        
                        # if X_rand == 0:
                            if K_list[my_key] == 0:
                                xor_line = output + "_from_XOR = XOR(KEYINPUT" + str(
                                    my_key+key_start) + ", " + my_f1 + ")\n"
                            else:

                                xor_line = output + "_from_XNOR = XNOR(KEYINPUT" + str(
                                    my_key+key_start) + ", " + my_f1 + ")\n"
                            processed_lines.append(xor_line)
                        else:
                            processed_lines.append(line)
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)
        
        return "\n".join(processed_lines)

    def chunkify(self, text, num_chunks):
        lines = text.split('\n')
        for i in range(0, len(lines), num_chunks):
            yield '\n'.join(lines[i:i + num_chunks])
    
    def parallel_process(self, file_data, selected_g, myDict, K_list):
        num_processes = mp.cpu_count()
        pool = mp.Pool(num_processes)
        chunk_size = len(file_data.split('\n')) // num_processes
        chunks = list(self.chunkify(file_data, chunk_size))
        tasks = [(chunk, selected_g, myDict, K_list) for chunk in chunks]
        results = pool.map(self.process_chunk, tasks)
        pool.close()
        pool.join()
        return '\n'.join(results)
    
    def parallel_process_omla(self, file_data, selected_g, myDict, K_list, key_start):
        num_processes = mp.cpu_count()
        pool = mp.Pool(num_processes)
        chunk_size = len(file_data.split('\n')) // num_processes
        chunks = list(self.chunkify(file_data, chunk_size))
        tasks = [(chunk, selected_g, myDict, K_list, key_start) for chunk in chunks]
        results = pool.map(self.process_chunk_omla, tasks)
        pool.close()
        pool.join()
        return '\n'.join(results)
    

    def decode_large(self, bench_as_a_string, kgss):
        # get the graph from the original benchmark
        G = verilog2gates(bench_as_a_string)
        # print(G)
        encoding_list = kgss.data
        key_len = len(encoding_list)
        K_list = []  # store the key value
        for index in range(key_len):
            K_list.append(encoding_list[index][4])
        ###################################################################
        # generate the key string and key input in the beginning of the file
        ###################################################################
        locked_file = ""
        i = 0
        locked_file = locked_file + "#key="
        while i < key_len:
            locked_file = locked_file + str(int(K_list[i]))
            i = i + 1
        locked_file = locked_file + ("\n")
        i = 0
        while i < key_len:
            locked_file = locked_file + "INPUT(keyinput" + str(i) + ")\n"
            i = i + 1
        ###################################################################
        # get the information from the kgss object about the f and g
        # reconstruct myDict, selected_g
        ###################################################################
        myDict = {}
        selected_g = []
        # format:[('Gxx1', index1), ('Gxx2', index2)]
        # start_time = time.time()
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        # end_time = time.time()
        # print("time for dict:", end_time - start_time)

        start_time = time.time()
        for item in encoding_list:
            f1_name = G_info_updated[int(item[0])]
            f2_name = G_info_updated[int(item[1])]
            g1_name = G_info_updated[int(item[2])]
            g2_name = G_info_updated[int(item[3])]
            key_index = int(item[5])
            myDict[g2_name] = [f2_name, f1_name, key_index]  # f1-> first input f2-> second input
            myDict[g1_name] = [f1_name, f2_name, key_index]
            selected_g.append(g1_name)
            selected_g.append(g2_name)
        end_time = time.time()
        print("time for part1:", end_time - start_time)
        K_list = [int(i) for i in K_list]
        # print(myDict)
        # ###################################################################
        # # continue to construct the locked circuits based on the same idea
        # # like locking function
        # ###################################################################
        count = 0
        detected = 0
        start_time = time.time()
        # for line in bench_as_a_string.split("\n"):
        #     count += 1
        #     line = line.strip()
        #     if any(ext + " =" in line for ext in selected_g):  # id -> gate_name -> search
        #         detected = detected + 1
        #         regex = "(\S+)\s*=\s*(NOT|BUF|OR|XOR|AND|NAND|XNOR|NOR)\((.+?)\)\s*"
        #         for output, function, net_str in re.findall(regex, line, flags=re.I | re.DOTALL):
        #             if output in myDict.keys():
        #                 my_f1 = myDict[output][0]
        #                 my_f2 = myDict[output][1]
        #                 my_key = myDict[output][2]
        #                 line = line.replace(my_f1 + ",", output + "_from_mux,")
        #                 line = line.replace(my_f1 + ")", output + "_from_mux)")
        #                 locked_file = locked_file + (line + "\n")
        #                 if K_list[my_key] == 0:
        #                     locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
        #                         my_key) + ", " + my_f1 + ", " + my_f2 + ")\n"
        #                 else:

        #                     locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
        #                         my_key) + ", " + my_f2 + ", " + my_f1 + ")\n"
        #             else:
        #                 locked_file = locked_file + line + "\n"
        #     else:
        #         locked_file = locked_file + line + "\n"
        locked_file_rest = self.parallel_process(bench_as_a_string, selected_g, myDict, K_list)
        locked_file = locked_file + locked_file_rest
        end_time = time.time()
        print("time for part2:", end_time - start_time)
        # text_file = open(self.model_path + "check_locked_" + "test1.txt", "w")
        # text_file.write(locked_file)
        # text_file.close()

        return locked_file
    
    def decode_large_scope(self, bench_as_a_string, kgss):
        # get the graph from the original benchmark
        G = verilog2gates(bench_as_a_string)
        # print(G)
        encoding_list = kgss.data
        key_len = len(encoding_list)
        K_list = []  # store the key value
        for index in range(key_len):
            K_list.append(encoding_list[index][4])
        ###################################################################
        # generate the key string and key input in the beginning of the file
        ###################################################################
        locked_file = ""
        i = 0
        locked_file = locked_file + "#key="
        while i < key_len:
            locked_file = locked_file + str(int(K_list[i]))
            i = i + 1
        locked_file = locked_file + ("\n")
        i = 0
        while i < key_len:
            locked_file = locked_file + "INPUT(keyinput" + str(i) + ")\n"
            i = i + 1
        ###################################################################
        # get the information from the kgss object about the f and g
        # reconstruct myDict, selected_g
        ###################################################################
        myDict = {}
        selected_g = []
        # format:[('Gxx1', index1), ('Gxx2', index2)]
        # start_time = time.time()
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        # end_time = time.time()
        # print("time for dict:", end_time - start_time)

        start_time = time.time()
        for item in encoding_list:
            f1_name = G_info_updated[int(item[0])]
            f2_name = G_info_updated[int(item[1])]
            g1_name = G_info_updated[int(item[2])]
            # g2_name = G_info_updated[int(item[3])]
            key_index = int(item[5])
            # myDict[g2_name] = [f2_name, f1_name, key_index]  # f1-> first input f2-> second input
            myDict[g1_name] = [f1_name, f2_name, key_index]
            selected_g.append(g1_name)
            # selected_g.append(g2_name)
        end_time = time.time()
        print("time for part1:", end_time - start_time)
        K_list = [int(i) for i in K_list]
        # print(myDict)
        # ###################################################################
        # # continue to construct the locked circuits based on the same idea
        # # like locking function
        # ###################################################################
        count = 0
        detected = 0
        start_time = time.time()
        # for line in bench_as_a_string.split("\n"):
        #     count += 1
        #     line = line.strip()
        #     if any(ext + " =" in line for ext in selected_g):  # id -> gate_name -> search
        #         detected = detected + 1
        #         regex = "(\S+)\s*=\s*(NOT|BUF|OR|XOR|AND|NAND|XNOR|NOR)\((.+?)\)\s*"
        #         for output, function, net_str in re.findall(regex, line, flags=re.I | re.DOTALL):
        #             if output in myDict.keys():
        #                 my_f1 = myDict[output][0]
        #                 my_f2 = myDict[output][1]
        #                 my_key = myDict[output][2]
        #                 line = line.replace(my_f1 + ",", output + "_from_mux,")
        #                 line = line.replace(my_f1 + ")", output + "_from_mux)")
        #                 locked_file = locked_file + (line + "\n")
        #                 if K_list[my_key] == 0:
        #                     locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
        #                         my_key) + ", " + my_f1 + ", " + my_f2 + ")\n"
        #                 else:

        #                     locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
        #                         my_key) + ", " + my_f2 + ", " + my_f1 + ")\n"
        #             else:
        #                 locked_file = locked_file + line + "\n"
        #     else:
        #         locked_file = locked_file + line + "\n"
        locked_file_rest = self.parallel_process(bench_as_a_string, selected_g, myDict, K_list)
        locked_file = locked_file + locked_file_rest
        end_time = time.time()
        print("time for part2:", end_time - start_time)
        # text_file = open(self.model_path + "check_locked_" + "test1.txt", "w")
        # text_file.write(locked_file)
        # text_file.close()

        return locked_file

    def decode_scope(self, bench_as_a_string, kgss):
        # get the graph from the original benchmark
        G = verilog2gates(bench_as_a_string)
        # print(G)
        encoding_list = kgss.data
        key_len = len(encoding_list)
        K_list = []  # store the key value
        for index in range(key_len):
            K_list.append(encoding_list[index][4])
        ###################################################################
        # generate the key string and key input in the beginning of the file
        ###################################################################
        locked_file = ""
        i = 0
        locked_file = locked_file + "#key="
        while i < key_len:
            locked_file = locked_file + str(int(K_list[i]))
            i = i + 1
        locked_file = locked_file + ("\n")
        i = 0
        while i < key_len:
            locked_file = locked_file + "INPUT(keyinput" + str(i) + ")\n"
            i = i + 1
        ###################################################################
        # get the information from the kgss object about the f and g
        # reconstruct myDict, selected_g
        ###################################################################
        myDict = {}
        selected_g = []
        # format:[('Gxx1', index1), ('Gxx2', index2)]
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}

        for item in encoding_list:
            f1_name = G_info_updated[int(item[0])]
            f2_name = G_info_updated[int(item[1])]
            g1_name = G_info_updated[int(item[2])]
            # g2_name = G_info_updated[int(item[3])]
            key_index = int(item[5])
            # myDict[g2_name] = [f2_name, f1_name, key_index]  # f1-> first input f2-> second input
            myDict[g1_name] = [f1_name, f2_name, key_index]
            selected_g.append(g1_name)
            # selected_g.append(g2_name)
        K_list = [int(i) for i in K_list]
        # print(myDict)
        # ###################################################################
        # # continue to construct the locked circuits based on the same idea
        # # like locking function
        # ###################################################################
        count = 0
        detected = 0
        for line in bench_as_a_string.split("\n"):
            count += 1
            line = line.strip()
            if any(ext + " =" in line for ext in selected_g):  # id -> gate_name -> search
                detected = detected + 1
                regex = "(\S+)\s*=\s*(NOT|BUF|OR|XOR|AND|NAND|XNOR|NOR)\((.+?)\)\s*"
                for output, function, net_str in re.findall(regex, line, flags=re.I | re.DOTALL):
                    if output in myDict.keys():
                        my_f1 = myDict[output][0]
                        my_f2 = myDict[output][1]
                        my_key = myDict[output][2]
                        line = line.replace(my_f1 + ",", output + "_from_mux,")
                        line = line.replace(my_f1 + ")", output + "_from_mux)")
                        locked_file = locked_file + (line + "\n")
                        if K_list[my_key] == 0:
                            locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
                                my_key) + ", " + my_f1 + ", " + my_f2 + ")\n"
                        else:

                            locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
                                my_key) + ", " + my_f2 + ", " + my_f1 + ")\n"
                    else:
                        locked_file = locked_file + line + "\n"
            else:
                locked_file = locked_file + line + "\n"

        # text_file = open(self.model_path + "check_locked_" + "test1.txt", "w")
        # text_file.write(locked_file)
        # text_file.close()

        return locked_file


    def decode_new(self, bench_as_a_string, kgss):
        # get the graph from the original benchmark
        G = verilog2gates(bench_as_a_string)
        # print(G)
        encoding_list = kgss.data
        key_len = len(encoding_list)
        K_list = []  # store the key value
        for index in range(key_len):
            K_list.append(encoding_list[index][4])
        ###################################################################
        # generate the key string and key input in the beginning of the file
        ###################################################################
        locked_file = ""
        i = 0
        locked_file = locked_file + "#key="
        while i < key_len:
            locked_file = locked_file + str(int(K_list[i]))
            i = i + 1
        locked_file = locked_file + ("\n")
        i = 0
        while i < key_len:
            locked_file = locked_file + "INPUT(keyinput" + str(i) + ")\n"
            i = i + 1
        ###################################################################
        # get the information from the kgss object about the f and g
        # reconstruct myDict, selected_g
        ###################################################################
        myDict = {}
        selected_g = []
        # format:[('Gxx1', index1), ('Gxx2', index2)]
        G_info = dict(list(G.nodes(data="count")))
        G_info_updated = {y: x for x, y in G_info.items()}
        all_fs = []
        all_gs = []
        # define target fs dictionary
        target_fs ={}
        for item in encoding_list:
            f1_name = G_info_updated[int(item[0])]
            f2_name = G_info_updated[int(item[1])]
            g1_name = G_info_updated[int(item[2])]
            g2_name = G_info_updated[int(item[3])]
            key_index = int(item[5])
            myDict[g2_name] = [f2_name, f1_name, key_index]  # f1-> first input f2-> second input
            myDict[g1_name] = [f1_name, f2_name, key_index]
            selected_g.append(g1_name)
            selected_g.append(g2_name)
            # merge all fs and gs
            all_fs.append(f1_name)
            all_fs.append(f2_name)
            all_gs.append(g1_name)
            all_gs.append(g2_name)
        # find the fs which are also occur in gs
        for g_temp in all_gs:
            if g_temp in all_fs:
                target_fs[g_temp] = 0
        # modify myDict
        for i in range(0, len(all_gs), 2):
            [f2_temp, f1_temp, key_index] = myDict[all_gs[i]]
            if f2_temp in target_fs.keys():
                f2_temp = f2_temp + "_from_mux"
                target_fs[f2_temp] = target_fs[f2_temp] + 1
            if f1_temp in target_fs.keys():
                f1_temp = f1_temp + "_from_mux"
                target_fs[f1_temp] = target_fs[f1_temp] + 1
            myDict[all_gs[i]]= [f2_temp, f1_temp, key_index]
            myDict[all_gs[i+1]]= [f1_temp, f2_temp, key_index]



        K_list = [int(i) for i in K_list]
        # print(myDict)
        # ###################################################################
        # # continue to construct the locked circuits based on the same idea
        # # like locking function
        # ###################################################################
        count = 0
        detected = 0
        for line in bench_as_a_string.split("\n"):
            count += 1
            line = line.strip()
            if any(ext + " =" in line for ext in selected_g):  # id -> gate_name -> search
                detected = detected + 1
                regex = "(\S+)\s*=\s*(NOT|BUF|OR|XOR|AND|NAND|XNOR|NOR)\((.+?)\)\s*"
                for output, function, net_str in re.findall(regex, line, flags=re.I | re.DOTALL):
                    if output in myDict.keys():
                        my_f1 = myDict[output][0]
                        my_f2 = myDict[output][1]
                        my_key = myDict[output][2]
                        line = line.replace(my_f1 + ",", output + "_from_mux,")
                        line = line.replace(my_f1 + ")", output + "_from_mux)")
                        locked_file = locked_file + (line + "\n")
                        if K_list[my_key] == 0:
                            locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
                                my_key) + ", " + my_f1 + ", " + my_f2 + ")\n"
                        else:

                            locked_file = locked_file + output + "_from_mux = MUX(keyinput" + str(
                                my_key) + ", " + my_f2 + ", " + my_f1 + ")\n"
                    else:
                        locked_file = locked_file + line + "\n"
            else:
                locked_file = locked_file + line + "\n"

        text_file = open(self.model_path + "check_locked_" + "test1.txt", "w")
        text_file.write(locked_file)
        text_file.close()

        return locked_file

    def get_max_id(self, bench_as_a_string):
        # get the graph from the original benchmark
        G = verilog2gates(bench_as_a_string)
        G_info = dict(list(G.nodes(data="count")))
        G_id = list(G_info.values())
        G_id_remove_None = []
        for item in G_id:
            if item != None:
                G_id_remove_None.append(item)
        G_id_int = [int(i) for i in G_id_remove_None]
        return max(G_id_int)

