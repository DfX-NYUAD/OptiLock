from ec.impl.muxlink_fitness_function_plus import MuxLinkFitnessFunctionPlus
from ec.impl.muxlink_fitness_function_large import MuxLinkFitnessFunctionLarge
from muxlink.muxlink import MuxLink
from muxlink.original import *
from utils.bench_parser import BenchParser
# from ec.impl.watch_dog import KPAFileHandler
from watchdog.observers import Observer

import random
from sa.dual_anneal_early_stop_quick import *
import math
import csv
import sys
import argparse
import shutil
import time
import subprocess
import threading
import copy



# this function is to read the file simplier
def file_to_string(target_file):
    target_str = ''
    with open(target_file, 'r') as f:
        target_str = f.read()
    return target_str

# set the global csv file to store the SA results
def setGlobalFile(csvFilePath):
    global GLOBAL_CSVFILE_WRITER
    csvFile = open(csvFilePath,'w+')
    fileHeader = ['tried_score','cur_score','best_score','temp','a_prob','a_flag']
    GLOBAL_CSVFILE_WRITER = csv.DictWriter(csvFile, fieldnames=fileHeader)
    GLOBAL_CSVFILE_WRITER.writeheader()

# encode the netlist and return the original kgss data list
def netlist_encode(netlist_str, circuit_name):
    muxlink = MuxLink(circuit_name)
    kgss = muxlink.encode(netlist_str)
    # print(dict(kgss.graph.nodes()))
    return kgss

# lock the chosen kgss with the solution vector, key value
# return the locked kgss data
def kgss_lock(kgss, sol_vector, key_value, edge_dict, founded_pairs_im):
    # edge_dict = kgss.get_all_possible_edges()

    kgss_data = []
    for i in range(0, len(sol_vector), 2):
        pair1 = int(math.floor(sol_vector[i]))
        # pair1 = int(math.ceil(sol_vector[i]/2.)*2)
        pair2 = int(math.floor(sol_vector[i + 1]))

        pair_f1_g1 = edge_dict[pair1]
        pair_f2_g2 = edge_dict[pair2]
        key_temp = key_value[int(i / 2)]  # random.randint(0, 1)
        key_value += str(key_temp)

        kgss_temp = [str(pair_f1_g1[0]), str(pair_f2_g2[0]), str(pair_f1_g1[1]), str(pair_f2_g2[1]), str(key_temp),
                     int(i / 2)]

        kgss_data.append(kgss_temp)
    # get the length of current kgss data
    kgss_data_length = len(kgss_data)
    # append the foundpairs with the following index 
    founded_pairs_temp = founded_pairs_im
    print("unchanged founded pairs:", founded_pairs_temp)
    for i in range(kgss_data_length, kgss_data_length+len(founded_pairs_im)):
        founded_pairs_temp[i-kgss_data_length][5] = i # update the index
        founded_pairs_temp[i-kgss_data_length][4] = key_value[i] # update the key value
        kgss_data.append(founded_pairs_temp[i-kgss_data_length])
    print("updated founded pairs:", founded_pairs_temp)
    kgss.data = kgss_data
    return kgss

# convert kgss data to solution vector
def kgss_to_sol(kgss, edge_dict):
    sol_vector = []
    for i in range(len(kgss.data)):
        pair = kgss.data[i]
        pair_f1_g1 = [int(pair[0]), int(pair[2])]
        pair_f2_g2 = [int(pair[1]), int(pair[3])]
        for key, value in edge_dict.items():
            if value == pair_f1_g1:
                sol_vector.append(key)
            if value == pair_f2_g2:
                sol_vector.append(key)
    return sol_vector

# convert kgss data to solution vector
def kgss_to_sol_all(found_pairs, edge_dict):
    sol_vector = []
    for i in range(len(found_pairs)):
        pair = found_pairs[i]
        pair_f1_g1 = [int(pair[0]), int(pair[2])]
        pair_f2_g2 = [int(pair[1]), int(pair[3])]
        for key, value in edge_dict.items():
            if value == pair_f1_g1:
                sol_vector.append(key)
            # else:
            #     print("pair_f1_g1:", pair_f1_g1)
            if value == pair_f2_g2:
                sol_vector.append(key)
            # else:
            #     print("pair_f2_g2:", pair_f2_g2)
    return sol_vector

# convert found kgss to solution vector
# update the edge dict if the found pairs are not empty
def kgss_to_sol_found(found_pairs, edge_dict):
    sol_vector = []
    for i in range(len(found_pairs)):
        pair = found_pairs[i]
        pair_f1_g1 = [int(pair[0]), int(pair[2])]
        pair_f2_g2 = [int(pair[1]), int(pair[3])]
        for key, value in edge_dict.items():
            if value == pair_f1_g1:
                sol_vector.append(key)
            if value == pair_f2_g2:
                sol_vector.append(key)
    # update the edge dict
    # remove the found pairs from the edge dict
    edge_dict_temp = {}
    for key, value in edge_dict.items():
        if key not in sol_vector:
            edge_dict_temp[key] = value
    # indexing it using the new key starting from 0 to the length
    index = 0
    edge_dict_new = {}
    for key, value in edge_dict_temp.items():
        edge_dict_new[index] = value
        index += 1
    return edge_dict_new

# this function is used to update the founded pairs
def update_pairs(founded_pairs, found_temp):
    # get all the pairs from the found_temp
    founded_list = []
    for item in founded_pairs:
        pair_temp = item[0:4]
        founded_list.append(pair_temp)

    for item in found_temp:
        pair_temp1 = item[0:4]
        pair_temp2 = [item[1], item[0], item[3], item[2]]
        if pair_temp1 not in founded_list and pair_temp2 not in founded_list:
            founded_pairs.append(item)
    return founded_pairs

# this function is used to update the best kgss 
def update_best_kgss(best_kgss_temp, found_pairs, key_value):
    # get the length of current kgss data
    kgss_data_length = len(best_kgss_temp.data)
    # append the foundpairs with the following index 
    for i in range(kgss_data_length, kgss_data_length+len(found_pairs)):
        found_pairs[i-kgss_data_length][5] = i # update the index
        found_pairs[i-kgss_data_length][4] = key_value[i] # update the key value
        best_kgss_temp.data.append(found_pairs[i-kgss_data_length])
    return best_kgss_temp

def split_dictionary(data, num_parts):
    # Calculate the number of items each part should have
    total_items = len(data)
    items_per_part = total_items // num_parts
    remainder = total_items % num_parts
    
    # Create a list to store the smaller dictionaries
    split_dicts = [{} for _ in range(num_parts)]
    
    # Initialize an index for tracking which part to add to
    part_index = 0
    item_count = 0
    
    # Iterate over the original dictionary and distribute the items
    for idx, (key, value) in enumerate(data.items()):
        # Add the key-value pair to the appropriate smaller dictionary with reset key
        split_dicts[part_index][item_count] = value
        item_count += 1
        
        # Move to the next dictionary if the current one is full
        if item_count == items_per_part + (1 if part_index < remainder else 0):
            part_index += 1
            item_count = 0
    
    return split_dicts

def split_dictionary_v2(dict_list, key_size):
    split_dicts = []
    for dict_temp in dict_list:
        if len(dict_temp) > 3000:
            split_dict = split_dictionary(dict_temp, len(dict_temp)//(key_size*5))
            split_dicts += split_dict
        else:
            split_dicts.append(dict_temp)
    split_dicts_final = []
    for item in split_dicts:
        seen_values = []
        clean_dict = {}
        for sublist in item.values():
            if sublist not in seen_values:
                seen_values.append(sublist)
        # relist them as a dictionary
        for i, sublist in enumerate(seen_values):
            clean_dict[i] = sublist
        # return clean_dict
        split_dicts_final.append(clean_dict)
    return split_dicts_final

# set up some parameter for SA function
def sa_setup(target_path, key_size, h_hop, chosen_sol, train_mark):
    # set key value
    key_value = ''.join(str(random.randint(0, 1)) for _ in range(key_size))
    print(key_value)
    # Load the benchmark
    netlist_str = file_to_string(target_path)
    fitness_function = None
    # create original kgss object
    circuit_name = target_path.split("/")[-1].split(".bench")[0].split("_")[0]
    kgss_ori = netlist_encode(netlist_str, circuit_name)
    edge_dict_list = kgss_ori.get_fan_cone_edges_all_large()
    edge_dict_list_part = edge_dict_list[0:20] # take the first fan in cone [0:20]

    edge_dict_list = split_dictionary_v2(edge_dict_list_part, key_size)

    # print the edge_dict_list
    for edge_dict in edge_dict_list:
        print(len(edge_dict), flush=True)

    return kgss_ori, fitness_function, key_value, edge_dict_list, circuit_name

# random select valid edge pair for the further attack
def random_select(kgss, edge_dict, key_value):
    max_edge_num = len(kgss.get_all_possible_edges())
    key_size = len(key_value)
    random_selected_sol = random.sample(list(range(max_edge_num-1)), key_size)
    kgss_temp = kgss_lock(kgss, random_selected_sol, key_value, edge_dict)
    # make it to be valid
    if kgss_temp.check_same_gs():
        kgss_data = kgss_temp.modify_same_gs_selected(edge_dict)
        kgss_temp.data = kgss_data
    return kgss_temp
    
# this function is generate the command to run sa_muxlink_parallel.py
def command_muxlink_parallel(target_path, h_hop, train_mark, bin_num):
    ROOT_DIR="/scratch/zw3464/ec-ll-SA-master/src"
    # in order to unify the format,
    # we added the kgss_data once we use it in each fun_eval function
    command_muxlink = "python " + ROOT_DIR + "/sa_muxlink_parallel.py --target-path " + target_path + " --h-hop " + str(h_hop) + " --train-mark " + str(train_mark) + " --bin-num " + str(bin_num) + " --kgss-data " 
    return command_muxlink

# this function is generate the command to run sa_fitness_multi_parallel.py
def command_multi_parallel(target_path, h_hop, train_mark, bin_num, start_num, total_num):
    ROOT_DIR="/scratch/zw3464/ec-ll-SA-master/src"
    # update target_path 
    command_multi = "python " + ROOT_DIR + "/sa_fitness_multi_parallel.py --target-path " + target_path + " --h-hop " + str(h_hop) + " --train-mark " + str(train_mark) + " --bin-num " + str(bin_num)  + " --start-num " + str(start_num) + " --total-num " + str(total_num) + " --kgss-data " #+ "\"" + str(kgss_data) + "\""
    return command_multi

def get_slurm_job_id(job_name, user=None):
    # Construct the squeue command
    cmd = ['squeue']
    if user:
        cmd.extend(['-u', user])
    
    # Execute squeue and filter results with grep
    result = subprocess.run(cmd, capture_output=True, text=True)
    matching_lines = [line for line in result.stdout.splitlines() if job_name in line]
    
    # Extract job IDs from matching lines
    job_ids = [line.split()[0] for line in matching_lines]

    return job_ids

# this function is to run the multi_muxlink 
def multi_muxlink(target_path, h_hop, train_mark, bin_num, start_num, total_num, kgss_data):
    # create the sbatch script folder to run the command 
    circuit_name = target_path.split('/')[-1].split('.')[0].split('_')[0]
    ROOT_DIR="/scratch/zw3464/ec-ll-SA-master/src"
    # save the command list to the first folder 
    command_path = "../ml_data_" + circuit_name + str(start_num) + "/command_multi.txt"
    with open(command_path, 'w') as command_file:
        for i in range(start_num, start_num+total_num):
            target_path = "../data/original/" + circuit_name + "_ori/"+ circuit_name + str(i) + ".bench"
            if "b" in circuit_name:
                target_path = "../data/original/" + circuit_name + "_ori/" + circuit_name + str(i) + "_C.bench"
            command = "python " + ROOT_DIR + "/sa_fitness_multi.py --target-path " + target_path + " --h-hop " + str(h_hop) + " --kgss-data " + "\"" + str(kgss_data) + "\"" + " --train-mark " + str(train_mark) + " --bin-num " + str(bin_num)
            command_file.write(command + "\n")
    run_command = "slurm_parallel_ja_submit.sh -t 00:20:00 -q small -j " + circuit_name + str(start_num) + "mux"  + " " + command_path
    # os.system(run_command)
    run_command = run_command.split()
    output = subprocess.run(run_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    job_id_current_list = get_slurm_job_id(circuit_name + str(start_num) + "mux")
    if "error" in output: # if there is error text in the output, we need to cancel the job directly 
        if len(job_id_current_list) > 0: # if the job has been runned wrongly, cancel it
            job_id_current = job_id_current_list[0].split("_")[0]
            # cancel the job
            os.system("scancel " + job_id_current)
            print("I am canceling this job:", job_id_current)
    # keep running it until there is no error in the output
    while "error" in output:
        time.sleep(100)
        job_ids_all = []
        for i in range(10):
            job_ids = get_slurm_job_id(circuit_name + str(start_num) + "mux")
            job_ids_all += job_ids
            time.sleep(10)
            # if the job_id is not empty, jump out of the loop
            if len(job_ids_all) > 0:
                # remove the [] form the job_ids
                # job_ids = job_ids[0].split("_")[0]
                break
        # until now, we maximum wait for 200+100 = 300 seconds
        print("job ids: ", job_ids_all, flush=True)
        if len(job_ids_all) > 0:
            print("the job is recovered successfully, but it is not stable")
            # jump out of the while loop with return True
            # cancel the job
            job_id = job_ids_all[0].split("_")[0]
            print("I am canceling this job:", job_id)
            os.system("scancel " + job_id)
            
        print("anytime, we need to rerun it to secure the result")
        run_command_run =  "slurm_parallel_ja_submit.sh -t 00:20:00 -q small -j " + circuit_name[:-2] + str(start_num) + "mux"  + " " + command_path
        run_command_run = run_command_run.split()
        output = subprocess.run(run_command_run, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
            
    print("do i have any output? ", output, flush=True)
    if "error" in output:
        return False
    else:
        return True

# this function is used to run the fitness function in serial
def attack_muxlink_serial_load(circuit_name, start_num, total_num, h_hop, train_mark, kgss_sol):
    # fitness_function_list = []
    kgss_data_index = 0 # reserved value for furture usage 
    # collect the all the wrong_list and uni_list
    wrong_list_all = []
    uni_list_all = []
    # based on the start num and total number to get the target path
    for i in range(start_num, start_num+total_num):
        # target_path_temp = "../data/original/" + circuit_name + "_ori/" + circuit_name + str(i) + ".bench"
        target_path_temp = "../data/original/" + circuit_name + "_ori/" + circuit_name + str(i) + ".bench"
        if "b" in circuit_name:
            target_path_temp = "../data/original/" + circuit_name + "_ori/" + circuit_name + str(i) + "_C.bench"
        fitness_function = MuxLinkFitnessFunctionLarge(target_path_temp, target_path_temp, h_hop, kgss_data_index, train_mark, epochs=100)
        # fitness_function_list.append(fitness_function)
    # return fitness_function_list
    
    # for fitness_function in fitness_function_list:
        try:
            result_list = fitness_function.evaluate_exact([kgss_sol])
        except Exception as e:
            print("error in the sa_fitness_multi_parallel.py")
            result_list = [1.0, [], []] 
        uni_list = result_list[1]
        wrong_list = result_list[2]
        uni_list_all += uni_list
        wrong_list_all += wrong_list
    # debug here
    print("uni_list_all:", uni_list_all)
    print("wrong_list_all:", wrong_list_all)
    # get the most occured element in the wrong_list_all and uni_list_all
    uni_dict = {}
    wrong_dict = {}
    for num in uni_list_all:
        if num in uni_dict:
            uni_dict[num] += 1
        else:
            uni_dict[num] = 1
    for num in wrong_list_all:
        if num in wrong_dict:
            wrong_dict[num] += 1
        else:
            wrong_dict[num] = 1
    # get the largest frequency of the uni_list and wrong_list
    # check if uni_dict.values() is empty
    if len(uni_dict.values()) == 0:
        max_uni = 0
        uni_list_final = []
    else:
        max_uni = max(uni_dict.values())
        uni_list_final = [num for num, freq in uni_dict.items() if freq == max_uni or freq == max_uni-1 or freq == max_uni-2]
    # check if wrong_dict.values() is empty
    if len(wrong_dict.values()) == 0:
        max_wrong = 0
        wrong_list_final = []
    else:
        max_wrong = max(wrong_dict.values())
        wrong_list_final =  [num for num, freq in wrong_dict.items() if freq == max_wrong or freq == max_wrong-1 or freq == max_wrong-2]
    # combine all the number in the uni_list_final and wrong_list_final, and sort them by values
    final_loc = list(set(uni_list_final + wrong_list_final))
    print("debug", final_loc)
    # print("debug here:", len(final_loc))
    # based on the final_loc, return the selected kgss.data based on the final_loc
    final_kgss_data = []
    for loc in final_loc:
        final_kgss_data.append(kgss_sol.data[loc])
    # # resort the index of final_kgss_data and give the new index from 0 to the length
    # final_kgss_data = sorted(final_kgss_data, key=lambda x: x[5])
    return final_kgss_data


# SA evaluation function
def fun_eval(x, kgss_ori, fitness_function, key_value, founded_pairs, edge_dict, thread_num, command_multi, circuit_name, bin_num, start_num, total_num, h_hop):
    x_temp = list(x)

    if len(set(x_temp)) != len(x_temp):
        print("same node?")
        return 1.2  # try to avoid the same node
    kgss = kgss_lock(kgss_ori, x, key_value, edge_dict, founded_pairs)
    # if the kgss is not valid, return a large value
    if kgss.check_same_gs():
        print("same gs?")
        # print(kgss.data)
        if not kgss.change_same_gs():
           return 1.2 # try to avoid wrong solution
        # otherwise, continue to check
    if kgss.check_f_in_g():
        print("f in g?")
        return 1.2
    print("kgss data")
    print(kgss.data)
    # kpa_list = fitness_function.evaluate([kgss])

    for i in range(total_num):
        location = "../ml_data_" + circuit_name + str(i+start_num)
        kpa_log_path = location + "/kpa.txt"
        if os.path.exists(kpa_log_path):
            print("remove the kpa log file")
            os.remove(kpa_log_path)
        all_files = os.listdir(location)
        # if there error and output file, remove them
        all_error_log = [f for f in all_files if "err" in f]
        for f in all_error_log:
            os.remove(os.path.join(location, f))
        all_output_log = [f for f in all_files if "out" in f]
        for f in all_output_log:
            os.remove(os.path.join(location, f))

    run_value_multi = False
    while not run_value_multi:
        run_value_multi = multi_muxlink(target_path, h_hop, train_mark, bin_num, start_num, total_num, kgss.data)

    all_kpa_log_list = []
    max_sleep = 10  # Maximum sleep time in seconds
    sleep_time = 1
    for i in range(start_num, start_num+total_num):
        location = "../ml_data_" + circuit_name + str(i)
        all_files = os.listdir(location)
        all_kpa_log = [f for f in all_files if "kpa.txt" in f]
        if len(all_kpa_log) == 0:
            continue
        else:
            all_kpa_log_list.append(all_kpa_log[0])
    while len(all_kpa_log_list) < total_num:
        all_kpa_log_list = []
        for i in range(start_num, start_num+total_num):
            location = "../ml_data_" + circuit_name + str(i)
            all_files = os.listdir(location)
            all_kpa_log = [f for f in all_files if "kpa.txt" in f]
            if len(all_kpa_log) == 0:
                continue
            else:
                all_kpa_log_list.append(all_kpa_log[0])
        # time.sleep(2)
        time.sleep(sleep_time)
        sleep_time = min(sleep_time * 2, max_sleep)
    

    print("all kpa log list", all_kpa_log_list)
    # read all the kpa log file and get the kpa value
    kpa_list = []
    acc_list = []
    prec_list = []
    for i in range(total_num):
        location = "../ml_data_" + circuit_name + str(i+start_num)
        kpa_log_path = location + "/kpa.txt"
        with open(kpa_log_path, 'r') as kpa_log:
            # read the line of the kpa_log, it contains acc, prec and kpa
            kpa_str = kpa_log.read()
            acc, prec, kpa = [float(i) for i in kpa_str.split(",")]
            # convert to float
            # kpa = float(kpa_log.read())
            kpa_list.append(kpa)
            acc_list.append(acc)
            prec_list.append(prec)
    # average the kpa value
    kpa = sum(kpa_list)/len(kpa_list)
    acc = sum(acc_list)/len(acc_list)
    prec = sum(prec_list)/len(prec_list)
    x_unpred = (kpa-acc)/kpa # here is x_unpred
    # print("kpa:", kpa)
    # print acc, prec and kpa
    print("acc:", acc, "prec:", prec, "kpa:", kpa, "x_unpred", x_unpred)
    # remove the kpa log file
    for i in range(total_num):
        location = "../ml_data_" + circuit_name + str(i+start_num)
        kpa_log_path = location + "/kpa.txt"
        os.remove(kpa_log_path)
    # remove the error and output file
    
    return float(kpa)

# SA callback function
def fun_callback(data_dict):
    GLOBAL_CSVFILE_WRITER.writerow({'tried_score': data_dict['tried_f'], 'cur_score': data_dict['cur_f'], \
                                        'best_score': data_dict['best_f'], 'temp': data_dict['temp'], \
                                        'a_prob': data_dict['a_prob'], 'a_flag': data_dict['a_flag']})
    ## GLOBAL_CSVFILE_WRITER.flush()
    print("\nScores- tried: {:.4f}, current: {:4f}, best: {:4f}".format(data_dict['tried_f'], data_dict['cur_f'],
                                                                        data_dict['best_f']))

    # print the current solution and tried solution
    print("Tried solution: ", data_dict['tried_x'])
    # print("Current solution: ", data_dict['cur_x'])
# SA main algorithm
def sa_main(key_size, result_path, target_path, iteration_num, h_hop, int_temp, chosen_sol, bin_num, train_mark, start_num, total_num):
    setGlobalFile(result_path)
    # set up the parameters
    kgss_ori, fitness_function, key_value, edge_dict, circuit_name = sa_setup(target_path, key_size, h_hop, chosen_sol, train_mark)
    # set up the command for parallel running for muxlink
    command_muxlink = command_muxlink_parallel(target_path, h_hop, train_mark, bin_num)
    command_multi = command_multi_parallel(target_path, h_hop, train_mark, bin_num, start_num, total_num)
    # set the thread num -default =1
    # print("edge_dict", edge_dict)
    thread_num = 0
    # get the maximum number of edges
    # max_edge_num = len(kgss_ori.get_all_possible_edges())
    max_edge_num = len(kgss_ori.get_fan_cone_edges())
    print("edge_dict", edge_dict)
    # max_edge_num = len(edge_dict) # design for the random selection
    # set the bounds
    lw = [0] * key_size *2
    up = [max_edge_num-1] * key_size *2
    print("max edge num: ", max_edge_num)
    # create the founded solution
    founded_pairs = []
    ret = dual_annealing(fun_eval, args=(kgss_ori, fitness_function, key_value, founded_pairs, edge_dict, thread_num, command_multi, circuit_name, bin_num, start_num, total_num, h_hop), bounds=list(zip(lw, up)), maxfun=iteration_num, initial_temp=int_temp,
                         visit=2.0, accept=-5000, no_local_search=True, 
                         callback=fun_callback)  # , x0=initial_vector)#, seed=random_seed)#counter*num_runs+seed_counter)
    best_solution = list(ret.x)
    best_fitness = float(ret.fun)
    print("best solution: ", best_solution)
    print("best fitness: ", best_fitness)


def sa_main_vertical_temp(key_size, result_path, target_path, iteration_num, h_hop, int_temp, chosen_sol, bin_num, train_mark, start_num, total_num):
    setGlobalFile(result_path)
    # set up the command for parallel running for muxlink
    command_multi = command_multi_parallel(target_path, h_hop, train_mark, bin_num, start_num, total_num)
    # set up the parameters for each time
    kgss_ori, fitness_function, key_value, edge_dict_list, circuit_name = sa_setup(target_path, key_size, h_hop, chosen_sol, train_mark)
    # set the thread num -default =0
    thread_num = 0
    # set the final_kgss_data
    final_kgss_data = []
    # get the whole edge dict
    # edge_dict_whole = kgss_ori.get_all_possible_edges()


# here is the sa main - vertical version
# it mianly includes these steps: 
# 1) based on the certain number of edges we defined, we will keep the logic cones
#    for the small cones, we will sum them up to get the satisified number of edges
#    for the large cones, we will select the edges under the certain number
#    we need to make sure the edges are not repeated selected, and also the logic cone is not overlapping
# 2) we will first run the SA algorithm for key size iterations; after that, we will keep the best locality;
#    we will reduce the key size by the number of found key size, and then run the SA algorithm again
# 3) we keep running this algorithm until we find the best solution, might be try 5 iterations.
def sa_main_vertical(key_size, result_path, target_path, iteration_num, h_hop, int_temp, chosen_sol, bin_num, train_mark, start_num, total_num, output_file):
    setGlobalFile(result_path)
    # set up the command for parallel running for muxlink
    command_multi = command_multi_parallel(target_path, h_hop, train_mark, bin_num, start_num, total_num)
    # set up the parameters for each time
    kgss_ori, fitness_function, key_value, edge_dict_list, circuit_name = sa_setup(target_path, key_size, h_hop, chosen_sol, train_mark)
    # set the thread num -default =0
    thread_num = 0
    # set the final_kgss_data
    final_kgss_data = []
    # get the whole edge dict
    edge_dict_whole = kgss_ori.get_all_possible_edges_large()
    # print(len(edge_dict_whole))
    # create the founded solution
    founded_pairs = []
    final_best_solution = []
    best_fitness_final = 1.0
    remove_signal = 0
    for iter in range(len(edge_dict_list)):
        edge_dict_temp = edge_dict_list[iter]
        # print("edge_dict", edge_dict)
        edge_dict = kgss_to_sol_found(founded_pairs, edge_dict_temp)
        max_edge_num = len(edge_dict)
        founded_key_size = len(founded_pairs)
        # keep trace of founded pairs
        founded_pairs_keep = copy.deepcopy(founded_pairs)
        # check if the solution size is large than edge dict
        # then we will break it
        if (key_size - founded_key_size)*2 > max_edge_num:
            print("Key Size - current search space is not valid")
            break
        
        if founded_key_size == key_size or remove_signal == 1:
            # remove the unconfused localities based on the final_kgss_temp
            kgss = kgss_lock(kgss_ori, [], key_value, edge_dict_whole, founded_pairs_keep)
            final_kgss_temp1 = attack_muxlink_serial_load(circuit_name, start_num, total_num, h_hop, train_mark, kgss)
            founded_pairs = update_pairs([], final_kgss_temp1)
            founded_key_size = len(founded_pairs)
            # update the edge_dict based on the founded pairs
            edge_dict = kgss_to_sol_found(founded_pairs, edge_dict_temp)
            max_edge_num = len(edge_dict)
            remove_signal = 0
        

        founded_pairs_keep = copy.deepcopy(founded_pairs) 
        print("founded pairs keep: ", founded_pairs_keep)
        print("founded key size: ", founded_key_size)   
        # set the bounds
        lw = [0] * (key_size - founded_key_size) *2
        up = [max_edge_num-1] * (key_size - founded_key_size) *2
        print("max edge num: ", max_edge_num)
        
    
        ret = dual_annealing(fun_eval, args=(kgss_ori, fitness_function, key_value, founded_pairs, edge_dict, thread_num, command_multi, circuit_name, bin_num, start_num, total_num, h_hop), bounds=list(zip(lw, up)), maxfun=iteration_num, initial_temp=int_temp,
                         visit=2.0, accept=-5000, no_local_search=True, 
                         callback=fun_callback)  # , x0=initial_vector)#, seed=random_seed)#counter*num_runs+seed_counter)
        # print("what is my founded pair2:", founded_pairs)
        if not ret.success:
            print("SA - current search space is not valid")
            continue
        best_solution_temp = list(ret.x) 

        kgss_best_temp = kgss_lock(kgss_ori, best_solution_temp, key_value, edge_dict, [])
        # find the solution vector based on the whole edge dict
        best_solution_temp_whole = kgss_to_sol(kgss_best_temp, edge_dict_whole)
        # print("what is the founded pair2:", founded_pairs_keep)
        found_solution_whole = kgss_to_sol_all(founded_pairs_keep, edge_dict_whole)
        best_solution = best_solution_temp_whole + found_solution_whole

        # get the best solution temp under the whole edge dict
        best_solution_temp = best_solution_temp_whole
        
        
        # get the kgss data from the best solution
        kgss = kgss_lock(kgss_ori, best_solution_temp, key_value, edge_dict_whole, founded_pairs_keep)
        # print("what is the founded pair4:", founded_pairs_keep)
        # evaluate the kgss and get the wrong and uni classified locality
        final_kgss_temp = attack_muxlink_serial_load(circuit_name, start_num, total_num, h_hop, train_mark, kgss)

        founded_pairs = update_pairs(founded_pairs_keep, final_kgss_temp)
        # founded_pairs = update_pairs([], final_kgss_temp)
        # print("what is the founded pair5:", founded_pairs_keep)
        best_fitness = float(ret.fun)
        
        # if the rest of key pairs are guessed wrongly, 
        # with the founded pairs, we can not reach to the 0.5, and then we remove the unconfused part
        rest_key_size = key_size - founded_key_size
        wrong_key_size = int(best_fitness * key_size)
        if rest_key_size + wrong_key_size < key_size*0.5:
            remove_signal = 1
        # print several information to indicate the running 
        print("####################################")
        print("this is the ", iter, "th iteration")
        print("best solution: ", best_solution)
        print("best fitness: ", best_fitness)
        print("founded pairs: ", founded_pairs)
        # print("edge dict: ", edge_dict)
        # update the best solution and best fitness
        if best_fitness < best_fitness_final:
            best_fitness_final = best_fitness
            final_best_solution = best_solution
        if best_fitness < 0.5:
            break
     
    print("final best solution: ", final_best_solution)
    print("final best fitness: ", best_fitness_final)
    
    #based on the final best solution, we can decode the netlist
    kgss_best = kgss_lock(kgss_ori, final_best_solution, key_value, edge_dict_whole, [])
    netlist_ori = netlist_str_spe(target_path)
    locked_str = netlist_decode(kgss_best, circuit_name, netlist_ori)
    # save the locked_str to the output file
    with open(output_file, 'w') as f:
        f.write(locked_str)





# Random Selection algorithm
def random_main(key_size, result_path, target_path):
    # set up
    kgss_ori, fitness_function, key_value, edge_dict = sa_setup(target_path, key_size)
    # result_list
    result = []
    best_re = []
    best_result = 1.0
    # random select one 
    for i in range(2000):
        kgss_rand = random_select(kgss_ori, edge_dict, key_value)
        print(kgss_rand)
        # run the attack on the rand kgss
        kpa = float(fitness_function.evaluate([kgss_rand])[0])
        if kpa < best_result:
            best_result = kpa
        best_re.append(best_result)
        result.append(kpa)
    with open(result_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # write header
        writer.writerow(['tried solution', 'best solution'])
        for re, be_re in zip(result, best_re):
            writer.writerow([re, be_re])
         
def set_up_exp(target_path, start_num, total_num, exp_num):
    # here the target_path is showing the original circuit path

    ml_data_path = "../ml_data_"
    circuit_name = target_path.split("/")[-1].split(".bench")[0].split("_")[0] # here is without "_C"
    
    circuit_ori = None
    itc_mark = False
    # get the original circuits
    for item in ['b14', 'b15', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'c1355', 'c1908', 'c2670', 'c3540', 'c5315', 'c6288', 'c7552']:
        if item in circuit_name:
            if "b" in item:
               circuit_ori = item + "_C"
               itc_mark = True
            if "c" in item:
                circuit_ori = item
    for item in ['arbiter', 'memory', 'multiplier', 'voter', "cva6"]:
        if item in circuit_name:
            circuit_ori = item
    target_path_base = "../data/original/" + circuit_name + "_ori"
    # create the folder 
    if not os.path.exists(target_path_base):
        os.makedirs(target_path_base)
    # based on the exp_num, we will know which kind of folder should be copied
    for j in range(start_num, start_num+total_num):
        # circuit name is circuit_name + str(i)
        circuit_name_temp = circuit_name + str(j - (exp_num-1)*total_num) # here is original we want to copy
        ml_data_ori_path = ml_data_path + circuit_name_temp
        circuit_name_copy = circuit_name + str(j) # here is the folder we want to copy to
        ml_data_copy_path = ml_data_path + circuit_name_copy
        # here I only want to copy the trained folder and D_MUX perl script
        # if the folder is empty 
        if os.path.exists(ml_data_copy_path) and len(os.listdir(ml_data_copy_path)) == 0:
            shutil.rmtree(ml_data_copy_path)
        if not os.path.exists(ml_data_copy_path):
            os.makedirs(ml_data_copy_path)
            shutil.copytree(ml_data_ori_path + "/trained_model", ml_data_copy_path + "/trained_model")
            shutil.copy(ml_data_ori_path + "/break_DMUX.pl", ml_data_copy_path + "/break_DMUX.pl")
        # also copy the circuit path which we need to have as target path for each model folder
        circuit_ori_path = "../data/original/" + circuit_ori + ".bench"
        circuit_copy_path = target_path_base + "/" + circuit_name_copy + ".bench"
        if "b" in circuit_name_copy:
            circuit_copy_path = target_path_base + "/" + circuit_name_copy + "_C.bench"
        if not os.path.exists(circuit_copy_path):
            shutil.copy(circuit_ori_path, circuit_copy_path)
    # also copy the circuit_ori_path to the target_path_base
    circuit_ori_path = "../data/original/" + circuit_ori + ".bench"
    circuit_copy_path = target_path_base + "/" + circuit_ori + ".bench"
    if not os.path.exists(circuit_copy_path):
        shutil.copy(circuit_ori_path, circuit_copy_path)

# encode the netlist str 
def netlist_str(target_path):
    netlist = BenchParser.instance().parse_file(target_path)
    netlist_string = netlist.to_string()
    return netlist_string    

def netlist_encode(netlist_str, circuit_name):
    muxlink = MuxLink(circuit_name)
    kgss = muxlink.encode(netlist_str)
    # print(dict(kgss.graph.nodes()))
    return kgss    

# decode the netlist and return the 
def netlist_decode_large(kgss, circuit_name, netlist_str):
    muxlink = MuxLink(circuit_name)
    locked_str = muxlink.decode_large(netlist_str, kgss)
    # print(dict(kgss.graph.nodes()))
    return locked_str

# decode the netlist and return the 
def netlist_decode(kgss, circuit_name, netlist_str):
    muxlink = MuxLink(circuit_name)
    locked_str = muxlink.decode(netlist_str, kgss)
    # print(dict(kgss.graph.nodes()))
    return locked_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SA")
    parser.add_argument("--key-size", type=int, help="key size")
    parser.add_argument("--result-path", type=str,  help="result path")
    parser.add_argument("--target-path", type=str, help="target path")
    parser.add_argument("--iteration", type=int,  help="iteration")
    parser.add_argument("--h-hop", type=int, help="hop size")
    parser.add_argument("--int-temp", type=int,  help="initial temp")
    parser.add_argument("--train-mark", type=str, help="train mark")
    parser.add_argument("--bin-num", type=int, help="bin num")
    parser.add_argument("--start-num", type=int, help="start num")
    parser.add_argument("--total-num", type=int, help="total num")
    parser.add_argument("--exp-num", type=int, help="exp num")
    parser.add_argument("--output-file", type=str, help="output file")
    args = parser.parse_args()
    key_size = args.key_size
    # set the result path
    # result_path = "sa_result_c2670_128bits_2.csv"
    result_path = args.result_path
    output_path = args.output_file
    # set the target path
    # target_path = "../data/original/c26706.bench"
    target_path = args.target_path
    # run the SA algorithm
    iteration_num = args.iteration
    hop_size = args.h_hop
    int_temp = args.int_temp
    chosen_sol = 0
    train_mark = args.train_mark
    bin_num = args.bin_num
    print("train_mark type: ", type(train_mark))
    print("train_mark: ", train_mark)
    if train_mark == "False":
        train_mark = False
    else:
        train_mark = True
    # train_mark = bool(train_mark)
    print("train_mark type: ", type(train_mark))
    print("train_mark: ", train_mark)
    start_num = args.start_num
    total_num = args.total_num
    exp_num = args.exp_num
    start_time = time.time()
    
    set_up_exp(target_path, start_num, total_num, exp_num)
    sa_main_vertical(key_size, result_path, target_path, iteration_num, hop_size, int_temp, chosen_sol, bin_num, train_mark, start_num, total_num, output_path)
    end_time = time.time()
    print("time: ", end_time - start_time)



    
