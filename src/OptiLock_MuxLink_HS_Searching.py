from ec.impl.muxlink_fitness_function_plus import MuxLinkFitnessFunctionPlus
from muxlink.muxlink import MuxLink
from muxlink.original import *
from utils.bench_parser import BenchParser
# from ec.impl.watch_dog import KPAFileHandler
from watchdog.observers import Observer

import random
from sa.dual_anneal_new import *
import math
import csv
import sys
import argparse
import shutil
import time
import subprocess
import threading


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
def kgss_lock(kgss, sol_vector, key_value, edge_dict):
    # edge_dict = kgss.get_all_possible_edges()

    kgss_data = []
    for i in range(0, len(sol_vector), 2):
        pair1 = int(math.floor(sol_vector[i]))
        # pair1 = int(math.ceil(sol_vector[i]/2.)*2)
        pair2 = int(math.floor(sol_vector[i + 1]))
        # print("pairs:", pair1, pair2)
        pair_f1_g1 = edge_dict[pair1]
        pair_f2_g2 = edge_dict[pair2]
        key_temp = key_value[int(i / 2)]  # random.randint(0, 1)
        key_value += str(key_temp)

        kgss_temp = [str(pair_f1_g1[0]), str(pair_f2_g2[0]), str(pair_f1_g1[1]), str(pair_f2_g2[1]), str(key_temp),
                     int(i / 2)]

        kgss_data.append(kgss_temp)
    kgss.data = kgss_data
    return kgss

# convert kgss data to solution vector
def kgss_to_sol(kgss, edge_dict):
    sol_vector = []
    for i in range(len(kgss.data)):
        pair = kgss.data[i]
        pair_f1_g1 = (int(pair[0]), int(pair[2]))
        pair_f2_g2 = (int(pair[1]), int(pair[3]))
        for key, value in edge_dict.items():
            if value == pair_f1_g1:
                sol_vector.append(key)
            if value == pair_f2_g2:
                sol_vector.append(key)
    return sol_vector

# set up some parameter for SA function
def sa_setup(target_path, key_size, h_hop, chosen_sol, train_mark):
    # set key value
    key_value = ''.join(str(random.randint(0, 1)) for _ in range(key_size))
    print(key_value)
    # Load the benchmark
    netlist = BenchParser.instance().parse_file(target_path)
    netlist_str = netlist.to_string()
    # create fitness function
    fitness_function = None
    # create original kgss object
    circuit_name = target_path.split("/")[-1].split(".bench")[0].split("_")[0]
    kgss_ori = netlist_encode(netlist_str, circuit_name)
    # edge_dict = kgss_ori.get_all_possible_edges()
    edge_dict = kgss_ori.get_fan_cone_edges()

    return kgss_ori, fitness_function, key_value, edge_dict, circuit_name

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
    run_command = "slurm_parallel_ja_submit.sh -t 00:05:00 -q small -j " + circuit_name + str(start_num) + "mux"  + " " + command_path
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
        run_command_run =  "slurm_parallel_ja_submit.sh -t 00:05:00 -q small -j " + circuit_name[:-2] + str(start_num) + "mux"  + " " + command_path
        run_command_run = run_command_run.split()
        output = subprocess.run(run_command_run, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
            
    print("do i have any output? ", output, flush=True)
    if "error" in output:
        return False
    else:
        return True




# SA evaluation function
def fun_eval(x, kgss_ori, fitness_function, key_value, founded_pairs, edge_dict, thread_num, command_multi, circuit_name, bin_num, start_num, total_num, h_hop):
    x_temp = list(x)
    # print("Debug: x_temp:", x_temp)
    if x_temp in founded_pairs:
        return 1.2
    if len(set(x_temp)) != len(x_temp):
        print("same node?")
        return 1.2  # try to avoid the same node
    kgss = kgss_lock(kgss_ori, x, key_value, edge_dict)
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
        all_kpa_log = [f for f in all_files if "kpa" in f]
        if len(all_kpa_log) == 0:
            continue
        else:
            all_kpa_log_list.append(all_kpa_log[0])
    while len(all_kpa_log_list) < total_num:
        all_kpa_log_list = []
        for i in range(start_num, start_num+total_num):
            location = "../ml_data_" + circuit_name + str(i)
            all_files = os.listdir(location)
            all_kpa_log = [f for f in all_files if "kpa" in f]
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
            kpa_str = kpa_log.read()
            acc, prec, kpa = [float(i) for i in kpa_str.split(",")]
            kpa_list.append(kpa)
            acc_list.append(acc)
            prec_list.append(prec)
    # average the kpa value
    kpa = sum(kpa_list)/len(kpa_list)
    acc = sum(acc_list)/len(acc_list)
    prec = sum(prec_list)/len(prec_list)
    x_unpred = (kpa-acc)/kpa # here is x_unpred
    print("acc:", acc, "prec:", prec, "kpa:", kpa, "x_unpred", x_unpred)
    # remove the kpa log file
    for i in range(total_num):
        location = "../ml_data_" + circuit_name + str(i+start_num)
        kpa_log_path = location + "/kpa.txt"
        os.remove(kpa_log_path)

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
def sa_main(key_size, result_path, target_path, iteration_num, h_hop, int_temp, chosen_sol, bin_num, train_mark, start_num, total_num, output_file):
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
    # get the impacted output
    impacted_output = kgss_ori.get_impacted_output()
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

    #based on the final best solution, we can decode the netlist
    kgss_best = kgss_lock(kgss_ori, best_solution, key_value, edge_dict, [])
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
         
# function to set up the experiment at the same time
# set up the folder coressponding with the circuit name
# here we assume the trained model is saved in the original ml_data folder
# instead of coping the trained model manually, we copy all the folders by coding
#################################################################################
# In order to make sure the folder simple, we will copy these files like this:
# 1) target_path: we need to bring all the cricuit path to the circuit_name_ori folder
# 2) ml_data_folder: we also need to create folder for the ml_data folder;
#                    after the experiment, pass them into the ml_data_save folder 
# 3) start_num: it is indicating that we are using this part of circuit temp folder
# 4) total_num: it is indicating that we are using the total number of circuit temp folder
# 5) exp_num: it is indicating that how many same experiments we want to lanuch.
#################################################################################
def set_up_exp(target_path, start_num, total_num, exp_num):
    # here the target_path is showing the original circuit path

    ml_data_path = "../ml_data_"
    circuit_name = target_path.split("/")[-1].split(".bench")[0].split("_")[0] # here is without "_C"
    
    circuit_ori = None
    itc_mark = False
    # get the original circuits
    for item in ['b14', 'b15', 'b17', 'b20', 'b21', 'b22', 'c1355', 'c1908', 'c2670', 'c3540', 'c5315', 'c6288', 'c7552']:
        if item in circuit_name:
            if "b" in item:
               circuit_ori = item + "_C"
               itc_mark = True
            if "c" in item:
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
        

if __name__ == '__main__':
    # set the key size
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
    result_path = args.result_path
    output_path = args.output_file
    # set the target path
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
    sa_main(key_size, result_path, target_path, iteration_num, hop_size, int_temp, chosen_sol, bin_num, train_mark, start_num, total_num, output_path)
    end_time = time.time()
    print("time: ", end_time - start_time)



    
