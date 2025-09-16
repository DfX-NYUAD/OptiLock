# this script is used to generate the code to train the code 
import os

def write_sbatch(file, job_name, memory, walltime, output, job_num, root_dir, circuit_name, h_hop, train_mark, key_size, start_num, total_num, iter_num, init_temp, bin_num, output_file):
    with open(file, 'w') as sbatch_file:
        sbatch_file.write("#!/bin/bash\n")
        sbatch_file.write("#SBATCH --job-name=" + job_name + "\n")
        sbatch_file.write("#SBATCH -c 1\n")
        sbatch_file.write("#SBATCH -t " + walltime + "\n")
        sbatch_file.write("#SBATCH --mem=" + memory + "\n")
        sbatch_file.write("#SBATCH --output=" + output + "_%" +"x" + "_%A_%" + "a.out\n")
        sbatch_file.write("#SBATCH -a 0-" + str(job_num -1) + "\n")
        sbatch_file.write("\n\n")

        sbatch_file.write("# Setup an array of export parameters\n")
        sbatch_file.write("export_params=(\n")
        start_num_list = []
        for i in range(job_num):
            start_num_temp = start_num + (i)*total_num
            start_num_list.append(start_num_temp)

        for i in range(1, job_num+1):
            # target_path = "../data/original/" + circuit_name + str(i) + ".bench"
            result_path = "OptiLock_result_" + circuit_name + "_" + str(key_size) + "_" + str(i) + "_h_" + str(h_hop) + "_HS.csv"
            start_num_temp = start_num_list[i-1]
            exp_num = i
            sbatch_file.write("    \"" + result_path + " " + str(start_num_temp) + " " + str(exp_num) + "\"\n")
            # sbatch_file.write("    \"" + target_path + "\"\n")
        sbatch_file.write(")\n\n")
        sbatch_file.write("export_params_set=\"${" + "export_params[$SLURM_ARRAY_TASK_ID]}\"\n")
        sbatch_file.write("IFS=" + "'" + " " + "'" + " read -r RESULT_PATH START_NUM EXP_NUM <<< ${" + "export_params_set}\n")
        sbatch_file.write("\n\n")
        sbatch_file.write("#Excute the python command\n")
        sbatch_file.write("export ROOT_DIR=\"" + root_dir + "\"\n")
        sbatch_file.write("export TARGET_PATH=\"" + "../data/original/" + circuit_name + "_ori/" + circuit_name + ".bench" + "\"\n")
        sbatch_file.write("export KEYSIZE=" + str(key_size) + "\n")
        sbatch_file.write("export ITER=" + str(iter_num) + "\n")
        sbatch_file.write("export H_HOP=" + str(h_hop) + "\n")
        sbatch_file.write("export INT_TEMP=" + str(init_temp) + "\n")
        sbatch_file.write("export TRAIN_MARK=" + "\"" +str(train_mark) + "\"" + "\n")
        sbatch_file.write("export BIN_NUM=" + str(bin_num) + "\n")
        sbatch_file.write("export TOTAL_NUM=" + str(total_num) + "\n")
        sbatch_file.write("export OUTPUT_FILE=" + str(output_file) + "\n")
        sbatch_file.write("module load abc\n")
        sbatch_file.write("python ${" + "ROOT_DIR}/OptiLock_MuxLink_HS_Searching.py --key-size ${"  +"KEYSIZE} --result-path ${" + "RESULT_PATH}" +" --target-path ${" + "TARGET_PATH} --iteration ${" + "ITER} --int-temp ${" + "INT_TEMP} --h-hop ${" + "H_HOP} --train-mark ${" + "TRAIN_MARK} --bin-num ${" + "BIN_NUM} --start-num ${" + "START_NUM} --total-num ${" + "TOTAL_NUM} --exp-num ${" + "EXP_NUM}\n")

if __name__ == '__main__':
    circuit_name_list = ["c1355"]
    for circuit_name in circuit_name_list:
        
        root_dir = os.getcwd()
        job_name = circuit_name + "_HS"
        memory = "70G"
        walltime = "168:00:00"
        output = "OptiLock"
        job_num = 1
        h_hop = 3
        train_mark = "False"
        key_size = 64
        start_num = 1
        total_num = 5
        iter_num = 10000 # need to change later
        init_temp = 800
        bin_num = 1
        output_file = "../optilock_" + circuit_name + "_locked.bench"
        train_sbatch_location = "./OptiLock_HS_script"
        target_path = "../data/original/" + circuit_name + "_ori/" + circuit_name + ".bench"
        # if there is no folder here, create a folder 
        if not os.path.exists(train_sbatch_location):
            os.makedirs(train_sbatch_location)
        file = train_sbatch_location + "/" + circuit_name + "_optilock_HS.sh"
        write_sbatch(file, job_name, memory, walltime, output, job_num, root_dir, circuit_name, h_hop, train_mark, key_size, start_num, total_num, iter_num, init_temp, bin_num, output_file)
        os.system("sbatch " + file)