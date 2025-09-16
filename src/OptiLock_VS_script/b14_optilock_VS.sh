#!/bin/bash
#SBATCH --job-name=b14_VS
#SBATCH -c 10
#SBATCH -t 168:00:00
#SBATCH --mem=70G
#SBATCH --output=OptiLock_%x_%A_%a.out
#SBATCH -a 0-0


# Setup an array of export parameters
export_params=(
    "OptiLock_result_b14_256_1_h_3_VS.csv 1 1"
)

export_params_set="${export_params[$SLURM_ARRAY_TASK_ID]}"
IFS=' ' read -r RESULT_PATH START_NUM EXP_NUM <<< ${export_params_set}


#Excute the python command
export ROOT_DIR="/scratch/zw3464/OptiLock/src"
export TARGET_PATH="../data/original/b14_ori/b14_C.bench"
export KEYSIZE=256
export ITER=10000
export H_HOP=3
export INT_TEMP=800
export TRAIN_MARK="False"
export BIN_NUM=1000
export TOTAL_NUM=5
export OUTPUT_FILE=../optilock_b14_locked.bench
python ${ROOT_DIR}/OptiLock_MuxLink_VS_Searching.py --key-size ${KEYSIZE} --result-path ${RESULT_PATH} --target-path ${TARGET_PATH} --iteration ${ITER} --int-temp ${INT_TEMP} --h-hop ${H_HOP} --train-mark ${TRAIN_MARK} --bin-num ${BIN_NUM} --start-num ${START_NUM} --total-num ${TOTAL_NUM} --exp-num ${EXP_NUM} --output-file ${OUTPUT_FILE}
