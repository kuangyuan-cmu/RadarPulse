#!/bin/bash

#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -t 0:45:00
#SBATCH --cpus-per-gpu 5
#SBATCH --gpus=v100-32:1

config=$1
user=$2
output_file="logs/train_${config}_leave_out_${user}.log"

module load anaconda3
python3 src/train.py --config $config --leave_out_users $user

###sbatch -o ./logs/heart_leave_yiwei.log run_leave_out.sh heart yiwei