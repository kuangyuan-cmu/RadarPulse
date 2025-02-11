#!/bin/bash

#SBATCH -p GPU-small
#SBATCH -N 1
#SBATCH -t 0:45:00
#SBATCH --cpus-per-gpu 5
#SBATCH --gpus=v100-32:1


# Find all checkpoint files
dataset=$1
site=$2

if [ -z "$dataset" ] || [ -z "$site" ]; then
    echo "Usage: $0 <dataset> <site>"
    echo "Example: $0 ultragood_v5 head"
    exit 1
fi

module load anaconda3
for ckpt in checkpoints/dataset/${dataset}/${site}/*.ckpt; do
    # Extract the checkpoint name without path and extension
    ckpt_name=$(basename "$ckpt" .ckpt)
    
    # Extract the user name from between "leave_out_" and "-loss"
    user_name=$(echo "$ckpt_name" | grep -o "leave_out_[^-]*" | sed 's/leave_out_//')
    
    echo "Testing checkpoint: $ckpt"
    echo "Leave out user: $user_name"
    
    # Run the test script with the extracted user name
    python src/test.py \
        --checkpoint "$ckpt" \
        --config ${site} \
        --leave_out_users "$user_name" \
        --debug
done

###sbatch -o ./logs/testing_neck.log run_tests_all_users.sh