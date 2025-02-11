#!/bin/bash

config=$1
dataset_path="dataset/ultragood_v5/processed"

# Create logs directory if it doesn't exist
mkdir -p logs

# Get all user folders and submit a job for each
for user_path in ${dataset_path}/*/; do
    # Extract just the user name from the path
    user=$(basename "$user_path")
    # if [ "$user" = "kuang" ] || [ "$user" = "biran" ]; then
    #     continue
    # fi
    
    echo "Submitting job for user: $user"
    sbatch -o ./logs/${config}_leave_${user}.log run_leave_out.sh $config $user
    
    # Optional: add a small delay between submissions to prevent overwhelming the scheduler
    sleep 1
done

echo "All jobs submitted. Check logs directory for output files." 