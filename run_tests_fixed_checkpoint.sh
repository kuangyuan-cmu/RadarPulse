#!/bin/bash

#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -t 0:20:00
#SBATCH --cpus-per-gpu 5
#SBATCH --gpus=v100-32:1

# Check if required arguments are provided
# if [ "$#" -ne 4 ]; then
#     echo "Usage: $0 <users_file> <checkpoint_path> <site> <dataset>"
#     echo "Example: $0 users.txt checkpoints/model.ckpt head ultragood_v5"
#     exit 1
# fi

users_file=$1
checkpoint=$2
site=$3

# Check if the files exist
if [ ! -f "$users_file" ]; then
    echo "Error: Users file '$users_file' not found"
    exit 1
fi

if [ ! -f "$checkpoint" ]; then
    echo "Error: Checkpoint file '$checkpoint' not found"
    exit 1
fi

module load anaconda3

# Check if users file has any non-empty, non-comment lines
user_count=0
while IFS= read -r username; do
    [[ -z "$username" || "$username" =~ ^#.*$ ]] && continue
    ((user_count++))
done < "$users_file"

if [ $user_count -eq 0 ]; then
    echo "Warning: No valid users found in '$users_file'"
    echo "The file appears to be empty or contains only empty lines/comments"
    exit 1
fi

echo "Found $user_count user(s) to process"

# Process each user (assuming one username per line)
while IFS= read -r username; do
    # Skip empty lines and lines starting with #
    [[ -z "$username" || "$username" =~ ^#.*$ ]] && continue
    
    echo "Processing user: $username for site: $site"
    
    # Run the test script
    python src/test.py \
        --checkpoint "$checkpoint" \
        --config "$site" \
        --leave_out_users "$username" \
        --debug
        
    if [ $? -eq 0 ]; then
        echo "Test completed successfully for user: $username"
    else
        echo "Error running test for user: $username"
    fi
done < "$users_file" 