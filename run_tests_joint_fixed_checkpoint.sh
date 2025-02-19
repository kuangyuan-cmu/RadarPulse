#!/bin/bash

#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -t 0:30:00
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
disable_fusion=${3:-false}  # Default to true if not provided

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

# Process each user (assuming one username per line)
while IFS= read -r username; do
    # Skip empty lines and lines starting with #
    [[ -z "$username" || "$username" =~ ^#.*$ ]] && continue
    
    echo "Processing user: $username"
    
    # Run the test script with conditional disable-fusion flag
    if [ "$disable_fusion" = "true" ]; then
        python src/test_joint.py \
            --checkpoint "$checkpoint" \
            --leave_out_users "$username" \
            --disable-fusion
    else
        python src/test_joint.py \
            --checkpoint "$checkpoint" \
            --leave_out_users "$username"
    fi
        
    if [ $? -eq 0 ]; then
        echo "Test completed successfully for user: $username"
    else
        echo "Error running test for user: $username"
    fi
done < "$users_file" 