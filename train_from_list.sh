#!/bin/bash

users_file=$1
config=$2

# Check if the users file exists
if [ ! -f "$users_file" ]; then
    echo "Error: Users file '$users_file' not found"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Process each user
while IFS= read -r user; do
    # Skip empty lines and lines starting with #
    [[ -z "$user" || "$user" =~ ^#.*$ ]] && continue
    
    echo "Submitting job for user: $user"
    
    # Submit training job for this user
    sbatch -o ./logs/${config}_leave_${user}.log \
           run_leave_out.sh $config $user   
    # Optional: add a small delay between submissions
    sleep 1
done < "$users_file"

echo "All jobs submitted. Check logs directory for output files." 