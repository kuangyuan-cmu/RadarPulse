#!/bin/bash

# Find all checkpoint files
for ckpt in checkpoints/dataset/ultragood_v2_0123_psc/head/*.ckpt; do
    # Extract the checkpoint name without path and extension
    ckpt_name=$(basename "$ckpt" .ckpt)
    
    # Extract the user name from between "leave_out_" and "-loss"
    user_name=$(echo "$ckpt_name" | grep -o "leave_out_[^-]*" | sed 's/leave_out_//')
    
    echo "Testing checkpoint: $ckpt"
    echo "Leave out user: $user_name"
    
    # Run the test script with the extracted user name
    python src/test.py \
        --checkpoint "$ckpt" \
        --config head \
        --name "results_${ckpt_name}" \
        --leave_out_users "$user_name" \
        --debug \
        --name "head_${user_name}"
done 