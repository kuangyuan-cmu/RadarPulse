#!/bin/bash

#SBATCH -p GPU-small
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH --cpus-per-gpu 5
#SBATCH --gpus=v100-32:1

module load anaconda3
python src/train_joint.py \
    --head-checkpoint checkpoints/dataset/ultragood_v5/head-phase_leave_out_test1-loss-epoch=29-val_loss=0.191.ckpt \
    --heart-checkpoint checkpoints/dataset/ultragood_v5/heart-both_leave_out_test1-loss-epoch=49-val_loss=0.132.ckpt \
    --wrist-checkpoint checkpoints/dataset/ultragood_v5/wrist-both_leave_out_test1-loss-epoch=24-val_loss=0.186.ckpt \
    --neck-checkpoint checkpoints/dataset/ultragood_v5/neck-both_leave_out_test1-loss-epoch=26-val_loss=0.154.ckpt \
    --leave_out_users test1
    # --head-checkpoint checkpoints/dataset/ultragood_v5/head-phase_leave_out_test2-loss-epoch=15-val_loss=0.185.ckpt \
    # --heart-checkpoint checkpoints/dataset/ultragood_v5/heart-both_leave_out_test2-loss-epoch=21-val_loss=0.156.ckpt \
    # --wrist-checkpoint checkpoints/dataset/ultragood_v5/wrist-both_leave_out_test2-loss-epoch=17-val_loss=0.184.ckpt \
    # --neck-checkpoint checkpoints/dataset/ultragood_v5/neck-both_leave_out_test2-loss-epoch=16-val_loss=0.153.ckpt \
    # --leave_out_users test2
    




###sbatch -o ./logs/joint_kuangbiran.log run_joint_train.sh


#  --head-checkpoint checkpoints/dataset/ultragood_v5/head-phase_leave_out_test1-loss-epoch=29-val_loss=0.191.ckpt \
#  --heart-checkpoint checkpoints/dataset/ultragood_v5/heart-both_leave_out_test1-loss-epoch=49-val_loss=0.132.ckpt \
#  --wrist-checkpoint checkpoints/dataset/ultragood_v5/wrist-both_leave_out_test1-loss-epoch=24-val_loss=0.186.ckpt \
#  --neck-checkpoint checkpoints/dataset/ultragood_v5/neck-both_leave_out_test1-loss-epoch=26-val_loss=0.154.ckpt \
#  --leave_out_users test1