#!/bin/bash

#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -t 1:30:00
#SBATCH --cpus-per-gpu 5
#SBATCH --gpus=v100-32:1

module load anaconda3
python src/train_joint.py \
    # --head-checkpoint checkpoints/dataset/ultragood_v5/head-phase_leave_out_dev_3points-loss-epoch=32-val_loss=0.121.ckpt \
    # --heart-checkpoint checkpoints/dataset/ultragood_v5/heart-both_leave_out_dev_3points-loss-epoch=31-val_loss=0.093.ckpt \
    # --wrist-checkpoint checkpoints/dataset/ultragood_v5/wrist-both_leave_out_dev_3points-loss-epoch=14-val_loss=0.178.ckpt \
    # --leave_out_users dev_3points

    # --head-checkpoint checkpoints/dataset/ultragood_v5/head-phase_leave_out_dev3-loss-epoch=17-val_loss=0.201.ckpt \
    # --heart-checkpoint checkpoints/dataset/ultragood_v5/heart-both_leave_out_dev3-loss-epoch=21-val_loss=0.104.ckpt \
    # --wrist-checkpoint checkpoints/dataset/ultragood_v5/wrist-both_leave_out_dev3-loss-epoch=12-val_loss=0.175.ckpt \
    # --neck-checkpoint checkpoints/dataset/ultragood_v5/neck-both_leave_out_dev3-loss-epoch=16-val_loss=0.140.ckpt \
    # --leave_out_users dev3

    # --head-checkpoint checkpoints/dataset/ultragood_v5/head-phase_leave_out_dev2-loss-epoch=20-val_loss=0.166.ckpt \
    # --heart-checkpoint checkpoints/dataset/ultragood_v5/heart-both_leave_out_dev2-loss-epoch=28-val_loss=0.113.ckpt \
    # --wrist-checkpoint checkpoints/dataset/ultragood_v5/wrist-both_leave_out_dev2-loss-epoch=09-val_loss=0.197.ckpt \
    # --neck-checkpoint checkpoints/dataset/ultragood_v5/neck-both_leave_out_dev2-loss-epoch=47-val_loss=0.137.ckpt \
    # --leave_out_users dev2

    # --head-checkpoint  checkpoints/dataset/ultragood_v5/head-phase_leave_out_test1-loss-epoch=34-val_loss=0.182.ckpt\
    # --heart-checkpoint checkpoints/dataset/ultragood_v5/heart-both_leave_out_test1-loss-epoch=43-val_loss=0.130.ckpt \
    # --wrist-checkpoint checkpoints/dataset/ultragood_v5/wrist-both_leave_out_test1-loss-epoch=11-val_loss=0.171.ckpt \
    # --neck-checkpoint checkpoints/dataset/ultragood_v5/neck-both_leave_out_test1-loss-epoch=32-val_loss=0.158.ckpt \
    # --leave_out_users test1
    # --head-checkpoint checkpoints/dataset/ultragood_v5/head-phase_leave_out_test2-loss-epoch=28-val_loss=0.179.ckpt \
    # --heart-checkpoint checkpoints/dataset/ultragood_v5/heart-both_leave_out_test2-loss-epoch=43-val_loss=0.158.ckpt \
    # --wrist-checkpoint checkpoints/dataset/ultragood_v5/wrist-both_leave_out_test2-loss-epoch=48-val_loss=0.180.ckpt \
    # --neck-checkpoint checkpoints/dataset/ultragood_v5/neck-both_leave_out_test2-loss-epoch=26-val_loss=0.148.ckpt \
    # --leave_out_users test2

    # --head-checkpoint checkpoints/dataset/ultragood_v5/head-phase_leave_out_dev_3points-loss-epoch=32-val_loss=0.121.ckpt \
    # --heart-checkpoint checkpoints/dataset/ultragood_v5/heart-both_leave_out_dev_3points-loss-epoch=31-val_loss=0.093.ckpt \
    # --wrist-checkpoint checkpoints/dataset/ultragood_v5/wrist-both_leave_out_dev_3points-loss-epoch=14-val_loss=0.178.ckpt \
    # --leave_out_users dev_3points

    # --head-checkpoint checkpoints/dataset/ultragood_v5/head-phase_leave_out_dev2-loss-epoch=20-val_loss=0.166.ckpt \
    # --heart-checkpoint checkpoints/dataset/ultragood_v5/heart-both_leave_out_dev2-loss-epoch=28-val_loss=0.113.ckpt \
    # --wrist-checkpoint checkpoints/dataset/ultragood_v5/wrist-both_leave_out_dev2-loss-epoch=09-val_loss=0.197.ckpt \
    # --neck-checkpoint checkpoints/dataset/ultragood_v5/neck-both_leave_out_dev2-loss-epoch=47-val_loss=0.137.ckpt \
    # --leave_out_users dev2







###sbatch -o ./logs/joint_kuangbiran.log run_joint_train.sh


#  --head-checkpoint checkpoints/dataset/ultragood_v5/head-phase_leave_out_test1-loss-epoch=29-val_loss=0.191.ckpt \
#  --heart-checkpoint checkpoints/dataset/ultragood_v5/heart-both_leave_out_test1-loss-epoch=49-val_loss=0.132.ckpt \
#  --wrist-checkpoint checkpoints/dataset/ultragood_v5/wrist-both_leave_out_test1-loss-epoch=24-val_loss=0.186.ckpt \
#  --neck-checkpoint checkpoints/dataset/ultragood_v5/neck-both_leave_out_test1-loss-epoch=26-val_loss=0.154.ckpt \
#  --leave_out_users test1