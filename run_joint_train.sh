#!/bin/bash

#SBATCH -p GPU-small
#SBATCH -N 1
#SBATCH -t 1:30:00
#SBATCH --cpus-per-gpu 5
#SBATCH --gpus=v100-32:1

module load anaconda3
python src/train_joint.py \
    --head-checkpoint checkpoints/dataset/ultragood_v5/head-phase_leave_out_biran,kuang,chinmaya,tingwei_16ch-loss-epoch=29-val_loss=0.196.ckpt \
    --heart-checkpoint checkpoints/dataset/ultragood_v5/heart-both_leave_out_kuang,biran,chinmaya,tingwei-loss-epoch=49-val_loss=0.119.ckpt \
    --wrist-checkpoint checkpoints/dataset/ultragood_v5/wrist-both_leave_out_biran,kuang,chinmaya,tingwei_4ch-loss-epoch=15-val_loss=0.201.ckpt \
    --neck-checkpoint checkpoints/dataset/ultragood_v5/neck-both_leave_out_biran,kuang,chinmaya,tingwei_8ch-loss-epoch=34-val_loss=0.159.ckpt \
    --leave_out_users biran,kuang,chinmaya,tingwei

###sbatch -o ./logs/joint_kuangbiran.log run_joint_train.sh

#  --head-checkpoint checkpoints/dataset/ultragood_v4/head-phase_leave_out_kuang-loss-epoch=04-val_loss=0.21.ckpt \
#  --heart-checkpoint checkpoints/dataset/ultragood_v4/heart-both_leave_out_kuang-loss-epoch=23-val_loss=0.13.ckpt \
#  --wrist-checkpoint checkpoints/dataset/ultragood_v4/wrist-both_leave_out_kuang-loss-epoch=10-val_loss=0.24.ckpt \
#  --neck-checkpoint checkpoints/dataset/ultragood_v4/neck-both_leave_out_kuang-loss-epoch=17-val_loss=0.14.ckpt \
#  --leave_out_users kuang