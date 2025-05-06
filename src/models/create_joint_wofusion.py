import sys
from pathlib import Path
# Add src to Python path
src_path = str(Path(__file__).parent.parent)
sys.path.append(src_path)

import pytorch_lightning as pl
from models.lightning_model_joint import LitModel_joint
from config.config_utils import load_config
import torch
import argparse

def create_joint_model(configs, checkpoint_paths, output_path):
    config_default = load_config('src/config', 'joint')
    pl.seed_everything(configs[0].training.seed, workers=True)
    
    # Create model without fusion
    model = LitModel_joint(configs, training_config=config_default, 
                          checkpoint_paths=checkpoint_paths, enable_fusion=False)
    
    # Save the joint model checkpoint in Lightning format
    checkpoint = {
        'state_dict': model.state_dict(),
        'hyper_parameters': {
            'config_list': configs,
            'training_config': config_default
        },
        'epoch': 0,
        'global_step': 0,
        'pytorch-lightning_version': pl.__version__,
        'callbacks': {},
        'optimizer_states': [],
        'lr_schedulers': [],
        'NativeMixedPrecisionPlugin': None,
        'version': None
    }
    torch.save(checkpoint, output_path)
    print(f'Saved joint model to {output_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--head-checkpoint', type=str, help='Path to checkpoint file for head model')
    parser.add_argument('--heart-checkpoint', type=str, help='Path to checkpoint file for heart model')
    parser.add_argument('--wrist-checkpoint', type=str, help='Path to checkpoint file for wrist model')
    parser.add_argument('--neck-checkpoint', type=str, help='Path to checkpoint file for neck model')
    
    parser.add_argument('--head-model-config', type=str, help='Path to config file for head model', default='head')
    parser.add_argument('--heart-model-config', type=str, help='Path to config file for heart model', default='heart')
    parser.add_argument('--wrist-model-config', type=str, help='Path to config file for wrist model', default='wrist')
    parser.add_argument('--neck-model-config', type=str, help='Path to config file for neck model', default='neck')
    
    parser.add_argument('--output', type=str, required=True, help='Output path for joint model checkpoint')
    
    args = parser.parse_args()

    checkpoints = [args.head_checkpoint, args.heart_checkpoint, args.wrist_checkpoint, args.neck_checkpoint]

    configs = [
        load_config('src/config', env=args.head_model_config),
        load_config('src/config', env=args.heart_model_config),
        load_config('src/config', env=args.wrist_model_config),
        load_config('src/config', env=args.neck_model_config)
    ]
        
    create_joint_model(configs, checkpoints, args.output)
    
if __name__ == '__main__':
    main()
    


"""
Command to call:
python src/models/create_joint_wofusion.py \
    --head-checkpoint checkpoints/dataset/ultragood_v5_saved/head-phase_leave_out_test2-loss-epoch=28-val_loss=0.179.ckpt \
    --heart-checkpoint checkpoints/dataset/ultragood_v5_saved/heart-both_leave_out_test2-loss-epoch=43-val_loss=0.158.ckpt \
    --wrist-checkpoint checkpoints/dataset/ultragood_v5_saved/wrist-both_leave_out_test2-loss-epoch=48-val_loss=0.180.ckpt \
    --neck-checkpoint checkpoints/dataset/ultragood_v5_saved/neck-both_leave_out_test2-loss-epoch=26-val_loss=0.148.ckpt \
    --output checkpoints/dataset/ultragood_v5_saved/joint_wofusion_test2.ckpt
"""