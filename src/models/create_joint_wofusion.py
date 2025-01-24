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
    parser.add_argument('--head-model-config', type=str, help='Path to config file for head model')
    parser.add_argument('--heart-model-config', type=str, help='Path to config file for heart model')
    parser.add_argument('--wrist-model-config', type=str, help='Path to config file for wrist model')
    parser.add_argument('--output', type=str, required=True, help='Output path for joint model checkpoint')
    
    args = parser.parse_args()

    checkpoints = [args.head_checkpoint, args.heart_checkpoint, args.wrist_checkpoint]

    configs = [
        load_config('src/config', env=args.head_model_config),
        load_config('src/config', env=args.heart_model_config),
        load_config('src/config', env=args.wrist_model_config)
    ]
        
    create_joint_model(configs, checkpoints, args.output)
    
if __name__ == '__main__':
    main()