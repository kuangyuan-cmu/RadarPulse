import pytorch_lightning as pl
import torch
from data.datamodule import PulseDataModule
from models.lightning_model import LitModel
from config.config_utils import load_config
import argparse
import numpy as np
import pandas as pd
import os

def test(config, checkpoint_path, exp_name, debug=False, leave_out_users=None):
    # Initialize data module
    data_module = PulseDataModule(
        data_path=config.data.data_path,
        data_config=config.data,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        leave_out_users=leave_out_users
    )

    model = LitModel(config, debug=debug)
    
    # Initialize trainer for testing
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
    )
    
    # Run test
    trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_path)
    
    if exp_name is None and leave_out_users is not None:
        exp_name = f'{leave_out_users}_{config.data.position}'
    if exp_name:
        # np.save(f'results/{exp_name}.npy', model.results, allow_pickle=True)
        os.makedirs(f'results/single_site_debug/{config.data.position}', exist_ok=True)
        pd.DataFrame(model.debug_metrics).to_csv(f'results/single_site_debug/{config.data.position}/{exp_name}.csv', index=False)
   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file for testing')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--name', type=str, help='Name of the experiment')
    parser.add_argument('--debug' , action='store_true', help='Debug mode')
    parser.add_argument('--leave_out_users', type=str, help='Leave out users')
    
    args = parser.parse_args()
    
    if not args.checkpoint:
        raise ValueError("Checkpoint path must be provided for testing")
    checkpoint = args.checkpoint
    if not args.config:
        config = load_config('src/config', to_class=False)
    else:
        config = load_config('src/config', env=args.config, to_class=False)
    name = args.name
    
    
    test(config, checkpoint, name, debug=args.debug, leave_out_users=args.leave_out_users)


if __name__ == '__main__':
    main()