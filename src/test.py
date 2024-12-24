import pytorch_lightning as pl
import torch
from data.datamodule import PulseDataModule
from models.lightning_model import LitModel
from config.config_utils import load_config
import argparse
import numpy as np

def test(config, checkpoint_path, exp_name, debug=False):
    # Initialize data module
    data_module = PulseDataModule(
        data_path=config.data.data_path,
        data_config=config.data,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    model = LitModel(config, debug=debug)
    
    # Initialize trainer for testing
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
    )
    
    # Run test
    trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_path)
    
    if exp_name:
        np.save(f'results/{exp_name}.npy', model.results, allow_pickle=True)
   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file for testing')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--name', type=str, help='Name of the experiment')
    parser.add_argument('--debug' , action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    if not args.checkpoint:
        raise ValueError("Checkpoint path must be provided for testing")
    checkpoint = args.checkpoint
    if not args.config:
        config = load_config('src/config', to_class=False)
    else:
        config = load_config('src/config', env=args.config, to_class=False)
    name = args.name
    
    # config = load_config('src/config', to_class=False)
    # # checkpoint = '/home/kyuan/RadarPulse/checkpoints/dataset/phase1_new_1214/wrist_kernel7-epoch=17-val_loss=0.21.ckpt'
    # checkpoint = '/home/kyuan/RadarPulse/checkpoints/dataset/phase1_new_1214_cross_sessions/heart_reducedfeature_top4-loss-epoch=23-val_loss=0.20.ckpt'
    # name = None
    
    test(config, checkpoint, name, debug=args.debug)

if __name__ == '__main__':
    main()