import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from data.datamodule import PulseDataModule
from models.lightning_model_joint import LitModel_joint
from config.config_utils import load_config
import torch
import argparse
import numpy as np

def test(configs, checkpoint_path, exp_name, enable_fusion=True):
    config_default = load_config('src/config', 'joint', to_class=False)

    data_module = PulseDataModule(
        data_path=config_default.data.data_path,
        data_config=[config.data for config in configs],
        batch_size=config_default.training.batch_size,
        num_workers=config_default.training.num_workers,   
    )

    model = LitModel_joint(configs, training_config=config_default, enable_fusion=enable_fusion)

    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
    )
    
    trainer.test(model, data_module, ckpt_path=checkpoint_path)
    if exp_name:
        np.save(f'results/{exp_name}.npy', model.results, allow_pickle=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='Name of the experiment')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file for head model')
    parser.add_argument('--head-model-config', type=str, help='Pateto config file for head model')
    parser.add_argument('--heart-model-config', type=str, help='Path to config file for heart model')
    parser.add_argument('--wrist-model-config', type=str, help='Path to config file for wrist model')
    parser.add_argument('--disable-fusion', action='store_true', help='Disable fusion', default=False)
    args = parser.parse_args()

    if not args.checkpoint:
        raise ValueError("Checkpoint path must be provided for testing")
    checkpoint = args.checkpoint
    enable_fusion = not args.disable_fusion
    configs = [
        load_config('src/config', env=args.head_model_config, to_class=False),
        load_config('src/config', env=args.heart_model_config, to_class=False),
        load_config('src/config', env=args.wrist_model_config, to_class=False)
    ]
        
    test(configs, checkpoint, args.name, enable_fusion)
    
if __name__ == '__main__':
    main()
    