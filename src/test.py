import pytorch_lightning as pl
from data.datamodule import PulseDataModule
from models.lightning_model import LitModel
from config.config_utils import load_config
import argparse

def test(config, checkpoint_path):
    # Initialize data module
    data_module = PulseDataModule(
        data_path=config.data.data_path,
        pulse_position=config.data.position,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        norm_2d=config.data.norm_2d,
    )

    model = LitModel(config)
    
    # Initialize trainer for testing
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
    )
    
    # Run test
    trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file for testing')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    
    if not args.checkpoint:
        raise ValueError("Checkpoint path must be provided for testing")
    checkpoint = args.checkpoint
    if not args.config:
        config = load_config('src/config')
    else:
        config = load_config(args.config, env=args.config)
        
    # config = load_config('src/config')
    # checkpoint = '/home/kyuan/RadarPulse/checkpoints/dataset/phase1_new_1214/wrist_kernel7-epoch=17-val_loss=0.21.ckpt'
    # checkpoint = '/home/kyuan/RadarPulse/checkpoints/dataset/phase1_new_1214/heart_try-epoch=45-val_loss=0.13.ckpt'
    
    test(config, checkpoint)

if __name__ == '__main__':
    main()