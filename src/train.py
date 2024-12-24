import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from data.datamodule import PulseDataModule
from models.lightning_model import LitModel
from config.config_utils import load_config
import torch
import argparse

def train(config, checkpoint_path=None, exp_name=None):
    pl.seed_everything(config.training.seed, workers=True)
    # Initialize data module
    data_module = PulseDataModule(
        data_path=config.data.data_path,
        # pulse_position=config.data.position,
        # signal_type=config.data.signal_type,
        # norm_2d=config.data.norm_2d,
        data_config=config.data,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        
    )
    # data_module.setup()
    
    model = LitModel(config)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    
    # Initialize trainer
    exp_name = f"{config.data.data_path}{config.data.position}-{config.data.signal_type}_{exp_name}"
    
    loss_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename= exp_name + '-loss-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,  # Save top 3 models
        monitor='val_loss',
        mode='min',
        save_last=False  # Additionally save the last model
    )

    wandb_logger = WandbLogger(
        project="radar-pulse-detection",
        name=exp_name,
        log_model=True,  # logs model checkpoints
        save_dir="./logs",  # local directory for logs
        offline=False,  # set True for offline logging
        # notes="",
        config=config  # log hyperparameters
    )
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator='auto',
        devices='auto',
        logger=wandb_logger,
        callbacks=[loss_checkpoint_callback],
    )
    
    trainer.fit(model, data_module)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--expname', type=str, help='Name of the experiment')
    args = parser.parse_args()
    
    if not args.config:
        config = load_config('src/config')
    else:
        config = load_config('src/config', env=args.config)
        
    train(config, args.checkpoint, args.expname)
    
if __name__ == '__main__':
    main()
    