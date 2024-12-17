import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from data.datamodule import PulseDataModule
from models.lightning_model import LitModel
from config.config_utils import load_config
import torch
import argparse

def train(config, checkpoint_path=None):
    pl.seed_everything(config.training.seed, workers=True)
    # Initialize data module
    data_module = PulseDataModule(
        data_path=config.data.data_path,
        pulse_position=config.data.position,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        norm_2d=config.data.norm_2d,
    )

    model = LitModel(config)
    if checkpoint_path:
        model.load_state_dict(torch.load('checkpoints/dataset/phase1_new_1214/heart_try-epoch=45-val_loss=0.13.ckpt')['state_dict'])
    
    # Initialize trainer
    exp_name = f"{config.data.data_path}{config.data.position}_spatial_conv_top4_wd1e-1"
    
    loss_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename= exp_name + '-loss-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,  # Save top 3 models
        monitor='val_loss',
        mode='min',
        save_last=False  # Additionally save the last model
    )
    # count_checkpoint_callback = ModelCheckpoint(
    #     dirpath='checkpoints',
    #     filename= exp_name + '-count-{epoch:02d}-{val_count_error:.2f}',
    #     save_top_k=1,  # Save top 3 models
    #     monitor='val_count_error',
    #     mode='min',
    #     save_last=False  # Additionally save the last model
    # )
    # pd_checkpoint_callback = ModelCheckpoint(
    #     dirpath='checkpoints',
    #     filename= exp_name + '-pd-{epoch:02d}-{val_distance_error:.2f}',
    #     save_top_k=1,  # Save top 3 models
    #     monitor='val_distance_error',
    #     mode='min',
    #     save_last=False  # Additionally save the last model
    # )
    
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
    args = parser.parse_args()
    
    if not args.config:
        config = load_config('src/config')
    else:
        config = load_config(args.config, env=args.config)
        
    train(config, args.checkpoint)
    
if __name__ == '__main__':
    main()