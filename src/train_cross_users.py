import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from data.datamodule import PulseDataModule
from models.lightning_model import LitModel
from config.config_utils import load_config
import torch
import argparse
from pathlib import Path

def train_single_fold(config, left_out_user, exp_name=None):
    pl.seed_everything(config.training.seed, workers=True)
    
    # Initialize data module with leave-out user
    data_module = PulseDataModule(
        data_path=config.data.data_path,
        data_config=config.data,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        leave_out_users=[left_out_user]  # Pass the user to leave out
    )
    
    model = LitModel(config)
    
    # Modify experiment name to include left out user
    fold_exp_name = f"{config.data.data_path}{config.data.position}-{config.data.signal_type}_leave_out_{left_out_user}"
    if exp_name:
        fold_exp_name = f"{fold_exp_name}_{exp_name}"
    
    loss_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename=fold_exp_name + '-loss-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        mode='min',
        save_last=False
    )

    wandb_logger = WandbLogger(
        project="radar-pulse-detection-leave-one-out",
        name=fold_exp_name,
        log_model=True,
        save_dir="./logs",
        offline=False,
        config=config
    )
    
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator='auto',
        devices='auto',
        logger=wandb_logger,
        callbacks=[loss_checkpoint_callback],
    )
    
    trainer.fit(model, data_module)
    return loss_checkpoint_callback.best_model_path

def get_all_users(data_path):
    data_path = Path(data_path)
    users = [d.name for d in data_path.glob('processed/*/') if d.is_dir()]
    return sorted(users)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--expname', type=str, help='Name of the experiment')
    args = parser.parse_args()
    
    if not args.config:
        config = load_config('src/config')
    else:
        config = load_config('src/config', env=args.config)
    
    # Get list of all users
    users = get_all_users(config.data.data_path)
    print(f"Found {len(users)} users: {users}")
    
    # Train one model for each left-out user
    best_models = {}
    for user in users:
        print(f"\nTraining model leaving out user: {user}")
        best_model_path = train_single_fold(config, user, args.expname)
        best_models[user] = best_model_path
    
    # Print summary of all trained models
    print("\nTraining completed. Best models for each fold:")
    for user, model_path in best_models.items():
        print(f"User {user}: {model_path}")

if __name__ == '__main__':
    main() 