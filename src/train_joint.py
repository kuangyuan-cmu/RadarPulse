import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from data.datamodule import PulseDataModule
from models.lightning_model_joint import LitModel_joint
from config.config_utils import load_config, combine_configs
import torch
import argparse

def train(configs, checkpoint_paths, joint_model_checkpoint_path=None):
    config_default = load_config('src/config', 'joint')
    pl.seed_everything(configs[0].training.seed, workers=True)
    # Initialize data module
    data_module = PulseDataModule(
        data_path=config_default.data.data_path,
        data_config=[config.data for config in configs],
        batch_size=config_default.training.batch_size,
        num_workers=config_default.training.num_workers,   
    )
    # data_module.setup()
    if not any(checkpoint_paths):
        checkpoint_paths = None
    model = LitModel_joint(configs, training_config=config_default, checkpoint_paths=checkpoint_paths)
    if checkpoint_paths is None and joint_model_checkpoint_path is not None:
        model.load_state_dict(torch.load(joint_model_checkpoint_path)['state_dict'])
        print(f'Loaded joint model from {joint_model_checkpoint_path}')
    
    # Initialize trainer
    exp_name = f"{config_default.data.data_path}Joint-reversebottleneck"
    
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
        config=combine_configs(config_default, *configs),
    )
    trainer = pl.Trainer(
        max_epochs=config_default.training.max_epochs,
        accelerator='auto',
        devices='auto',
        logger=wandb_logger,
        callbacks=[loss_checkpoint_callback],
    )

    # trainer.validate(model, datamodule=data_module)
    trainer.fit(model, data_module)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--head-checkpoint', type=str, help='Path to checkpoint file for head model')
    parser.add_argument('--heart-checkpoint', type=str, help='Path to checkpoint file for heart model')
    parser.add_argument('--wrist-checkpoint', type=str, help='Path to checkpoint file for wrist model')
    parser.add_argument('--head-model-config', type=str, help='Path to config file for head model')
    parser.add_argument('--heart-model-config', type=str, help='Path to config file for heart model')
    parser.add_argument('--wrist-model-config', type=str, help='Path to config file for wrist model')
    
    parser.add_argument('--joint-model-checkpoint', type=str, help='Path to config file for wrist model')
    args = parser.parse_args()

    checkpoints = [args.head_checkpoint, args.heart_checkpoint, args.wrist_checkpoint]

    configs = [
        load_config('src/config', env=args.head_model_config),
        load_config('src/config', env=args.heart_model_config),
        load_config('src/config', env=args.wrist_model_config)
    ]
        
    train(configs, checkpoints, args.joint_model_checkpoint)
    
if __name__ == '__main__':
    main()
    