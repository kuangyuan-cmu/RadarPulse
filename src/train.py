import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from data.datamodule import PulseDataModule
from models.lightning_model import LitModel
from config.config_utils import load_config

def main():
    # Load configuration
    config = load_config('src/config')
    
    # Initialize data module
    data_module = PulseDataModule(
        data_path=config.data.data_path,
        pulse_position=config.data.position,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers
    )
    data_module.setup(stage='fit')
    
    # Initialize model
    for batch in data_module.train_dataloader():
        n_channels = batch[0].shape[2]
        break
    model = LitModel(config, n_channels)
    
    # Initialize trainer
    # wandb_logger = WandbLogger(
    #     project="radar-pulse-detection",
    #     name="init_run",
    #     log_model=True,  # logs model checkpoints
    #     save_dir="./logs",  # local directory for logs
    #     offline=False,  # set True for offline logging
    #     notes="initial running",
    #     config=config  # log hyperparameters
    # )
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator='auto',
        devices='auto',
        # logger=wandb_logger,
    )
    
    # Start training
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()