import pytorch_lightning as pl
from data.datamodule import PulseDataModule
from models.lightning_model import LitModel
from config.config_utils import load_config

def main():
    # Load configuration
    config = load_config('src/config')
    
    # Initialize data module
    data_module = PulseDataModule(
        data_path=config.data.data_path,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers
    )
    data_module.setup(stage='fit')
    
    # Initialize model
    for batch in data_module.train_dataloader():
        # head_data_shape = batch[0].shape
        heart_data_shape = batch[2].shape
        # wrist_data_shape = batch[4].shape
        break
    channels_heart = heart_data_shape[2]
    model = LitModel(config, type='heart', n_channels=channels_heart)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator='auto',
        devices='auto',
    )
    
    # Start training
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()