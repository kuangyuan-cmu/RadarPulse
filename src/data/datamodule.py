import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import PulseDataset

class PulseDataModule(pl.LightningDataModule):
    def __init__(self, data_path, data_config, batch_size, num_workers):
        super().__init__()
        # judge if data_config is a list
        self.data_path = data_path
        if isinstance(data_config, list):
            self.pulse_position = [config.position for config in data_config]
            self.signal_type = [config.signal_type for config in data_config]
            self.norm_2d = [config.norm_2d for config in data_config]
        else:
            self.pulse_position = data_config.position
            self.signal_type = data_config.signal_type
            self.norm_2d = data_config.norm_2d
            
        self.batch_size = batch_size
        self.num_workers = num_workers
       
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            print("Setting up training dataset")
            self.train_dataset = PulseDataset(
                f"{self.data_path}/train",
                self.pulse_position,
                self.signal_type,
                self.norm_2d
            )
            print("Setting up validation dataset")
            self.val_dataset = PulseDataset(
                f"{self.data_path}/dev",
                self.pulse_position,
                self.signal_type,
                self.norm_2d
            ) 
        if stage == 'validate':
            print("Setting up validation dataset")
            self.val_dataset = PulseDataset(
                f"{self.data_path}/dev",
                self.pulse_position,
                self.signal_type,
                self.norm_2d
            )
            
        if stage == 'test':
            print("Setting up testing dataset")
            self.test_dataset = PulseDataset(
                f"{self.data_path}/dev",
                self.pulse_position,
                self.signal_type,
                self.norm_2d
            )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )