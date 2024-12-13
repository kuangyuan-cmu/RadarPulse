import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import PulseDataset

class PulseDataModule(pl.LightningDataModule):
    def __init__(self, data_path, pulse_position, batch_size, num_workers):
        super().__init__()
        self.data_path = data_path
        self.pulse_position = pulse_position
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            print("Setting up training dataset")
            self.train_dataset = PulseDataset(
                f"{self.data_path}/train",
                self.pulse_position
            )
            print("Setting up validation dataset")
            self.val_dataset = PulseDataset(
                f"{self.data_path}/dev",
                self.pulse_position
            ) 
            
        if stage == 'test':
            print("Setting up testing dataset")
            self.test_dataset = PulseDataset(
                f"{self.data_path}/dev",
                self.pulse_position
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