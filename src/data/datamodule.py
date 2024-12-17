import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import PulseDataset

class PulseDataModule(pl.LightningDataModule):
    def __init__(self, data_path, pulse_position, batch_size, num_workers, norm_2d):
        super().__init__()
        self.data_path = data_path
        self.pulse_position = pulse_position
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.norm_2d = norm_2d
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            print("Setting up training dataset")
            self.train_dataset = PulseDataset(
                f"{self.data_path}/train",
                self.pulse_position,
                self.norm_2d
            )
            print("Setting up validation dataset")
            self.val_dataset = PulseDataset(
                f"{self.data_path}/dev",
                self.pulse_position,
                self.norm_2d
            ) 
            
        if stage == 'test':
            print("Setting up testing dataset")
            self.test_dataset = PulseDataset(
                f"{self.data_path}/dev",
                self.pulse_position,
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