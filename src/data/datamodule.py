import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import PulseDataset

class PulseDataModule(pl.LightningDataModule):
    def __init__(self, data_path, data_config, batch_size, num_workers, leave_out_users=None):
        super().__init__()
        self.data_path = data_path
        self.data_config = data_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.leave_out_users = leave_out_users
        
    def setup(self, stage=None):
        if self.leave_out_users is None:
            if stage == 'fit' or stage is None:
                print("Setting up training dataset")
                self.train_dataset = PulseDataset(
                    f"{self.data_path}/train",
                    self.data_config,
                )
                print("Setting up validation dataset")
                self.val_dataset = PulseDataset(
                    f"{self.data_path}/dev",
                    self.data_config,
                )
            if stage == 'validate':
                print("Setting up validation dataset")
                self.val_dataset = PulseDataset(
                    f"{self.data_path}/dev",
                    self.data_config
                )
            if stage == 'test':
                print("Setting up testing dataset")
                self.test_dataset = PulseDataset(
                    f"{self.data_path}/dev",
                    self.data_config
                )
        else:
            print(f"Setting up dataset for leave-out users: {self.leave_out_users}")
            if stage == 'fit' or stage is None:
                self.train_dataset = PulseDataset(
                    f"{self.data_path}",
                    self.data_config,
                    exclude_users=self.leave_out_users,
                    enable_augment=True
                )
                self.val_dataset = PulseDataset(
                    f"{self.data_path}",
                    self.data_config,
                    include_users=self.leave_out_users,
                )
            if stage == 'validate':
                self.val_dataset = PulseDataset(
                    f"{self.data_path}",
                    self.data_config,
                    include_users=self.leave_out_users
                )
            if stage == 'test':
                self.test_dataset = PulseDataset(
                    f"{self.data_path}",
                    self.data_config,
                    include_users=self.leave_out_users
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