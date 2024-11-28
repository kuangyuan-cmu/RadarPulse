import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from .network import PulseDetectionNet 
from .loss import PulseLoss

class LitModel(pl.LightningModule):
    def __init__(self, config, type, n_channels):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        if type == 'head':
            self.data_ind = 0
            self.label_ind = 1
        elif type == 'heart':
            self.data_ind = 2
            self.label_ind = 3
        elif type == 'wrist':
            self.data_ind = 4
            self.label_ind = 5
        else:
            raise ValueError("Invalid type. Must be one of 'head', 'heart', 'wrist'.")
        
        # Initialize network and loss
        self.model = PulseDetectionNet(
            in_channels=n_channels, 
            **self.config.network.__dict__
        )
        self.criterion = PulseLoss(**self.config.loss.__dict__)
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x = batch[self.data_ind]
        y = batch[self.label_ind]
        y_hat = self(x)
        loss, loss_components = self.criterion(y_hat, y)
        
        # Log all loss components
        self.log('train_loss', loss, prog_bar=True)
        for name, value in loss_components.items():
            self.log(f'train_{name}', value)
            
        return loss
        
    def validation_step(self, batch, batch_idx):
        x = batch[self.data_ind]
        y = batch[self.label_ind]
        y_hat = self(x)
        loss, loss_components = self.criterion(y_hat, y)
        
        # Log all loss components
        self.log('val_loss', loss, prog_bar=True)
        for name, value in loss_components.items():
            self.log(f'val_{name}', value)
            
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.scheduler.T_max,
            eta_min=self.config.scheduler.min_lr
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }