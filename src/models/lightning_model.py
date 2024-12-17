import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from .network import PulseDetectionNet 
from .loss import PulseLoss
from .eval_metrics import peak_error

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        self.model = PulseDetectionNet(
            in_channels=config.data.n_channels, 
            **self.config.network.__dict__
        )
        self.criterion = PulseLoss(**self.config.loss.__dict__)

    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        y_hat = self(x)
        loss, loss_components = self.criterion(y_hat, y)
        
        # Log all loss components
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, on_step=False, on_epoch=True)
        for name, value in loss_components.items():
            self.log(f'train_{name}', value)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        y_hat = self(x)
        loss, loss_components = self.criterion(y_hat, y)

        # Log all loss components
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, value in loss_components.items():
            self.log(f'val_{name}', value)
        
        # if batch_idx % 5 == 0:
        _, count_error, distance_error = peak_error(y_hat, y, peak_min_distance=self.config.loss.min_peak_distance, heights=[0.5])
        self.log('val_count_error', count_error[0], prog_bar=True)
        self.log('val_distance_error', distance_error[0], prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        y_hat = self(x)
        loss, loss_components = self.criterion(y_hat, y)
        _, count_error, distance_error = peak_error(y_hat, y, peak_min_distance=self.config.loss.min_peak_distance, heights=[0.75])
        # heights, count_error_rates, distance_errors = peak_error(y_hat, y, peak_min_distance=self.config.loss.min_peak_distance)
        # Log all loss components
        result_dict = {
            'test_loss': loss,
            'test_count_error': count_error[0],
            'test_distance_error': distance_error[0]
        }
        self.log_dict(result_dict, on_step=True, on_epoch=True, prog_bar=True)
        
        for name, value in loss_components.items():
            self.log(f'test_{name}', value)
            
        return result_dict
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # scheduler = CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=self.config.scheduler.T_max,
        #     T_mult=1,
        #     eta_min=self.config.scheduler.min_lr
        # )
        
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "interval": 'epoch',
            #     "frequency": 1
            # }
        }