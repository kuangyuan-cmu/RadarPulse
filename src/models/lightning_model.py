import pytorch_lightning as pl
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from .network import PulseDetectionNet 
from .loss import PulseLoss
from .eval_metrics import PulseEval
import matplotlib.pyplot as plt

class LitModel(pl.LightningModule):
    def __init__(self, config, debug=False):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.debug = debug
        self.model = PulseDetectionNet(
            **self.config.network
        )
        self.criterion = PulseLoss(**self.config.loss)
        self.evaluation = PulseEval(peak_min_distance=self.config.loss.min_peak_distance)
        self.results = None

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        y_hat = self(x)
        loss, loss_components = self.criterion(y_hat, y)
        
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
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, value in loss_components.items():
            self.log(f'val_{name}', value)
        
        # _, count_error, distance_error, _ = peak_error(y_hat, y, peak_min_distance=self.config.loss.min_peak_distance, heights=[0.5])
        _, count_error, distance_error, _ = self.evaluation.peak_error(y_hat, y, heights=[0.5])
        self.log('val_count_error', count_error[0], prog_bar=True)
        self.log('val_distance_error', distance_error[0], prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        fnames = batch[2]
        y_hat = self(x)
        
        # x_cpu = x.detach().cpu().numpy()
        # y_cpu = y.detach().cpu().numpy()
        # y_hat_cpu = y_hat.detach().cpu().numpy()
        # from scipy import signal
        # b, a = signal.butter(2, 30/250, btype='low', analog=False)
        # nsig = 64-5
        # nbin = 10
        # filtered_data = signal.filtfilt(b, a, x_cpu[nsig,:,nbin])
        # plt.plot(np.diff(filtered_data,n=2))
        # plt.plot((y_cpu[nsig,:,0]-0.5)/100)
        
        loss, loss_components = self.criterion(y_hat, y)
        if self.debug:
            # peak_error(y_hat, y, peak_min_distance=self.config.loss.min_peak_distance, heights=[0.45], debug_fnames=batch[2])
            _, count_errors, all_distance_errors, signed_all_distance_errors = self.evaluation.peak_error(y_hat, y, heights=[0.45], debug_fnames=fnames)
            assert np.unique(fnames).shape[0] == 1, 'Multiple users in the batch'
            print(f"fname: {fnames[0]}, count_error: {count_errors[0]}, median_distance: {np.median(signed_all_distance_errors)}, median_abs_distance: {np.median(np.abs(signed_all_distance_errors))}")
            self.debug_metrics.append({
                'fname': fnames[0],
                'count_error': count_errors[0],
                'median_distance': np.median(signed_all_distance_errors),
                'median_abs_distance': np.median(np.abs(signed_all_distance_errors))
            })
            
        # heights, count_errors, distance_errors, all_distance_errors = peak_error(y_hat, y, peak_min_distance=self.config.loss.min_peak_distance)
        heights, count_errors, all_distance_errors, signed_all_distance_errors = self.evaluation.peak_error(y_hat, y)
        
        if self.thrs is None:
            self.thrs = heights
            self.distance_errs_at_thrs = [[] for _ in range(len(heights))]
            self.count_errs_at_thrs = [[] for _ in range(len(heights))]

        for id, (height, count_error, distance_errors) in enumerate(zip(heights, count_errors, all_distance_errors)):
            result_dict = {
                'count_error_{:.2f}'.format(height): count_error,
                'distance_error_{:.2f}'.format(height): np.median(distance_errors)
            }
            self.log_dict(result_dict, on_step=True, on_epoch=True)
            
            self.distance_errs_at_thrs[id].extend(all_distance_errors[id])
            self.count_errs_at_thrs[id].append(count_error)
        
        # id = np.argmin(np.abs(np.array(count_errors) - self.config.eval.count_err_thr))
        # print(id, heights[id], count_errors[id], distance_errors[id])
        # result_dict = {
        #     'test_loss': loss,
        #     'heights': heights,
        #     'test_count_error': count_errors,
        #     'test_distance_error': distance_errors
        # }
        # self.log_dict(result_dict, on_step=True, on_epoch=True, prog_bar=True)
        
        for name, value in loss_components.items():
            self.log(f'test_{name}', value)
            
        return result_dict
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        if self.config.scheduler.type == 'none':
            return {"optimizer": optimizer}
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.scheduler.T_max,
            T_mult=1,
            eta_min=self.config.scheduler.min_lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": 'epoch',
                "frequency": 1
            }
        }
    
    def on_test_start(self):
        self.thrs = None
        self.distance_errs_at_thrs = None
        self.count_errs_at_thrs = None
        # Initialize list to store debug metrics
        self.debug_metrics = []
    
    def on_test_epoch_end(self):
        for i in range(len(self.thrs)):
            self.distance_errs_at_thrs[i] = np.array(self.distance_errs_at_thrs[i])
            self.count_errs_at_thrs[i] = np.array(self.count_errs_at_thrs[i])
        
        distance_errs_at_thrs = np.array(self.distance_errs_at_thrs, dtype=object)
        count_errs_at_thrs = np.array(self.count_errs_at_thrs, dtype=object)
        thrs = np.array(self.thrs)
        
        self.results =  {
            'thrs': thrs,
            'distance_errs_at_thrs': distance_errs_at_thrs,
            'count_errs_at_thrs': count_errs_at_thrs
        }
    
    