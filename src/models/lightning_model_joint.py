import pytorch_lightning as pl
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from .network_joint import MultiSitePulseDetectionNet
from .loss import MultiSitePulseLoss
from .eval_metrics import peak_error, PulseEval
import matplotlib.pyplot as plt

class LitModel_joint(pl.LightningModule):
    def __init__(self, config_list, training_config, checkpoint_paths = None, debug=False):
        super().__init__()
        self.save_hyperparameters()
        self.config_list = config_list
        self.training_config = training_config
        self.sites_names = [config.data.position for config in config_list]
        self.num_sites = len(config_list)
        self.debug = debug
        
        self.network_configs = [config.network for config in config_list]
        self.loss_configs = [config.loss for config in config_list]
        self.model = MultiSitePulseDetectionNet(self.network_configs)
        self.criterion = MultiSitePulseLoss(self.loss_configs, names=self.sites_names)
        self.evaluation = PulseEval(peak_min_distance=self.loss_configs[0].min_peak_distance)
        
        if checkpoint_paths is not None:
            self.model.load_pretrained(checkpoint_paths)
            self.model.freeze_all_sites()
        self.results = None
        
        # (site_1, site_2, min_distance, max_distance)
        self.ptt_queries = [ 
            (1, 2, -130, -60),
            (0, 2, -105, -35)
        ]

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
        
        for i in range(self.num_sites):
            y_hat_site = y_hat[:, i, :, :]
            y_site = y[:, i, :, :]
            _, count_error, distance_error, _ = peak_error(y_hat_site, y_site, peak_min_distance=self.loss_configs[0].min_peak_distance, heights=[0.5])
            self.log(f'val_count_error_{self.sites_names[i]}', count_error[0], prog_bar=True)
            self.log(f'val_distance_error_{self.sites_names[i]}', distance_error[0], prog_bar=True)
            
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        names = batch[2]
        # print(np.unique(names))
        
        y_hat = self(x)
        loss, loss_components = self.criterion(y_hat, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for i in range(self.num_sites):
            y_hat_site = y_hat[:, i, :, :]
            y_site = y[:, i, :, :]
            # if i == 0:
            #     _, count_error, distance_error, _ = self.evaluation.peak_error(y_hat_site, y_site, heights=[0.6], debug_fnames=batch[2])
            
            heights, count_errors, distance_errors, all_distance_errors = self.evaluation.peak_error(y_hat_site, y_site)
            if len(self.distance_errs_at_thrs[self.sites_names[i]]) == 0:
                self.thrs = heights
                self.distance_errs_at_thrs[self.sites_names[i]] = [[] for _ in range(len(heights))]
                self.count_errs_at_thrs[self.sites_names[i]] = [[] for _ in range(len(heights))]
            for id, (height, count_error, distance_error) in enumerate(zip(heights, count_errors, distance_errors)):
                self.distance_errs_at_thrs[self.sites_names[i]][id].extend(all_distance_errors[id])
                self.count_errs_at_thrs[self.sites_names[i]][id].append(count_error)
                result_dict = {
                    f'{self.sites_names[i]}_count_error_{height:.2f}': count_error,
                    f'{self.sites_names[i]}_distance_error_{height:.2f}': distance_error
                    
                }
                self.log_dict(result_dict, on_step=True, on_epoch=True)
            
        if np.unique(names).shape[0] == 1:
            bname = names[0]
            print(bname)
            
        ptt_metrics, ptt_samples = self.evaluation.ptt_error(y_hat, y, ptt_queries=self.ptt_queries, height_thrs=[0.68, 0.78, 0.66])
        for ptt_metric in ptt_metrics:
            for name, value in ptt_metric.items():
                self.log(name, value)
        
        for i, ptt_sample in enumerate(ptt_samples):
            for name, value in ptt_sample.items():
                print(name, np.median(value))
                self.ptt_samples[i][name].extend(value)
        
        for name, value in loss_components.items():
            self.log(f'test_{name}', value)
            
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config.training.learning_rate,
            weight_decay=self.training_config.training.weight_decay
        )
        if self.training_config.scheduler.type == 'none':
            return {"optimizer": optimizer}
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.training_config.scheduler.T_max,
            T_mult=1,
            eta_min=self.training_config.scheduler.min_lr
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
        self.distance_errs_at_thrs = {f'{site}': [] for site in self.sites_names}
        self.count_errs_at_thrs = {f'{site}': [] for site in self.sites_names}
        self.ptt_samples = [
            {'gt_ptt': [], 'pred_ptt': []} for _ in range(len(self.ptt_queries))
        ]
        
    def on_test_epoch_end(self):
        thrs = np.array(self.thrs)
        self.results = {
            'thrs': thrs,
        }
        for site in self.sites_names:
            for i in range(len(self.thrs)):
                self.distance_errs_at_thrs[site][i] = np.array(self.distance_errs_at_thrs[site][i])
                self.count_errs_at_thrs[site][i] = np.array(self.count_errs_at_thrs[site][i])
            distance_errs_at_thrs = np.array(self.distance_errs_at_thrs[site], dtype=object)
            count_errs_at_thrs = np.array(self.count_errs_at_thrs[site], dtype=object)
            self.results[f'{site}_distance_errs_at_thrs'] = distance_errs_at_thrs
            self.results[f'{site}_count_errs_at_thrs'] = count_errs_at_thrs
        
        self.results['ptt_samples'] = self.ptt_samples
        self.results['ptt_queries'] = self.ptt_queries
    
    