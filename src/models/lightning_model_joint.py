import pytorch_lightning as pl
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from .network_joint import MultiSitePulseDetectionNet
from .loss import MultiSitePulseLoss, DirectPTTRegressionLoss
from .eval_metrics import PulseEval
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os
import json

class LitModel_joint(pl.LightningModule):
    def __init__(self, config_list, training_config, checkpoint_paths=None, debug=False, enable_fusion=True):
        super().__init__()
        self.save_hyperparameters()
        self.config_list = config_list
        self.training_config = training_config
        self.sites_names = [config.data.position for config in config_list]
        self.num_sites = len(config_list)
        self.debug = debug
        
        self.network_configs = [config.network for config in config_list]
        self.loss_configs = [config.loss for config in config_list]
        self.pairs = [(1,2), (1,3), (2,3), (0,1), (0,2)]
        self.model = MultiSitePulseDetectionNet(self.network_configs, enable_fusion=enable_fusion, direct_ptt=True, pairs=self.pairs)
        # self.criterion = MultiSitePulseLoss(self.loss_configs, names=self.sites_names)
        self.criterion = DirectPTTRegressionLoss(pairs=self.pairs)
        self.evaluation = PulseEval(peak_min_distance=self.loss_configs[0].min_peak_distance)
        
        if checkpoint_paths is not None:
            self.model.load_pretrained(checkpoint_paths)
            self.model.freeze_all_sites()
        self.results = None
        self.username = None
        # (site_1, site_2, min_distance, max_distance)
        self.ptt_queries = [ 
            (1, 2, -95, -40),
            (1, 3, -50, 10),
            (2, 3, 10, 80),

            # (1, 2, -105, -30),
            # (1, 3, -45, 15),
            # (2, 3, 20, 90),

            (0, 1, 5, 65),
            # (0, 1, 0, 100),
            (0, 2, -60, -5),
            # (0, 2, -60, 20),
            (0, 3, 0, 45),
            # (0, 3, -10, 80),

            # for david and bill
            # (0, 1, 5, 100),
            # (0, 2, -75, 20),
            # (0, 3, -10, 100)
        ]
        self.height_thrs = [0.35, 0.55, 0.25, 0.5]
        # self.height_thrs = [0.55, 0.8, 0.65, 0.65]
        
        # self.height_thrs = [0.25, 0.5, 0.5, 0.5]
        # self.height_thrs = [0.6, 0.80, 0.68, 0.70]
        # self.height_thrs = [0.25]*4

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
        
        # for i in range(self.num_sites):
        #     y_hat_site = y_hat[:, i, :, :]
        #     y_site = y[:, i, :, :]
        #     # _, count_error, distance_error, _ = peak_error(y_hat_site, y_site, peak_min_distance=self.loss_configs[0].min_peak_distance, heights=[0.5])
        #     _, count_error, distance_errors, signed_distance_errors = self.evaluation.peak_error(y_hat_site, y_site, heights=[0.5])
        #     self.log(f'val_count_error_{self.sites_names[i]}', count_error[0], prog_bar=True)
        #     self.log(f'val_distance_error_{self.sites_names[i]}', np.median(distance_errors[0]), prog_bar=True)
            
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        names = batch[2]
        y_hat = self(x)
        loss, loss_components = self.criterion(y_hat, y)
        
        if np.unique(names).shape[0] == 1:
            bname = names[0]
            print(bname)
            if self.username is None:
                self.username = bname.split('_')[-2]
                for i in range(len(self.ptt_queries)):
                    os.makedirs(f'results/ptt_figures/{self.username}_{self.sites_names[self.ptt_queries[i][0]]}_{self.sites_names[self.ptt_queries[i][1]]}', exist_ok=True)
        
        # log loss components
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, value in loss_components.items():
            self.log(f'test_{name}', value)
            
        # log metrics of each site
        for i in range(self.num_sites):
            y_hat_site = y_hat[:, i, :, :]
            y_site = y[:, i, :, :]
            heights, count_errors, all_distance_errors, signed_all_distance_errors = self.evaluation.peak_error(y_hat_site, y_site)
            if len(self.distance_errs_at_thrs[self.sites_names[i]]) == 0:
                self.thrs = heights
                self.distance_errs_at_thrs[self.sites_names[i]] = [[] for _ in range(len(heights))]
                self.count_errs_at_thrs[self.sites_names[i]] = [[] for _ in range(len(heights))]
            for id, (height, count_error, distance_errors) in enumerate(zip(heights, count_errors, all_distance_errors)):
                self.distance_errs_at_thrs[self.sites_names[i]][id].extend(distance_errors)
                self.count_errs_at_thrs[self.sites_names[i]][id].append(count_error)
                result_dict = {
                    f'{self.sites_names[i]}_count_error_{height:.2f}': count_error,
                    f'{self.sites_names[i]}_distance_error_{height:.2f}': np.median(distance_errors)
                }
                self.log_dict(result_dict, on_step=True, on_epoch=True)
            
        
        # log ptt metrics
        ptt_metrics, ptt_samples = self.evaluation.ptt_error(y_hat, y, ptt_queries=self.ptt_queries, height_thrs=self.height_thrs)
        # for ptt_metric in ptt_metrics:
        #     for name, value in ptt_metric.items():
        #         self.log(name, value)
        for i in range(len(self.ptt_queries)):
            self.ptt_detect_rates[i]['gt_ptt'].append(ptt_metrics[i]['gt_ptt_rate'])
            self.ptt_detect_rates[i]['pred_ptt'].append(ptt_metrics[i]['pred_ptt_rate'])
            
            ptt_gt = ptt_samples[i]['gt_ptt']
            ptt_pred = ptt_samples[i]['pred_ptt']
            self.median_ptt_batch[i]['gt_ptt'].append(np.median(ptt_gt))
            self.median_ptt_batch[i]['pred_ptt'].append(np.median(ptt_pred))
            self.ptt_samples[i]['gt_ptt'].extend(ptt_gt)
            self.ptt_samples[i]['pred_ptt'].extend(ptt_pred)
            
            site1 = self.ptt_queries[i][0]
            site2 = self.ptt_queries[i][1]
            _, count_error_site1, _, signed_distance_error_site1 = self.evaluation.peak_error(y_hat[:, site1, :, :], y[:, site1, :, :], heights=[self.height_thrs[site1]])
            _, count_error_site2, _, signed_distance_error_site2 = self.evaluation.peak_error(y_hat[:, site2, :, :], y[:, site2, :, :], heights=[self.height_thrs[site2]])
            cnt_err_site1, bias_site1, err_site1 = count_error_site1[0], np.median(signed_distance_error_site1[0]), np.median(np.abs(signed_distance_error_site1[0]))
            cnt_err_site2, bias_site2, err_site2 = count_error_site2[0], np.median(signed_distance_error_site2[0]), np.median(np.abs(signed_distance_error_site2[0]))

            gt_ptt = ptt_samples[i]['gt_ptt']
            pred_ptt = ptt_samples[i]['pred_ptt']
            
            # plot these two and save the figure
            plt.plot(gt_ptt, label='gt_ptt')
            plt.plot(pred_ptt, label='pred_ptt')
            plt.ylim(self.ptt_queries[i][2], self.ptt_queries[i][3])
            plt.legend()
            
            plt.text(0.02, 0.98, 
                    f'Median PTT: {np.median(gt_ptt):.1f} vs {np.median(pred_ptt):.1f}\n' + 
                    f'{self.sites_names[site1]}: Count Error: {cnt_err_site1:.2f}, Bias: {bias_site1:.2f}, Error: {err_site1:.2f}\n' +
                    f'{self.sites_names[site2]}: Count Error: {cnt_err_site2:.2f}, Bias: {bias_site2:.2f}, Error: {err_site2:.2f}',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.savefig(f'results/ptt_figures/{self.username}_{self.sites_names[self.ptt_queries[i][0]]}_{self.sites_names[self.ptt_queries[i][1]]}/{bname}.png')
            plt.close()
        
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
        self.ptt_detect_rates = [
            {'gt_ptt': [], 'pred_ptt': []} for _ in range(len(self.ptt_queries))
        ]
        self.median_ptt_batch = [
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
        
        # self.ptt_corr = []
        # for i in range(len(self.ptt_queries)):
        #     self.ptt_corr.append(np.corrcoef(self.median_ptt_batch[i]['gt_ptt'], self.median_ptt_batch[i]['pred_ptt']))
        
        # self.ptt_errs = []
            
        # load bp ground truth from results/gt_bp/username.csv
        bp_gt = pd.read_csv(f'results/gt_bp/{self.username}.csv', sep='\t')
        sys = bp_gt['sys']
        dia = bp_gt['dia']
        eval_metrics = {}
        for i in range(len(self.ptt_queries)):
            # generate a cdf plot of the ptt_errs
            savepath = f'results/ptt_figures/{self.username}_{self.sites_names[self.ptt_queries[i][0]]}_{self.sites_names[self.ptt_queries[i][1]]}'
            ptt_errs = np.abs(np.array(self.ptt_samples[i]['gt_ptt']) - np.array(self.ptt_samples[i]['pred_ptt']))
            plt.plot(np.sort(ptt_errs)*2, np.arange(len(ptt_errs)) / len(ptt_errs))
            plt.xlabel('PPT Error (ms)')
            plt.ylabel('CDF')
            plt.title(f'PPT Error CDF for {self.sites_names[self.ptt_queries[i][0]]} and {self.sites_names[self.ptt_queries[i][1]]}')
            plt.savefig(f'{savepath}/ptt_errs_cdf.png')
            plt.close()
            
            # calculate the correlation between median_ptt_batch and bp_gt
            corr_gt_sys = np.corrcoef(self.median_ptt_batch[i]['gt_ptt'], sys)[0,1]
            corr_gt_dia = np.corrcoef(self.median_ptt_batch[i]['gt_ptt'], dia)[0,1]
            corr_pred_sys = np.corrcoef(self.median_ptt_batch[i]['pred_ptt'], sys)[0,1]
            corr_pred_dia = np.corrcoef(self.median_ptt_batch[i]['pred_ptt'], dia)[0,1]
            corr_gt_pred = np.corrcoef(self.median_ptt_batch[i]['gt_ptt'], self.median_ptt_batch[i]['pred_ptt'])[0,1]
            print(f'corr_gt_sys: {corr_gt_sys}, corr_gt_dia: {corr_gt_dia}, corr_pred_sys: {corr_pred_sys}, corr_pred_dia: {corr_pred_dia}, corr_gt_pred: {corr_gt_pred}')
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            # Plot PTT data on left y-axis
            ax1.plot(self.median_ptt_batch[i]['gt_ptt'], label='gt_ptt', color='blue')
            ax1.plot(self.median_ptt_batch[i]['pred_ptt'], label='pred_ptt', color='orange')
            ax1.set_ylabel('PTT')

            # Plot BP data on right y-axis  
            ax2.plot(sys, label='sys', color='green')
            ax2.plot(dia, label='dia', color='red')
            ax2.set_ylabel('Blood Pressure (mmHg)')

            # Add correlation values as text
            plt.text(0.02, 0.95, f'GT-Sys corr: {corr_gt_sys:.3f}\nGT-Dia corr: {corr_gt_dia:.3f}\nPred-Sys corr: {corr_pred_sys:.3f}\nPred-Dia corr: {corr_pred_dia:.3f}\nGT-Pred corr: {corr_gt_pred:.3f}', 
                    transform=ax1.transAxes, verticalalignment='top')

            # Add legends for both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            # plt.legend()
            plt.savefig(f'{savepath}/bp_corr.png')
            plt.close()
            
            eval_metrics[f'{self.sites_names[self.ptt_queries[i][0]]}_{self.sites_names[self.ptt_queries[i][1]]}'] = {
                'GT-Sys corr': corr_gt_sys,
                'GT-Dia corr': corr_gt_dia,
                'Pred-Sys corr': corr_pred_sys,
                'Pred-Dia corr': corr_pred_dia,
                'GT-Pred corr': corr_gt_pred,
                'GT-PPT detect rate': np.mean(self.ptt_detect_rates[i]['gt_ptt']),
                'Pred-PPT detect rate': np.mean(self.ptt_detect_rates[i]['pred_ptt']),
                'PPT error': np.median(ptt_errs)
            }

        with open(f'results/ptt_figures/{self.username}_eval_metrics.json', 'w') as f:
            json.dump(eval_metrics, f)

        data = {
            'sys': sys,
            'dia': dia,
        }
        # Add PTT data for each site pair
        for i in range(len(self.ptt_queries)):
            site1 = self.sites_names[self.ptt_queries[i][0]]
            site2 = self.sites_names[self.ptt_queries[i][1]]
            column_name = f'ptt_{site1}_{site2}'
            data[f'{column_name}_gt'] = self.median_ptt_batch[i]['gt_ptt']
            data[f'{column_name}_pred'] = self.median_ptt_batch[i]['pred_ptt']
        
        # Create and save DataFrame
        df = pd.DataFrame(data)
        df.to_csv(f'results/bp_eval/{self.username}_ptt_bp_data.csv', index=False)
        
        
