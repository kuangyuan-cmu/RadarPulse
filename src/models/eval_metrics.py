    
import torch
import numpy as np
from scipy.signal import find_peaks
from typing import Dict
import matplotlib.pyplot as plt
from scipy import stats
import os

class PulseEval():
    def __init__(self, peak_min_distance: int = 250, site=None):
        self.peak_min_distance = peak_min_distance
        self.site = site
    
    def peak_detection(self, target, pred, peak_min_height=0.25):
        target_peaks_list = []
        pred_peaks_list = []
        pred_heights_list = []
        for i in range(target.shape[0]):
            peaks, properties = find_peaks(pred[i].squeeze(), height=peak_min_height, distance=self.peak_min_distance)
            pred_peaks_list.append(torch.tensor(peaks))
            pred_heights_list.append(properties['peak_heights'])
            target_peaks = np.where(target[i] == 1)[0]
            target_peaks_list.append(target_peaks)
        
        # match the predicted peaks to target peaks, if can not find a matched predicted peak within peak_min_distance, add a None
        matched_peaks_list = []
        # matched_distances_list = []
        signed_distances_list = []
        peak_heights_list = []
        for i, (pred_peaks, target_peaks) in enumerate(zip(pred_peaks_list, target_peaks_list)):
            matched_peaks = []
            matched_distances = []
            signed_distances = []
            peak_heights = []
            if len(pred_peaks) == 0:
                matched_peaks.extend([None] * len(target_peaks))
                matched_distances.extend([None] * len(target_peaks))
                signed_distances.extend([None] * len(target_peaks))
                peak_heights.extend([0] * len(target_peaks))
                matched_peaks_list.append(matched_peaks)
                # matched_distances_list.append(matched_distances)
                signed_distances_list.append(signed_distances)
                peak_heights_list.append(peak_heights)
                continue
            
            distances = np.expand_dims(pred_peaks, axis=1) - np.expand_dims(target_peaks, axis=0)
            abs_distances = np.abs(distances)
            min_distances = np.min(abs_distances, axis=0)
            min_idx = np.argmin(abs_distances, axis=0)
            
            for pi, min_distance in enumerate(min_distances):
                if min_distance < self.peak_min_distance // 2:
                    matched_peaks.append(pred_peaks[min_idx[pi]])
                    matched_distances.append(min_distance)
                    signed_distances.append(distances[min_idx[pi], pi])
                    peak_heights.append(pred_heights_list[i][min_idx[pi]])
                else:
                    matched_peaks.append(None)
                    matched_distances.append(None)
                    signed_distances.append(None)
                    peak_heights.append(0)
            matched_peaks_list.append(matched_peaks)
            # matched_distances_list.append(matched_distances)
            signed_distances_list.append(signed_distances)
            peak_heights_list.append(peak_heights)
        # return target_peaks_list, matched_peaks_list, matched_distances_list, signed_distances_list
        return target_peaks_list, matched_peaks_list, signed_distances_list, peak_heights_list # (Batch, NumTargetPeaks)
        # return target_peaks_list, matched_peaks_list, signed_distances_list
    
    def debug_plot(self, pred, target, target_peaks_list, matched_peaks_list, signed_distances_list, debug_fnames):
        fig_width = 4
        fig_height = 16
        fig = plt.figure(figsize=(10, 20))
        gs = fig.add_gridspec(fig_height, fig_width)
        axs = gs.subplots()

        for i in range(target.shape[0]):
            plt_x = i // fig_width
            plt_y = i % fig_width
            axs[plt_x, plt_y].plot(target[i].squeeze(), color='orange')
            # for peak in target_peaks_list[i]:
            #     axs[plt_x, plt_y].plot(peak, target[i, peak], 'go')
            axs[plt_x, plt_y].plot(pred[i].squeeze(), color='gray')
            for pi, peak in enumerate(matched_peaks_list[i]):
                if peak is not None:
                    color = 'red' if signed_distances_list[i][pi] > 0 else 'green'
                    axs[plt_x, plt_y].plot(peak, pred[i, peak], 'x', color=color)
                    axs[plt_x, plt_y].text(peak, pred[i, peak], str(np.abs(signed_distances_list[i][pi])), fontsize=11, color=color)
            
            axs[plt_x, plt_y].set_title(debug_fnames[i], fontsize=6)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.99, bottom=0.01, hspace=0.7)
        
        save_path= f'results/figures/{self.site}'
        os.makedirs(save_path, exist_ok=True)
        if save_path:
            save_name = f'{save_path}/{debug_fnames[0]}.png'
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            print(f'Saved figure to {save_name}')
        else:
            plt.show()
        
    # def peak_error_at_height(self, target, pred, peak_min_height: float = 0.5, debug_fnames=None):
    #     # target_peaks_list, matched_peaks_list, matched_distances_list, signed_distances_list = self.peak_detection(target, pred, peak_min_height)
    #     target_peaks_list, matched_peaks_list, signed_distances_list = self.peak_detection(target, pred, peak_min_height)
    #     if debug_fnames:
    #         self.debug_plot(pred, target, target_peaks_list, matched_peaks_list, signed_distances_list, debug_fnames)
        
    #     # aggregate the matched distances
    #     matched_distances_all = []
    #     signed_matched_distances_all = []
    #     peak_num = 0
    #     for signed_distances in signed_distances_list:
    #         peak_num += len(signed_distances)
    #         matched_distances_all.extend([abs(d) for d in signed_distances if d is not None])
    #         signed_matched_distances_all.extend([d for d in signed_distances if d is not None])
    #     error_count_rate = (peak_num - len(matched_distances_all)) / peak_num
        
    #     return error_count_rate, matched_distances_all, signed_matched_distances_all
    
    def peak_error(self, pred: torch.Tensor, target: torch.Tensor, heights: list = [], debug_fnames=None) -> Dict[str, float]:
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        if len(heights) == 0:
            heights = np.arange(0.1, 0.96, 0.01)
        
        avg_count_error_rates = []
        distance_errors = []
        signed_distance_errors = []
        
        target_peaks_list, matched_peaks_list, signed_distances_list, peak_heights_list = self.peak_detection(target_np, pred_np)
        for height in heights:
            matched_distances_all = []
            signed_matched_distances_all = []
            all_peak_num = 0
            for (target_peaks, matched_peaks, signed_distances, peak_heights) in zip(target_peaks_list, matched_peaks_list, signed_distances_list, peak_heights_list):
                valid_mask = np.array(peak_heights) > height
                valid_indices = np.where(valid_mask)[0]
                matched_distances_all.extend([abs(d) for i, d in enumerate(signed_distances) if i in valid_indices and d is not None])
                signed_matched_distances_all.extend([d for i, d in enumerate(signed_distances) if i in valid_indices and d is not None])
                all_peak_num += len(target_peaks)

            error_count_rate = (all_peak_num - len(matched_distances_all)) / all_peak_num
            avg_count_error_rates.append(error_count_rate)
            distance_errors.append(matched_distances_all)
            signed_distance_errors.append(signed_matched_distances_all)
        
        # for height in heights:
        #     error_count_rate, matched_distances, signed_matched_distances = self.peak_error_at_height(target_np, pred_np, peak_min_height=height, debug_fnames=debug_fnames)
        #     avg_count_error_rates.append(error_count_rate)
        #     distance_errors.append(matched_distances)
        #     signed_distance_errors.append(signed_matched_distances)
        return heights, avg_count_error_rates, distance_errors, signed_distance_errors # (NumHeight, NumMatchedPeaks)
    
    def pulse_transit_time(self, peaks_list_1, peaks_list_2, min_dist: int, max_dist: int, height_list_1=None, height_list_2=None):
        # inputsize: (Batch, NumTargetPeaks)
        ptts_list = []
        peaks_list = []
        total_height_list = []
        for i, (peaks_1, peaks_2) in enumerate(zip(peaks_list_1, peaks_list_2)):
            peaks_1_clean = np.array([float('inf') if x is None else x for x in peaks_1])
            peaks_2_clean = np.array([float('inf') if x is None else x for x in peaks_2])
            distances = np.expand_dims(peaks_1_clean, axis=1) - np.expand_dims(peaks_2_clean, axis=0)
            # distances = np.expand_dims(peaks_1, axis=1) - np.expand_dims(peaks_2, axis=0)
            valid_mask = (distances > min_dist) & (distances < max_dist)

            ptts = []
            heights = []
            for row_idx in range(valid_mask.shape[0]):
                valid_indices = np.nonzero(valid_mask[row_idx])[0]
                ptts.append(distances[row_idx, valid_indices[0]]
                        if len(valid_indices) > 0 else None)
                if height_list_1 is not None and height_list_2 is not None:
                    heights.append(height_list_1[i][row_idx] + height_list_2[i][valid_indices[0]] if len(valid_indices) > 0 else None)
            ptts_list.extend(ptts)
            peaks_list.extend(peaks_1)
            # print(heights)
            total_height_list.extend(heights)
            
        if height_list_1 is not None and height_list_2 is not None:
            return np.array(ptts_list), np.array(peaks_list), np.array(total_height_list)
        else:
            return np.array(ptts_list), np.array(peaks_list)
                    
    def remove_outliers(self, seq1, seq2):
        # detect the inds of outliers in two sequences using z-score
        z_scores1 = stats.zscore(np.array(seq1, dtype=np.float64))
        z_scores2 = stats.zscore(np.array(seq2, dtype=np.float64))
        
        # Find indices where either sequence has outliers (z-score > 3)
        outlier_mask = (np.abs(z_scores1) < 2) & (np.abs(z_scores2) < 2)
        
        # Remove outliers from both sequences
        seq1_clean = seq1[outlier_mask]
        seq2_clean = seq2[outlier_mask]
        
        return seq1_clean, seq2_clean
    
    def smooth_sequence(self, seq: np.array):
        return np.convolve(seq, np.ones(5)/5, mode='valid')
    
    def ptt_error(self, pred: torch.Tensor, target: torch.Tensor, ptt_queries: list, height_thrs: list = [], topn: int = 20):
        pred_np = pred.detach().cpu().numpy() # (Batch, NumSites, SeqLen)
        target_np = target.detach().cpu().numpy()
        assert pred_np.shape[1] == target_np.shape[1] == len(height_thrs)
        num_sites = pred_np.shape[1]
        
        target_peaks_sites = []
        pred_peaks_sites = []
        pred_heights_sites = []
        for i in range(num_sites):
            # target_peaks_list, matched_peaks_list, _ = self.peak_detection(target_np[:, i, :], pred_np[:, i, :], peak_min_height=height_thrs[i])
            target_peaks_list, matched_peaks_list, _, pred_heights_list = self.peak_detection(target_np[:, i, :], pred_np[:, i, :], peak_min_height=height_thrs[i])
            target_peaks_sites.append(target_peaks_list)
            pred_peaks_sites.append(matched_peaks_list) # have none
            pred_heights_sites.append(pred_heights_list)
        # size: (NumSites, Batch, NumTargetPeaks)
        
        
        metrics = []
        full_data = []
        for (site_ind_1, site_ind_2, min_dist, max_dist) in ptt_queries:
            gt_ptts, gt_peaks = self.pulse_transit_time(target_peaks_sites[site_ind_1], target_peaks_sites[site_ind_2], min_dist, max_dist)
            pred_ptts, pred_peaks, pred_heights = self.pulse_transit_time(pred_peaks_sites[site_ind_1], pred_peaks_sites[site_ind_2], min_dist, max_dist, pred_heights_sites[site_ind_1], pred_heights_sites[site_ind_2])
            # pred_ptts, pred_peaks = self.pulse_transit_time(pred_peaks_sites[site_ind_1], pred_peaks_sites[site_ind_2], min_dist, max_dist)
            assert len(gt_ptts) == len(pred_ptts) == len(gt_peaks) == len(pred_peaks)
            
            gt_ptts_valid_inds = np.where(gt_ptts != None)[0]
            gt_ptts_rates = gt_ptts_valid_inds.shape[0] / len(gt_ptts)
            pred_ptts_valid_inds = (np.where((pred_ptts != None) & (gt_ptts != None)))[0]
            pred_ptts_rates = pred_ptts_valid_inds.shape[0] / len(pred_ptts)
            
            pred_ptts_valid = pred_ptts[pred_ptts_valid_inds]
            gt_ptts_valid = gt_ptts[pred_ptts_valid_inds]
            pred_heights_valid = pred_heights[pred_ptts_valid_inds]
            # remove the outliers, smooth the two sequences
            gt_ptts_valid, pred_ptts_valid = self.remove_outliers(gt_ptts_valid, pred_ptts_valid)
            
            # pick the topn ind in pred_heights_valid            
            # topn_inds = np.argsort(pred_heights_valid)[-len(pred_heights_valid)//2:]
            # gt_ptts_valid = gt_ptts_valid[topn_inds]
            # pred_ptts_valid = pred_ptts_valid[topn_inds]
            # pred_heights_valid = pred_heights_valid[topn_inds]
            
            if len(gt_ptts_valid) > 0:
                gt_ptts_valid = self.smooth_sequence(gt_ptts_valid)
                pred_ptts_valid = self.smooth_sequence(pred_ptts_valid)
            
            pred_ptts_errs = np.abs(gt_ptts_valid - pred_ptts_valid)
            # plt.plot(gt_ptts[pred_ptts_valid_inds])
            # plt.plot(pred_ptts[pred_ptts_valid_inds])
            # plt.show()
            metrics.append({
                f'gt_ptt_rate': gt_ptts_rates,
                f'pred_ptt_rate': pred_ptts_rates,
                f'ptt_err': np.median(pred_ptts_errs),
            })
            full_data.append({
                'gt_ptt': gt_ptts_valid,
                'pred_ptt': pred_ptts_valid,
            })
        return metrics, full_data
            
            
            
    

# def peak_error_at_height(target, pred, peak_min_height: float = 0.5, peak_min_distance: int = 250, debug_fnames=None):
#     """
#     Calculate peak error between predicted and target signals for batched data.
    
#     Args:
#         target_peak_list (list): List of target peak locations for each batch item
#         pred (torch.Tensor): Predicted signal tensor of shape [batch_size, sequence_length]
#         peak_min_height (float): Minimum height threshold for peak detection
#         peak_min_distance (int): Minimum samples between peaks
    
#     Returns:
#         dict: Dictionary containing:
#             - count_error: Absolute difference in number of peaks
#             - position_error: Average absolute distance between matched peaks
#     """
#     # Process each sequence in batch
#     batch_size = target.shape[0]
    
#     pred_peaks_list = []
#     target_peaks_list = []
#     for i in range(batch_size):
#         peaks, _ = find_peaks(pred[i].squeeze(), height=peak_min_height, distance=peak_min_distance)
#         pred_peaks_list.append(torch.tensor(peaks))
#         target_peaks = np.where(target[i] == 1)[0]
#         target_peaks_list.append(target_peaks)
        
#     # Calculate errors for each batch item
#     count_error_rates = []
#     position_errors = []
    
#     if debug_fnames:
#         fig_width = 4
#         fig_height = 16
#         fig, axs = plt.subplots(fig_height, fig_width)
#     for i, (pred_peaks, target_peaks) in enumerate(zip(pred_peaks_list, target_peaks_list)):
#         count_error_rates.append(abs(len(pred_peaks) - len(target_peaks)) / len(target_peaks))
#         # if len(pred_peaks) == 0 or len(target_peaks) == 0:
#         #     continue
        
#         # Calculate raw distances (keeping sign)
#         raw_distances = np.expand_dims(pred_peaks, axis=1) - np.expand_dims(target_peaks, axis=0)
#         distances = np.abs(raw_distances)
#         min_distances_idx = np.argmin(distances, axis=1)
#         min_distances = np.array([raw_distances[i, min_distances_idx[i]] for i in range(len(pred_peaks))])

#         position_errors.extend(np.abs(min_distances))
#         if debug_fnames:
#             plt_x = i // fig_width
#             plt_y = i % fig_width
#             axs[plt_x, plt_y].plot(pred[i].squeeze())
#             axs[plt_x, plt_y].plot(target[i].squeeze())
#             for pi, peak in enumerate(pred_peaks):
#                 axs[plt_x, plt_y].plot(peak, pred[i, peak], 'rx')
#                 # mark the number of errors with different colors based on sign
#                 text_color = 'red' if min_distances[pi] > 0 else 'blue'
#                 axs[plt_x, plt_y].text(peak, pred[i, peak], str(abs(min_distances[pi])), 
#                                      fontsize=12, color=text_color)
                
#             for peak in target_peaks:
#                 axs[plt_x, plt_y].plot(peak, target[i, peak], 'go')
#             axs[plt_x, plt_y].set_title(debug_fnames[i])
            
#     if debug_fnames:
#         plt.subplots_adjust(left=0.05, right=0.95, top=0.99, bottom=0.01, hspace=0.7)
#         plt.show()

#     # Average errors across batch
#     avg_count_error_rate = sum(count_error_rates) / batch_size
#     avg_position_error = np.median(position_errors)
    
#     return avg_count_error_rate, avg_position_error, position_errors
    
    

# def peak_error(pred: torch.Tensor, target: torch.Tensor, heights: list = [], peak_min_distance: int = 250, debug_fnames=None) -> Dict[str, float]:
#     """
#     Calculate peak error between predicted and target signals for batched data.
    
#     Args:
#         pred (torch.Tensor): Predicted signal tensor of shape [batch_size, sequence_length]
#         target (torch.Tensor): Target tensor of shape [batch_size, sequence_length] with binary values
#         peak_min_height (float): Minimum height threshold for peak detection
#         peak_min_distance (int): Minimum samples between peaks
        
#     Returns:
#         dict: Dictionary containing averaged errors across batch:
#             - count_error: Average absolute difference in number of peaks
#             - position_error: Average absolute distance between matched peaks
#     """
#     batch_size = pred.shape[0]
        
#     # Convert pred to numpy for find_peaks processing
#     pred_np = pred.detach().cpu().numpy()
#     target_np = target.detach().cpu().numpy()
#     if len(heights) == 0:
#         heights = np.arange(0.1, 0.96, 0.01)
#     avg_count_error_rates = []
#     avg_position_errors = []
#     all_position_errors = []
#     for height in heights:
#         avg_count_error_rate, avg_position_error, postion_errors = peak_error_at_height(target_np, pred_np, 
#                                                                peak_min_height=height, peak_min_distance=peak_min_distance, debug_fnames=debug_fnames)
#         avg_count_error_rates.append(avg_count_error_rate)
#         avg_position_errors.append(avg_position_error)
#         all_position_errors.append(postion_errors)
    
#     if len(avg_count_error_rates) == 0:
#         return None, [float('inf')], [float('inf')], []
#     return heights, avg_count_error_rates, avg_position_errors, all_position_errors
    
    