    
import torch
import numpy as np
from scipy.signal import find_peaks
from typing import Dict
import matplotlib.pyplot as plt

def peak_error_at_height(target, pred, peak_min_height: float = 0.5, peak_min_distance: int = 250, debug_fnames=None) -> Dict[str, float]:
    """
    Calculate peak error between predicted and target signals for batched data.
    
    Args:
        target_peak_list (list): List of target peak locations for each batch item
        pred (torch.Tensor): Predicted signal tensor of shape [batch_size, sequence_length]
        peak_min_height (float): Minimum height threshold for peak detection
        peak_min_distance (int): Minimum samples between peaks
    
    Returns:
        dict: Dictionary containing:
            - count_error: Absolute difference in number of peaks
            - position_error: Average absolute distance between matched peaks
    """
    # Process each sequence in batch
    batch_size = target.shape[0]
    
    pred_peaks_list = []
    target_peaks_list = []
    for i in range(batch_size):
        peaks, _ = find_peaks(pred[i].squeeze(), height=peak_min_height, distance=peak_min_distance)
        pred_peaks_list.append(torch.tensor(peaks))
        target_peaks = np.where(target[i] == 1)[0]
        target_peaks_list.append(target_peaks)
        
    # Calculate errors for each batch item
    count_error_rates = []
    position_errors = []
    
    
    if debug_fnames:
        fig_width = 4
        fig_height = 16
        fig, axs = plt.subplots(fig_height, fig_width)
    for i, (pred_peaks, target_peaks) in enumerate(zip(pred_peaks_list, target_peaks_list)):
        # Count error
        count_error_rates.append(abs(len(pred_peaks) - len(target_peaks)) / len(target_peaks))
        
        # if len(pred_peaks) == 0 or len(target_peaks) == 0:
        #     continue
        
        distances = np.abs(np.expand_dims(pred_peaks, axis=1) - np.expand_dims(target_peaks, axis=0))
        min_distances = np.min(distances, axis=1)

        position_errors.extend(min_distances)
        if debug_fnames:
            plt_x = i // fig_width
            plt_y = i % fig_width
            axs[plt_x, plt_y].plot(pred[i].squeeze())
            axs[plt_x, plt_y].plot(target[i].squeeze())
            for pi, peak in enumerate(pred_peaks):
                axs[plt_x, plt_y].plot(peak, pred[i, peak], 'rx')
                # mark the number of errors
                axs[plt_x, plt_y].text(peak, pred[i, peak], str(min_distances[pi]), fontsize=12, color='red')
                
            for peak in target_peaks:
                axs[plt_x, plt_y].plot(peak, target[i, peak], 'go')
            axs[plt_x, plt_y].set_title(debug_fnames[i])
            
    if debug_fnames:
        plt.subplots_adjust(left=0.05, right=0.95, top=0.99, bottom=0.01, hspace=0.7)
        plt.show()

    # Average errors across batch
    avg_count_error_rate = sum(count_error_rates) / batch_size
    avg_position_error = np.median(position_errors)
    
    return avg_count_error_rate, avg_position_error, position_errors
    
    

def peak_error(pred: torch.Tensor, target: torch.Tensor, heights: list = [], peak_min_distance: int = 250, debug_fnames=None) -> Dict[str, float]:
    """
    Calculate peak error between predicted and target signals for batched data.
    
    Args:
        pred (torch.Tensor): Predicted signal tensor of shape [batch_size, sequence_length]
        target (torch.Tensor): Target tensor of shape [batch_size, sequence_length] with binary values
        peak_min_height (float): Minimum height threshold for peak detection
        peak_min_distance (int): Minimum samples between peaks
        
    Returns:
        dict: Dictionary containing averaged errors across batch:
            - count_error: Average absolute difference in number of peaks
            - position_error: Average absolute distance between matched peaks
    """
    batch_size = pred.shape[0]
        
    # Process target peaks (indices where value is 1) for each batch
    # target_peaks_list = []
    # for i in range(batch_size):
    #     target_peaks_list.append(torch.where(target[i] == 1)[0])
        
    # Convert pred to numpy for find_peaks processing
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    if len(heights) == 0:
        heights = np.arange(0.1, 0.96, 0.01)
    avg_count_error_rates = []
    avg_position_errors = []
    all_position_errors = []
    for height in heights:
        avg_count_error_rate, avg_position_error, postion_errors = peak_error_at_height(target_np, pred_np, 
                                                               peak_min_height=height, peak_min_distance=peak_min_distance, debug_fnames=debug_fnames)
        avg_count_error_rates.append(avg_count_error_rate)
        avg_position_errors.append(avg_position_error)
        all_position_errors.append(postion_errors)
    
    if len(avg_count_error_rates) == 0:
        return None, [float('inf')], [float('inf')], []
    return heights, avg_count_error_rates, avg_position_errors, all_position_errors
    
    