# import torch
# import numpy as np
# from scipy.signal import find_peaks

# def peak_error(pred: torch.Tensor, target: torch.Tensor, peak_min_height: float = 0.1, peak_min_distance: int = 250) -> dict:
#     """
#     Calculate peak error between predicted and target signals.
#     Target tensor contains binary values (0 and 1) where 1s indicate peak locations.
    
#     Args:
#         pred (torch.Tensor): Predicted signal tensor
#         target (torch.Tensor): Target tensor with binary values (0 and 1)
#         peak_min_height (float): Minimum height threshold for peak detection
#         peak_min_distance (int): Minimum samples between peaks
        
#     Returns:
#         dict: Dictionary containing:
#             - count_error: Absolute difference in number of peaks
#             - position_error: Average absolute distance between matched peaks
#     """
#     # Get target peak locations (indices where value is 1)
#     target_peaks = torch.where(target.squeeze() == 1)[0]
#     print(target_peaks.shape)
#     # Find peaks in predicted signal
#     pred_np = pred.detach().cpu().numpy().squeeze()
#     pred_peaks = torch.tensor(find_peaks(pred_np, height=peak_min_height, 
#                                        distance=peak_min_distance)[0],
#                             device=pred.device)
    
#     # Calculate count error
#     count_error = abs(len(pred_peaks) - len(target_peaks))
    
#     # Calculate position error
#     if len(pred_peaks) == 0 or len(target_peaks) == 0:
#         position_error = float('inf')  # or another appropriate value
#     else:
#         # Calculate distances between each pred peak and all target peaks
#         distances = torch.abs(pred_peaks.unsqueeze(1) - target_peaks.unsqueeze(0))
#         # For each predicted peak, find distance to closest target peak
#         min_distances = torch.min(distances, dim=1)[0]
#         position_error = torch.mean(min_distances).item()
    
#     return {
#         'count_error': count_error,
#         'position_error': position_error
#     }
    
    
import torch
import numpy as np
from scipy.signal import find_peaks
from typing import Dict

def peak_error_at_height(target_peaks_list, pred, peak_min_height: float = 0.5, peak_min_distance: int = 250) -> Dict[str, float]:
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
    batch_size = len(target_peaks_list)
    pred_peaks_list = []
    for i in range(batch_size):
        peaks, _ = find_peaks(pred[i].squeeze(), height=peak_min_height, distance=peak_min_distance)
        pred_peaks_list.append(torch.tensor(peaks, device=target_peaks_list[0].device))
    
    # Calculate errors for each batch item
    count_error_rates = []
    position_errors = []
    
    for i, (pred_peaks, target_peaks) in enumerate(zip(pred_peaks_list, target_peaks_list)):
        # Count error
        count_error_rates.append(abs(len(pred_peaks) - len(target_peaks)) / len(target_peaks))
        
        if len(pred_peaks) == 0 or len(target_peaks) == 0:
            continue
        distances = torch.abs(pred_peaks.unsqueeze(1) - target_peaks.unsqueeze(0))
        min_distances = torch.min(distances, dim=1)[0].float()
        mean_distance_error = torch.median(min_distances).item()
        position_errors.append(mean_distance_error)

    # Average errors across batch
    avg_count_error_rate = sum(count_error_rates) / batch_size
    if len(position_errors) == 0:
        avg_position_error = float('inf')
    else:
        avg_position_error = np.mean(position_errors)
    
    return avg_count_error_rate, avg_position_error
    
    

def peak_error(pred: torch.Tensor, target: torch.Tensor, heights: list = [], peak_min_distance: int = 250) -> Dict[str, float]:
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
    device = pred.device
    
    # Process target peaks (indices where value is 1) for each batch
    target_peaks_list = []
    for i in range(batch_size):
        target_peaks_list.append(torch.where(target[i] == 1)[0])
        
    # Convert pred to numpy for find_peaks processing
    pred_np = pred.detach().cpu().numpy()
    if len(heights) == 0:
        heights = np.arange(0.1, 0.96, 0.05)
    count_error_rates = []
    position_errors = []
    
    for height in heights:
        count_error_rate, position_error = peak_error_at_height(target_peaks_list, pred_np, 
                                                               peak_min_height=height, peak_min_distance=peak_min_distance)
        if position_error == float('inf'):
            break
        count_error_rates.append(count_error_rate)
        position_errors.append(position_error)
    
    return heights[:len(count_error_rates)], count_error_rates, position_errors
    
    