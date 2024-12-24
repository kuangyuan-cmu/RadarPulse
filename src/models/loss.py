import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class PulseLoss(nn.Module):
    def __init__(self, 
                 seq_len: int,
                 sigma: float = 2.0,
                 min_peak_distance: int = 0.5*500,  # minimum allowed distance between peaks
                 max_peak_distance: int = 1.2*500,  # maximum allowed distance between peaks
                 distance_weight: float = 0.1,
                 count_weight: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.sigma = sigma
        self.min_peak_distance = min_peak_distance
        self.max_peak_distance = max_peak_distance
        self.distance_weight = distance_weight
        self.count_weight = count_weight
        
    def generate_gaussian_target(self, peak_locations, seq_len):
        device = peak_locations.device
        time_idx = torch.arange(seq_len, device=device).float()
        gaussian_target = torch.zeros_like(peak_locations, dtype=torch.float32)
        
        peak_indices = torch.where(peak_locations > 0)
        if peak_indices[0].shape[0] > 0:
            for b, t in zip(peak_indices[0], peak_indices[1]):
                gaussian = torch.exp(-0.5 * ((time_idx - t.float()) / self.sigma) ** 2)
                gaussian_target[b, :, 0] += gaussian.squeeze()
                
        return torch.clamp(gaussian_target, 0, 1)

    # def peak_distance_loss(self, pred):
    #     """Calculate loss based on distances between predicted peaks"""
    #     batch_size = pred.shape[0]
    #     device = pred.device
    #     total_distance_loss = torch.tensor(0.0, device=device)
        
    #     # Soft peak detection
    #     threshold = 0.5
    #     kernel_size = 3
        
    #     for b in range(batch_size):
    #         # Find local maxima
    #         x = pred[b, :, 0]
            
    #         # Add dimensions for max_pool1d
    #         x_expanded = x.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len)
            
    #         # Use max_pool1d to find local maxima
    #         # padding='same' maintains input size
    #         local_max_vals = F.max_pool1d(x_expanded, kernel_size=kernel_size, 
    #                                     stride=1, padding=kernel_size//2)
            
    #         # Squeeze back to original dimensions
    #         local_max_vals = local_max_vals.squeeze()
            
    #         # Find peaks
    #         local_max = (x == local_max_vals) & (x > threshold)
            
    #         # Get peak positions
    #         peak_positions = torch.where(local_max)[0]
            
    #         if len(peak_positions) > 1:
    #             # Calculate distances between consecutive peaks
    #             distances = peak_positions[1:] - peak_positions[:-1]
                
    #             # Penalty for distances outside acceptable range
    #             distance_penalty = torch.where(
    #                 distances < self.min_peak_distance,
    #                 self.min_peak_distance - distances,
    #                 torch.where(
    #                     distances > self.max_peak_distance,
    #                     distances - self.max_peak_distance,
    #                     torch.zeros_like(distances)
    #                 )
    #             )
                
    #             total_distance_loss += distance_penalty.sum()
        
    #     return total_distance_loss / batch_size

    # def peak_count_loss(self, pred, target):
    #     """Calculate loss based on difference in peak counts"""
    #     batch_size = pred.shape[0]
    #     device = pred.device
        
    #     # Parameters for peak detection
    #     threshold = 0.5
    #     kernel_size = 3
        
    #     # Initialize tensor for predicted peak counts
    #     pred_peak_counts = torch.zeros(batch_size, device=device)
        
    #     for b in range(batch_size):
    #         # Get prediction for this batch
    #         x = pred[b, :, 0]
            
    #         # Add dimensions for max_pool1d
    #         x_expanded = x.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len)
            
    #         # Find local maxima using max_pool1d
    #         local_max_vals = F.max_pool1d(x_expanded, kernel_size=kernel_size, 
    #                                     stride=1, padding=kernel_size//2)
            
    #         # Squeeze back to original dimensions
    #         local_max_vals = local_max_vals.squeeze()
            
    #         # A point is a peak if it's both:
    #         # 1. A local maximum (equal to the max in its neighborhood)
    #         # 2. Above threshold
    #         peaks = (x == local_max_vals) & (x > threshold)
            
    #         # Count peaks for this batch
    #         pred_peak_counts[b] = peaks.sum()
        
    #     # Calculate target peak counts
    #     target_peak_counts = target.sum(dim=1).squeeze(-1)
        
    #     # Calculate absolute difference in counts
    #     count_diff = torch.abs(pred_peak_counts - target_peak_counts)
        
        return count_diff.mean()
    
    def debug(self, preds, targets, names):
        # plot all of the preds and targets for a single batch
        for i in range(preds.shape[0]):
            plt.plot(preds[i, :, 0].detach().cpu().numpy())
            plt.plot(targets[i, :, 0].detach().cpu().numpy())
            plt.title(names[i])
            plt.show()
            
    def forward(self, pred, target, name=None):
        # Generate Gaussian target
        gaussian_target = self.generate_gaussian_target(target, self.seq_len)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy(pred, gaussian_target)
        
        if name is not None:
            self.debug(pred, gaussian_target, name)
        # Peak count loss
        # count_loss = self.count_weight * self.peak_count_loss(pred, target)
        # total_loss = bce_loss + count_loss
        
        # Peak distance loss
        # distance_loss = self.distance_weight * self.peak_distance_loss(pred)
        # total_loss = bce_loss + count_loss + distance_loss
        total_loss = bce_loss
        return total_loss, {
            'bce_loss': bce_loss.item(),
            # 'count_loss': count_loss.item(),
            # 'distance_loss': distance_loss.item()
        }

class MultiSitePulseLoss(nn.Module):
    def __init__(self, config_list, weights=None, names=['head', 'heart', 'wrist']):
        super().__init__()
        self.losses = nn.ModuleList([
            PulseLoss(**config) for config in config_list
        ])
        if weights is None:
            weights = [1.0] * len(config_list)
        self.weights = torch.tensor(weights)
        self.names = names
    
    def forward(self, preds, targets):
        total_loss = torch.tensor(0.0, device=preds[0].device)
        loss_components = {}
        for i, (loss, name) in enumerate(zip(self.losses, self.names)):
            pred, target = preds[:, i, :, :], targets[:, i, :, :]
            loss_value, components = loss(pred, target)
            total_loss += self.weights[i] * loss_value
            for key, value in components.items():
                loss_components[f'{name}_{key}'] = value
        return total_loss, loss_components