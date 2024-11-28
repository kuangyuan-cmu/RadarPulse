import torch
import torch.nn as nn
import torch.nn.functional as F


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
                gaussian = torch.exp(-0.5 * ((time_idx - t) / self.sigma) ** 2)
                gaussian_target[b, :, 0] += gaussian
                
        return torch.clamp(gaussian_target, 0, 1)

    def peak_distance_loss(self, pred):
        """Calculate loss based on distances between predicted peaks"""
        batch_size = pred.shape[0]
        device = pred.device
        total_distance_loss = torch.tensor(0.0, device=device)
        
        # Soft peak detection
        # Use local maxima above threshold as peaks
        threshold = 0.5
        kernel_size = 3
        pad = kernel_size // 2
        
        for b in range(batch_size):
            # Find local maxima
            x = pred[b, :, 0]
            padded = F.pad(x, (pad, pad), mode='reflect')
            windows = padded.unfold(0, kernel_size, 1)
            local_max = (x == windows.max(1)[0]) & (x > threshold)
            
            # Get peak positions
            peak_positions = torch.where(local_max)[0]
            
            if len(peak_positions) > 1:
                # Calculate distances between consecutive peaks
                distances = peak_positions[1:] - peak_positions[:-1]
                
                # Penalty for distances outside acceptable range
                distance_penalty = torch.where(
                    distances < self.min_peak_distance,
                    self.min_peak_distance - distances,
                    torch.where(
                        distances > self.max_peak_distance,
                        distances - self.max_peak_distance,
                        torch.zeros_like(distances)
                    )
                )
                
                total_distance_loss += distance_penalty.sum()
        
        return total_distance_loss / batch_size

    def forward(self, pred, target):
        # Generate Gaussian target
        gaussian_target = self.generate_gaussian_target(target, self.seq_len)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy(pred, gaussian_target)
        
        # Peak count loss
        pred_peaks = (pred > 0.5).float()
        count_diff = torch.abs(pred_peaks.sum(dim=1) - target.sum(dim=1))
        count_loss = self.count_weight * count_diff.mean()
        
        # Peak distance loss
        # distance_loss = self.distance_weight * self.peak_distance_loss(pred)
        # total_loss = bce_loss + count_loss + distance_loss
        
        total_loss = bce_loss + count_loss
        
        return total_loss, {
            'bce_loss': bce_loss.item(),
            'count_loss': count_loss.item(),
            # 'distance_loss': distance_loss.item()
        }