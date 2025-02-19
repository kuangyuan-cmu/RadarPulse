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

# New loss module for direct PTT regression.
class DirectPTTRegressionLoss(nn.Module):
    def __init__(self, pairs):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.pairs = pairs
    def forward(self, preds, targets):
        """
        Args:
            preds: Tensor of shape (N, num_pairs) with predicted PTT values.
            targets: Tensor of shape (N, num_pairs) with ground truth PTT values.
        """
        mse_loss = self.criterion(preds, targets)
        
        # Calculate MSE for each pair
        pair_losses = {}
        for i, pair in enumerate(self.pairs):
            pair_name = f'mse_loss_{pair[0]}_{pair[1]}'
            pair_loss = F.mse_loss(preds[:,i], targets[:,i])
            pair_losses[pair_name] = pair_loss.item()
            
        return mse_loss, pair_losses