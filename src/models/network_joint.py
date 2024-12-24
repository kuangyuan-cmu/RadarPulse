import torch
import torch.nn as nn
import torch.nn.functional as F
from .network import PulseDetectionNet 
from torchinfo import summary

class CrossSiteAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, site_features):
        # site_features: list of (N, L, C) tensors
        stacked = torch.stack(site_features, dim=1)  # (N, num_sites, L, C)
        N, num_sites, L, C = stacked.shape
        
        # Reshape to process each timepoint: (N*L, num_sites, C)
        # This way sites can attend to each other at each timepoint
        reshaped = stacked.transpose(1, 2).reshape(N*L, num_sites, C)
        
        # Self attention across sites
        attn_out, _ = self.mha(reshaped, reshaped, reshaped)
        
        # Residual connection and normalization
        attn_out = attn_out + reshaped
        attn_out = self.norm(attn_out)
        
        # Reshape back: (N, L, num_sites, C)
        attn_out = attn_out.reshape(N, L, num_sites, C).transpose(1, 2)
        
        # Split back to list of tensors [(N, L, C), ...]
        return [attn_out[:, i, :, :] for i in range(num_sites)]

class CrossSiteTemporalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.site_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.temporal_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, site_features):
        # site_features: list of (N, L, C) tensors
        stacked = torch.stack(site_features, dim=1)  # (N, num_sites, L, C)
        N, num_sites, L, C = stacked.shape
        
        # 1. Cross-site attention at each timepoint
        reshaped = stacked.transpose(1, 2).reshape(N*L, num_sites, C)
        site_attn, _ = self.site_attention(reshaped, reshaped, reshaped)
        site_attn = self.norm1(site_attn + reshaped)
        
        # Reshape for temporal attention: (N*num_sites, L, C)
        temporal_input = site_attn.reshape(N, L, num_sites, C).transpose(1, 2)
        temporal_input = temporal_input.reshape(N*num_sites, L, C)
        
        # 2. Temporal attention for each site's features
        temporal_attn, _ = self.temporal_attention(temporal_input, temporal_input, temporal_input)
        temporal_attn = self.norm2(temporal_attn + temporal_input)
        
        # Reshape back: (N, num_sites, L, C)
        output = temporal_attn.reshape(N, num_sites, L, C)
        
        # Return list of tensors [(N, L, C), ...]
        return [output[:, i, :, :] for i in range(num_sites)]

class MultiSitePulseDetectionNet(nn.Module):
    def __init__(self, site_configs):
        super().__init__()
        
        self.num_sites = len(site_configs)
        
        # Create site-specific networks
        self.site_networks = nn.ModuleList([
            PulseDetectionNet(**config) for config in site_configs
        ])
        self.site_in_channels = [config.in_channels for config in site_configs]
        # Cross-site fusion
        lstm_hidden_size = site_configs[0]['lstm_hidden_size'] * 2  # bidirectional
        # self.cross_site_attention = CrossSiteAttention(
        #     hidden_size=lstm_hidden_size
        # )
        self.cross_site_attention = CrossSiteTemporalAttention(
            hidden_size=lstm_hidden_size
        )
        
    def forward(self, site_inputs):
        # size_inputs: cell
        # Encode each site
        encoded_features = []
        skip_connections = []
        
        for i, network in enumerate(self.site_networks):
            # Run encoder
            feat, skip = network.encode(site_inputs[:, i, :, :self.site_in_channels[i]])
            # feat, skip = network.encode(site_inputs[i])
            # Run bottleneck
            feat = network.bottleneck(feat)
            encoded_features.append(feat)
            skip_connections.append(skip)
            
        # Cross-site fusion
        fused_features = self.cross_site_attention(encoded_features)
        
        # Decode each site
        outputs = []
        for i, network in enumerate(self.site_networks):
            out = network.decode(fused_features[i], skip_connections[i])
            outputs.append(out)
            
        return torch.stack(outputs, dim=1) # (N, n_sites, L, C)
    
    # def forward(self, site_inputs):
    #     # size_inputs: cell
    #     # Encode each site
    #     encoded_features = []
    #     skip_connections = []
        
    #     for i, network in enumerate(self.site_networks):
    #         # Run encoder
    #         feat, skip = network.encode(site_inputs[:, i, :, :self.site_in_channels[i]])
    #         encoded_features.append(feat.transpose(1, 2))
    #         skip_connections.append(skip)
            
    #     # Cross-site fusion
    #     fused_features = self.cross_site_attention(encoded_features)
        
    #     # LSTM 
    #     for i, network in enumerate(self.site_networks):
    #         fused_features[i] = network.bottleneck(fused_features[i].transpose(1, 2))
            
    #     # Decode each site
    #     outputs = []
    #     for i, network in enumerate(self.site_networks):
    #         out = network.decode(fused_features[i], skip_connections[i])
    #         outputs.append(out)
            
    #     return torch.stack(outputs, dim=1) # (N, n_sites, L, C)
    
    def load_pretrained(self, paths):
        for network, path in zip(self.site_networks, paths):
            state_dict = torch.load(path)['state_dict']
            new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            network.load_state_dict(new_state_dict)
    
    def unfreeze_site(self, site_idx):
        """Unfreeze parameters for a specific site"""
        for param in self.site_networks[site_idx].parameters():
            param.requires_grad = True
            
    def freeze_site(self, site_idx):
        """Freeze parameters for a specific site"""
        for param in self.site_networks[site_idx].parameters():
            param.requires_grad = False
    
    def freeze_all_sites(self):
        for i in range(self.num_sites):
            self.freeze_site(i)

if __name__ == '__main__':
    in_channels = 21
    seq_len = 5000
    lstm_hidden_size = 64
    site_configs = [
        {
            'in_channels': in_channels,
            'seq_len': seq_len,
            'lstm_hidden_size': lstm_hidden_size
        },
        {
            'in_channels': in_channels*2,
            'seq_len': seq_len,
            'lstm_hidden_size': lstm_hidden_size
        },
        {
            'in_channels': in_channels,
            'seq_len': seq_len,
            'lstm_hidden_size': lstm_hidden_size
        },
    ]
    
    model = MultiSitePulseDetectionNet(site_configs)
    info = summary(model, input_size=(3,16, seq_len, in_channels), device='cpu')
    inputs = (torch.randn(16, seq_len, in_channels), torch.randn(16, seq_len, in_channels*2), torch.randn(16, seq_len, in_channels))

    outputs = model(inputs)
    print(outputs.shape)