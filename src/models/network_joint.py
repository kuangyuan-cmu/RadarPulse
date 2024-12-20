import torch
import torch.nn as nn
import torch.nn.functional as F
from .network import PulseDetectionNet


class CrossSiteAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_size, num_heads)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, site_features):
        # site_features: list of (N, L, C)
        stacked = torch.stack(site_features, dim=0)  # (num_sites, N, L, C)
        
        # Self attention across sites
        attn_out, _ = self.mha(stacked, stacked, stacked)
        
        # Residual connection and normalization
        attn_out = attn_out + stacked
        attn_out = self.norm(attn_out)
        
        # Split back to list of tensors
        return list(attn_out)


class MultiSitePulseDetectionNet(nn.Module):
    def __init__(self, site_configs, num_sites):
        super().__init__()
        
        self.num_sites = num_sites
        
        # Create site-specific networks
        self.site_networks = nn.ModuleList([
            PulseDetectionNet(**config) for config in site_configs
        ])
        
        # Cross-site fusion
        lstm_hidden_size = site_configs[0]['lstm_hidden_size'] * 2  # bidirectional
        self.cross_site_attention = CrossSiteAttention(
            hidden_size=lstm_hidden_size
        )
        
        # Optional: Site embeddings
        self.site_embeddings = nn.Parameter(
            torch.randn(num_sites, lstm_hidden_size)
        )
        
    def forward(self, site_inputs):
        # site_inputs: list of tensors [(N, L, C), ...]
        batch_size = site_inputs[0].shape[0]
        
        # Encode each site
        encoded_features = []
        skip_connections = []
        
        for i, (network, x) in enumerate(zip(self.site_networks, site_inputs)):
            # Run encoder
            feat, skip = network.encode(x)
            # Run bottleneck
            feat = network.bottleneck(feat)
            
            # Add site-specific embedding
            site_emb = self.site_embeddings[i].unsqueeze(0).expand(batch_size, -1)
            feat = feat + site_emb.unsqueeze(1)
            
            encoded_features.append(feat)
            skip_connections.append(skip)
            
        # Cross-site fusion
        fused_features = self.cross_site_attention(encoded_features)
        
        # Decode each site
        outputs = []
        for i, network in enumerate(self.site_networks):
            out = network.decode(fused_features[i], skip_connections[i])
            outputs.append(out)
            
        return outputs