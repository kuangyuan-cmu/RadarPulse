import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
# from torchsummary import summary
from typing import List
import numpy as np

class SpatialReductionPreprocessing(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dual_type: bool):
        """
        Args:
            in_channels: Number of input spatial points
            out_channels: Desired number of output spatial points
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dual_type = dual_type
        if dual_type:
            self.spatial_temporal_processing_type1 = nn.Sequential(
                # Reshape input to 2D format implicitly in forward pass
                nn.Conv2d(
                    in_channels=1,  # Single channel input after reshape
                    out_channels=16,  # Intermediate feature maps
                    kernel_size=(1, 7),  # (spatial, temporal) kernel
                    padding=(0, 3),  # Same padding to maintain temporal dimension
                    stride=1
                ),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                # Second conv for feature processing
                nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(1, 7),
                    padding=(0, 3),
                    stride=1
                ),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )
            self.spatial_temporal_processing_type2 = nn.Sequential(
                # Reshape input to 2D format implicitly in forward pass
                nn.Conv2d(
                    in_channels=1,  # Single channel input after reshape
                    out_channels=16,  # Intermediate feature maps
                    kernel_size=(1, 7),  # (spatial, temporal) kernel
                    padding=(0, 3),  # Same padding to maintain temporal dimension
                    stride=1
                ),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                # Second conv for feature processing
                nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(1, 7),
                    padding=(0, 3),
                    stride=1
                ),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )
        else:
            self.spatial_temporal_processing = nn.Sequential(
                # Reshape input to 2D format implicitly in forward pass
                nn.Conv2d(
                    in_channels=1,  # Single channel input after reshape
                    out_channels=16,  # Intermediate feature maps
                    kernel_size=(1, 7),  # (spatial, temporal) kernel
                    padding=(0, 3),  # Same padding to maintain temporal dimension
                    stride=1
                ),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                # Second conv for feature processing
                nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(1, 7),
                    padding=(0, 3),
                    stride=1
                ),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )
            
        # Final 1x1 conv to get exact channel count
        self.channel_adjust = nn.Conv2d(16, 1, kernel_size=1)
        
    def forward(self, x):
        # x shape: (batch, spatial, temporal)
        batch, spatial, temporal = x.shape
        
        # Add channel dimension for 2D CNN
        x = x.unsqueeze(1)  # (batch, 1, spatial, temporal)

        if self.dual_type:
            x1 = self.spatial_temporal_processing_type1(x[:,:,:spatial//2,:])
            x2 = self.spatial_temporal_processing_type2(x[:,:,spatial//2:,:])
            x = torch.cat([x1, x2], dim=2)
        else:
            x = self.spatial_temporal_processing(x)  # (batch, 16, spatial, temporal)
            
        x = torch.topk(x, self.out_channels, dim=2)[0]  # (batch, 16, reduce_spatial, temporal)
        
        # Adjust channels
        x = self.channel_adjust(x)  # (batch, 1, reduced_spatial, temporal)
        
        # Remove extra channel dimension and reshape
        x = x.squeeze(1)  # (batch, reduced_spatial, temporal)
        
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(2)
    
    def forward(self, x):
        features = self.conv(x)
        return self.pool(features), features

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels*2, in_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x, skip):
        x = self.transposed_conv(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

    
class PulseDetectionNet(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 seq_len: int,
                 dual_type: bool = False,
                 reduce_channels: int = 8,
                 hidden_channels: List[int] = [32, 64, 128],
                 kernel_size: int = 7,
                 use_lstm: bool = True,
                 lstm_hidden_size: int = 64,
                 lstm_num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.seq_len = seq_len
        self.use_lstm = use_lstm
        
        self.spatial_processing = SpatialReductionPreprocessing(in_channels, reduce_channels, dual_type)
        current_channels = reduce_channels
        # current_channels = in_channels
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        
        for hidden_ch in hidden_channels:
            self.encoder_blocks.append(
                EncoderBlock(current_channels, hidden_ch, kernel_size)
            )
            current_channels = hidden_ch
            
        # LSTM at bottleneck
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_channels[-1],
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                bidirectional=True,
                dropout=dropout if lstm_num_layers > 1 else 0,
                batch_first=True
            )
            current_channels = lstm_hidden_size * 2  # bidirectional
            
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        hidden_channels_reversed = hidden_channels[::-1]
        hidden_channels_reversed.append(32)
        
        for i in range(len(hidden_channels_reversed) - 1):
            in_ch = current_channels
            out_ch = hidden_channels_reversed[i + 1]
            self.decoder_blocks.append(
                DecoderBlock(in_ch, out_ch, kernel_size)
            )
            current_channels = out_ch
            
        # Final convolution
        self.final = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        # Returns encoder features and skip connections
        x = x.transpose(1, 2)  # (N, C, L)
        x = self.spatial_processing(x)
        
        skip_connections = []
        for encoder in self.encoder_blocks:
            x, features = encoder(x)
            skip_connections.append(features)
            
        return x, skip_connections

    def bottleneck(self, x):
        # LSTM processing
        batch_size, channels, seq_len = x.shape
        x = x.transpose(1, 2)  # (N, L, C)
        x, _ = self.lstm(x)
        return x  # Keep in (N, L, C) for attention

    def decode(self, x, skip_connections):
        # x shape: (N, L, C) from bottleneck
        x = x.transpose(1, 2)  # (N, C, L)
        
        skip_connections = skip_connections[::-1]
        for decoder, skip in zip(self.decoder_blocks, skip_connections):
            x = decoder(x, skip)
            
        x = self.final(x)
        return x.transpose(1, 2)  # (N, L, 1)

    def forward(self, x):
        x, skip_connections = self.encode(x)
        x = self.bottleneck(x)
        x = self.decode(x, skip_connections)
        return x
    
    # def forward(self, x):
    #     # Input shape: (N, L, C)
    #     x = x.transpose(1, 2)  # (N, C, L)
        
    #     x = self.spatial_processing(x) # (N, reduced_channels, L)
    #     # Store skip connections
    #     skip_connections = []
        
    #     # Encoder
    #     for encoder in self.encoder_blocks:
    #         x, features = encoder(x)  # Get both output and features
    #         skip_connections.append(features)  # Store features for skip connection
            
    #     # LSTM at bottleneck
    #     if self.use_lstm:
    #         batch_size, channels, seq_len = x.shape
    #         x = x.transpose(1, 2)  # (N, L, C)
    #         x, _ = self.lstm(x)
    #         x = x.transpose(1, 2)  # (N, C, L)
            
    #     # Decoder
    #     skip_connections = skip_connections[::-1]  # Reverse and skip the last encoder output
    #     for decoder, skip in zip(self.decoder_blocks, skip_connections):
    #         x = decoder(x, skip)
            
    #     # Final convolution
    #     x = self.final(x)
        
    #     return x.transpose(1, 2)  # (N, L, 1)

if __name__ == '__main__':
    # Test network
    torch.manual_seed(0)
    net = PulseDetectionNet(in_channels=40, seq_len=5000)
    # model summary
    info = summary(net, input_size=(16, 5000, 40), device='cpu')
    
    x = torch.randn(16, 5000, 40)
    y = net(x)
    print(y.shape)
    print(y[0, :20, 0])