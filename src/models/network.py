import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

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
                 hidden_channels: List[int] = [64, 128, 256],
                 kernel_size: int = 5,
                 use_lstm: bool = True,
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.seq_len = seq_len
        self.use_lstm = use_lstm
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        current_channels = in_channels
        
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
            # Calculate correct input channels for each decoder block
            # in_ch = current_channels + hidden_channels_reversed[i]
            in_ch = current_channels
            out_ch = hidden_channels_reversed[i + 1]
            self.decoder_blocks.append(
                DecoderBlock(in_ch, out_ch, kernel_size)
            )
            current_channels = out_ch
            
        # Final convolution
        self.final = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='linear', align_corners=False),  # Add this
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input shape: (N, L, C)
        x = x.transpose(1, 2)  # (N, C, L)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder
        for encoder in self.encoder_blocks:
            x, features = encoder(x)  # Get both output and features
            skip_connections.append(features)  # Store features for skip connection
            
        # LSTM at bottleneck
        if self.use_lstm:
            batch_size, channels, seq_len = x.shape
            x = x.transpose(1, 2)  # (N, L, C)
            x, _ = self.lstm(x)
            x = x.transpose(1, 2)  # (N, C, L)
            
        # Decoder
        skip_connections = skip_connections[::-1]  # Reverse and skip the last encoder output
        for decoder, skip in zip(self.decoder_blocks, skip_connections):
            x = decoder(x, skip)
            
        # Final convolution
        x = self.final(x)
        
        return x.transpose(1, 2)  # (N, L, 1)

if __name__ == '__main__':
    # Test network
    net = PulseDetectionNet(in_channels=40, seq_len=5000)
    # model summary
    print(net)
    
    x = torch.randn(32, 5000, 40)
    y = net(x)
    print(y.shape)