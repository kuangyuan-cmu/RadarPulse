import torch
from torch.utils.data import Dataset
import tqdm
import numpy as np
from pathlib import Path


class PulseDataset(Dataset):
    def __init__(self, data_path, pulse_position):
        super().__init__()
        self.data_path = Path(data_path)
        self.pulse_position = pulse_position
        self._load_data()
        
    def _load_data(self):
        file_list = list(self.data_path.glob('*.npz'))
        if len(file_list) == 0:
            raise FileNotFoundError(f"No files found in {self.data_path}")
        
        self.data = []
        self.label = []
        
        for file_item in tqdm.tqdm(file_list):
            loaded_data = np.load(file_item)
            
            if self.pulse_position == 'head':
                data = torch.from_numpy(loaded_data['head_data'])
                data = data.view(data.shape[0], data.shape[1], -1)
                data = torch.cat((data.real, data.imag), dim=-1)
                self.data.append(data)
                self.label.append(torch.from_numpy(loaded_data['head_label']))
                continue
        
            if self.pulse_position == 'heart':
                data = torch.from_numpy(loaded_data['heart_data'])
                data = data.view(data.shape[0], data.shape[1], -1)
                data = torch.cat((data.real, data.imag), dim=-1)
                self.data.append(data)
                self.label.append(torch.from_numpy(loaded_data['heart_label']))
                continue
            
            if self.pulse_position == 'wrist':
                data = torch.from_numpy(loaded_data['wrist_data'])
                data = data.view(data.shape[0], data.shape[1], -1)
                data = torch.cat((data.real, data.imag), dim=-1)
                self.data.append(data)
                self.label.append(torch.from_numpy(loaded_data['wrist_label']))
                continue

        self.data = torch.cat(self.data, dim=0).float()
        self.label = torch.cat(self.label, dim=0).float()
        print(self.pulse_position, "Data shape: ", self.data.shape, "Label shape: ", self.label.shape)
        return
        
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
    
        return self.data[idx], self.label[idx]