import torch
from torch.utils.data import Dataset
import tqdm
import numpy as np
from pathlib import Path


class PulseDataset(Dataset):
    def __init__(self, data_path, pulse_position, norm_2d=False):
        super().__init__()
        self.data_path = Path(data_path)
        self.pulse_position = pulse_position
        self.norm_2d = norm_2d
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
                data = loaded_data['head_data'].sum(axis=2)
                label = loaded_data['head_label']
        
            elif self.pulse_position == 'heart':
                data = loaded_data['heart_data']
                label = loaded_data['heart_label']
            
            elif self.pulse_position == 'wrist':
                data = loaded_data['wrist_data']
                label = loaded_data['wrist_label']
      
            data = data.reshape(data.shape[0], data.shape[1], -1)
            # data = torch.cat((data.real, data.imag), dim=-1)
            # unwrap phase
            data = np.unwrap(np.angle(data), axis=1)
            self.data.append(torch.from_numpy(data))
            self.label.append(torch.from_numpy(label))

        self.data = torch.cat(self.data, dim=0).float()
        if self.norm_2d:
            self.data = (self.data - self.data.mean(axis=(1,2), keepdims=True)) / self.data.std(axis=(1,2), keepdims=True)
        else:
            self.data = (self.data - self.data.mean(axis=1, keepdims=True)) / self.data.std(axis=1, keepdims=True)
        self.label = torch.cat(self.label, dim=0).float()
        print(self.pulse_position, "Data shape: ", self.data.shape, "Label shape: ", self.label.shape)
        return
        
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]