import torch
from torch.utils.data import Dataset
import tqdm
import numpy as np
from pathlib import Path


class PulseDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = Path(data_path)
        self._load_data()
        
    def _load_data(self):
        file_list = list(self.data_path.glob('*.npz'))
        if len(file_list) == 0:
            raise FileNotFoundError(f"No files found in {self.data_path}")
        
        self.head_data = []
        self.head_label = []
        self.heart_data = []
        self.heart_label = []
        self.wrist_data = []
        self.wrist_label = []
        
        for file_item in tqdm.tqdm(file_list):
            loaded_data = np.load(file_item)
            head_data = torch.from_numpy(loaded_data['head_data'])
            head_data = head_data.view(head_data.shape[0], head_data.shape[1], -1) # reshape the bins dimension to 1D
            head_data = torch.cat((head_data.real, head_data.imag), dim=-1) # convert the complex data to 2D tensor
            self.head_data.append(head_data)
            self.head_label.append(torch.from_numpy(loaded_data['head_label']))
            
            heart_data = torch.from_numpy(loaded_data['heart_data'])
            heart_data = heart_data.view(heart_data.shape[0], heart_data.shape[1], -1)
            heart_data = torch.cat((heart_data.real, heart_data.imag), dim=-1)
            self.heart_data.append(heart_data)
            self.heart_label.append(torch.from_numpy(loaded_data['heart_label']))
            
            wrist_data = torch.from_numpy(loaded_data['wrist_data'])
            wrist_data = wrist_data.view(wrist_data.shape[0], wrist_data.shape[1], -1)
            wrist_data = torch.cat((wrist_data.real, wrist_data.imag), dim=-1)
            self.wrist_data.append(wrist_data)
            self.wrist_label.append(torch.from_numpy(loaded_data['wrist_label']))

        # self.head_data = torch.cat(self.head_data, dim=0).float()
        # self.head_label = torch.cat(self.head_label, dim=0).float()
        # print("Head data shape: ", self.head_data.shape, "Head label shape: ", self.head_label.shape)
        
        self.heart_data = torch.cat(self.heart_data, dim=0).float()
        self.heart_label = torch.cat(self.heart_label, dim=0).float()
        print("Heart data shape: ", self.heart_data.shape, "Heart label shape: ", self.heart_label.shape)
        
        # self.wrist_data = torch.cat(self.wrist_data, dim=0).float()
        # self.wrist_label = torch.cat(self.wrist_label, dim=0).float()
        # print("Wrist data shape: ", self.wrist_data.shape, "Wrist label shape: ", self.wrist_label.shape)
        
        return
        
    def __len__(self):
        return self.heart_data.shape[0]
        
    def __getitem__(self, idx):
        # return self.head_data[idx], self.head_label[idx], self.heart_data[idx], self.heart_label[idx], self.wrist_data[idx], self.wrist_label[idx]
        return [], [], self.heart_data[idx], self.heart_label[idx], [], []