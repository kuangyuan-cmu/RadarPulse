import torch
from torch.utils.data import Dataset
import tqdm
import numpy as np
from pathlib import Path
import LnkParse3


class PulseDataset(Dataset):
    def __init__(self, data_path, pulse_position, signal_type, norm_2d=False):
        super().__init__()
        self.data_path = Path(data_path)
        self.pulse_position = pulse_position
        self.norm_2d = norm_2d
        self.signal_type = signal_type
        self.is_joint = isinstance(self.pulse_position, list)
        self.data = []
        self.label = []
        self.file_name = []
        self._load_data()
        
    def _load_data(self):
        file_list = list(self.data_path.glob('*.npz*'))
        if len(file_list) == 0:
            raise FileNotFoundError(f"No files found in {self.data_path}")
        
        for file_item in tqdm.tqdm(file_list):
            
            # with open(file_item, 'rb') as indata:
            #     lnk = LnkParse3.lnk_file(indata)
            #     file_item = '/' + lnk.get_json()['link_info']['common_path_suffix'].replace("\\", '/')
            # file_name = file_item.split('/')[-1].split('.')[0]
            
            file_name = file_item.name.split('.')[0]
            
            loaded_data = np.load(file_item, mmap_mode='r')
            
            if self.is_joint:
                self.norm_2d = False
                
                head_data = loaded_data['head_data'].sum(axis=2)
                head_label = loaded_data['head_label']
                heart_data = loaded_data['heart_data']
                heart_label = loaded_data['heart_label']
                wrist_data = loaded_data['wrist_data']
                wrist_label = loaded_data['wrist_label']
                
                head_data = head_data.reshape(head_data.shape[0], head_data.shape[1], -1)
                heart_data = heart_data.reshape(heart_data.shape[0], heart_data.shape[1], -1)
                wrist_data = wrist_data.reshape(wrist_data.shape[0], wrist_data.shape[1], -1)
                
                head_data = self.signal_conversion(head_data, type=self.signal_type[self.pulse_position.index('head')])
                heart_data = self.signal_conversion(heart_data, type=self.signal_type[self.pulse_position.index('heart')])
                wrist_data = self.signal_conversion(wrist_data, type=self.signal_type[self.pulse_position.index('wrist')])
                
                # normalize the data
                head_data = (head_data - head_data.mean(axis=-2, keepdims=True)) / head_data.std(axis=-2, keepdims=True)
                heart_data = (heart_data - heart_data.mean(axis=-2, keepdims=True)) / heart_data.std(axis=-2, keepdims=True)
                wrist_data = (wrist_data - wrist_data.mean(axis=-2, keepdims=True)) / wrist_data.std(axis=-2, keepdims=True)
                
                # convert the last dimension to share the same length
                max_dim = max(head_data.shape[-1], heart_data.shape[-1], wrist_data.shape[-1])
                data = torch.zeros((head_data.shape[0], 3, head_data.shape[1], max_dim))
                label = torch.zeros((head_data.shape[0], 3, head_data.shape[1], 1))
                data[:, 0, :, :head_data.shape[-1]] = torch.from_numpy(head_data)
                data[:, 1, :, :heart_data.shape[-1]] = torch.from_numpy(heart_data)
                data[:, 2, :, :wrist_data.shape[-1]] = torch.from_numpy(wrist_data)
            
                label[:, 0, :, :] = torch.from_numpy(head_label)
                label[:, 1, :, :] = torch.from_numpy(heart_label)
                label[:, 2, :, :] = torch.from_numpy(wrist_label)
                
                  
            else:
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
                data = self.signal_conversion(data, type=self.signal_type)
                
                if self.norm_2d:
                    data = (data - data.mean(axis=(-1,-2), keepdims=True)) / data.std(axis=(-1,-2), keepdims=True)
                else:
                    data = (data - data.mean(axis=-2, keepdims=True)) / data.std(axis=-2, keepdims=True)
                data = torch.from_numpy(data)
                label = torch.from_numpy(label)
                
            self.data.append(data)
            self.label.append(label)
            self.file_name.extend([file_name] * label.shape[0])
        
        print("Concatenating data and label")
        self.data = torch.cat(self.data, dim=0).float()
        self.label = torch.cat(self.label, dim=0).float()
        
        # if self.norm_2d:
        #     self.data = (self.data - self.data.mean(axis=(-1,-2), keepdims=True)) / self.data.std(axis=(-1,-2), keepdims=True)
        # else:
        #     self.data = (self.data - self.data.mean(axis=-2, keepdims=True)) / self.data.std(axis=-2, keepdims=True)
        
        print(self.pulse_position, self.signal_type, "Data shape: ", self.data.shape, "Label shape: ", self.label.shape)
        
        return
    
    def signal_conversion(self, data, type):
        if type == 'phase':
            return np.unwrap(np.angle(data), axis=1)
        elif type == 'mag':
            return np.abs(data)
        elif type == 'both':
            phase = np.unwrap(np.angle(data), axis=1)
            mag = np.abs(data)
            return np.concatenate((phase, mag), axis=-1)
        else:
            raise ValueError(f"Invalid signal type: {type}")
        
    def __len__(self):
        return self.label.shape[0]
        
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.file_name[idx]
    

    
if __name__ == '__main__':
    Dataset = PulseDataset(data_path='/home/kyuan/RadarPulse/dataset/phase1_new_1214_cross_sessions/dev/', pulse_position='wrist')