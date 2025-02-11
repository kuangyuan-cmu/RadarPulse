import torch
from torch.utils.data import Dataset
import tqdm
import numpy as np
from pathlib import Path
import LnkParse3


class PulseDataset(Dataset):
    def __init__(self, data_path, data_config, include_users=None, exclude_users=None, enable_augment=False):
        super().__init__()
        self.data_path = Path(data_path)
        self.data_config = data_config
        self.include_users = include_users
        self.exclude_users = exclude_users
        self.is_joint = isinstance(self.data_config, list)
        
        if self.is_joint:
            self.pulse_position = [config.position for config in self.data_config]
            self.signal_type = [config.signal_type for config in self.data_config]
            self.norm_2d = [config.norm_2d for config in self.data_config]
            self.sample_len = self.data_config[0].sample_len
            self.n_channels = [config.n_channels for config in self.data_config]
            self.in_channels = max(self.n_channels)  # Keep max channels for getitem output shape
            self.enable_augment = False
        else:
            self.pulse_position = self.data_config.position
            self.signal_type = self.data_config.signal_type
            self.norm_2d = self.data_config.norm_2d
            self.sample_len = self.data_config.sample_len
            self.in_channels = self.data_config.n_channels
            self.enable_augment = enable_augment
            if self.enable_augment:
                self.augment_channels = self.data_config.augment_channels
            else:
                self.augment_channels = None
            
        self.data = []
        self.label = []
        self.file_name = []
        self._load_data()
        
    def _load_data(self):
        
        if self.include_users is None and self.exclude_users is None:
            file_list = sorted(list(self.data_path.glob('*.npz*')), key=lambda x: int(x.stem.split('r')[-1]))
            
        else:
            user_dirs = [d for d in self.data_path.glob('processed/*/') if d.is_dir()]
            if isinstance(self.include_users, str):
                self.include_users = [self.include_users]
            if isinstance(self.exclude_users, str):
                self.exclude_users = [self.exclude_users]
            if self.include_users:
                user_dirs = [d for d in user_dirs if d.name in self.include_users]
            if self.exclude_users:
                user_dirs = [d for d in user_dirs if d.name not in self.exclude_users]
                
            if len(user_dirs) == 0:
                raise FileNotFoundError(f"No valid user directories found in {self.data_path}")
            file_list = []
            for user_dir in user_dirs:
                file_list.extend(list(user_dir.glob('*.npz*')))
            file_list.sort(key=lambda x: int(x.stem.split('r')[-1]))
        

        if len(file_list) == 0:
            raise FileNotFoundError(f"No files found in {self.data_path}")
        
        # Calculate total size first and pre-allocate tensors
        total_samples = 0
        for file_item in tqdm.tqdm(file_list, desc="Calculating total size"):
            if file_item.suffix == '.lnk':
                with open(file_item, 'rb') as indata:
                    lnk = LnkParse3.lnk_file(indata)
                    file_item = '/' + lnk.get_json()['link_info']['common_path_suffix'].replace("\\", '/')
            loaded_data = np.load(file_item, mmap_mode='r')
            if self.is_joint:
                # load data only when all sites have data
                if not all(f'{pos}_label' in loaded_data for pos in self.pulse_position):
                    continue
                total_samples += loaded_data['heart_label'].shape[0]
            else:
                if f'{self.pulse_position}_label' not in loaded_data:
                    continue
                total_samples += loaded_data[f'{self.pulse_position}_label'].shape[0]

        # Pre-allocate tensors
        if self.is_joint:
            # Create separate data arrays for each site
            self.data = []
            for i, n_chan in enumerate(self.n_channels):
                self.data.append(torch.empty((total_samples, self.sample_len, n_chan), dtype=torch.float32))
            self.label = torch.empty((total_samples, len(self.pulse_position), self.sample_len, 1), dtype=torch.float32)
        else:
            self.data = torch.empty((total_samples, self.sample_len, self.in_channels), dtype=torch.float32)
            self.label = torch.empty((total_samples, self.sample_len, 1), dtype=torch.float32)

        print(f"Label shape: {self.label.shape}")
        if self.is_joint:
            for i, d in enumerate(self.data):
                print(f"Data shape for {self.pulse_position[i]}: {d.shape}")
        else:
            print(f"Data shape: {self.data.shape}")
        
        # Fill the pre-allocated tensors
        current_idx = 0
        
        for file_item in tqdm.tqdm(file_list, desc="Loading data"):
            # if the file is a link file, get the real file path
            if file_item.suffix == '.lnk':
                with open(file_item, 'rb') as indata:
                    lnk = LnkParse3.lnk_file(indata)
                    file_item = '/' + lnk.get_json()['link_info']['common_path_suffix'].replace("\\", '/')
                file_name = file_item.split('/')[-1].split('.')[0]
            else:
                file_name = file_item.name.split('.')[0]
            
            loaded_data = np.load(file_item, mmap_mode='r')
            
            if self.is_joint:
                if not all(f'{pos}_label' in loaded_data for pos in self.pulse_position):
                    continue
                
                # Load data and labels for each position
                for i, pos in enumerate(self.pulse_position):
                    data = loaded_data[f'{pos}_data']
                    # if pos == 'head':
                        # data = data.sum(axis=2)
                        # data = data[:,:,1:-1]
                    data = data.reshape(data.shape[0], self.sample_len, -1)
                    data = self.signal_conversion(data, type=self.signal_type[self.pulse_position.index(pos)])
                    if self.norm_2d[i]:
                        data = (data - data.mean(axis=(-1,-2), keepdims=True)) / data.std(axis=(-1,-2), keepdims=True)
                    else:
                        data = (data - data.mean(axis=-2, keepdims=True)) / data.std(axis=-2, keepdims=True)
                    
                    # Store data and label directly in pre-allocated arrays
                    batch_size = data.shape[0]
                    self.data[i][current_idx:current_idx + batch_size] = torch.from_numpy(data)
                    self.label[current_idx:current_idx + batch_size, i, :, :] = torch.from_numpy(loaded_data[f'{pos}_label'])
                  
            else:
                data_key = f'{self.pulse_position}_data'
                label_key = f'{self.pulse_position}_label'
                
                if data_key not in loaded_data:
                    continue
                    
                data = loaded_data[data_key]
                label = loaded_data[label_key]
                
                # Special handling for head position
                # if self.pulse_position == 'head':
                    # data = data.sum(axis=2)
                    # data = data[:,:,1:-1]
        
                data = data.reshape(data.shape[0], data.shape[1], -1)
                data = self.signal_conversion(data, type=self.signal_type)
                
                if self.norm_2d:
                    data = (data - data.mean(axis=(-1,-2), keepdims=True)) / data.std(axis=(-1,-2), keepdims=True)
                else:
                    data = (data - data.mean(axis=-2, keepdims=True)) / data.std(axis=-2, keepdims=True)
                data = torch.from_numpy(data)
                label = torch.from_numpy(label)
                
                batch_size = data.shape[0]
                self.data[current_idx:current_idx + batch_size] = data
                self.label[current_idx:current_idx + batch_size] = label
            
            self.file_name.extend([file_name] * batch_size)
            current_idx += batch_size
            # Free memory
            # del data, label
            if 'loaded_data' in locals():
                del loaded_data


        if self.is_joint:
            print(self.pulse_position, self.signal_type)
            for i, d in enumerate(self.data):
                print(f"Final data shape for {self.pulse_position[i]}: {d.shape}")
            print(f"Final label shape: {self.label.shape}")
        else:
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
        if self.is_joint:
            # Create output tensor with max channels
            data = torch.zeros((len(self.pulse_position), self.sample_len, self.in_channels))
            # Fill each position's data up to its number of channels
            for i, site_data in enumerate(self.data):
                data[i, :, :self.n_channels[i]] = site_data[idx]
            return data, self.label[idx], self.file_name[idx]
        elif self.enable_augment:
            data = self.data[idx]
            # Randomly select self.augment_channels number of channels
            channel_indices = torch.randperm(self.in_channels)[:self.augment_channels]
            data = data[..., channel_indices]
            return data, self.label[idx], self.file_name[idx]
        else:
            return self.data[idx], self.label[idx], self.file_name[idx]
    

    
if __name__ == '__main__':
    Dataset = PulseDataset(data_path='/home/kyuan/RadarPulse/dataset/phase1_new_1214_cross_sessions/dev/', pulse_position='wrist')