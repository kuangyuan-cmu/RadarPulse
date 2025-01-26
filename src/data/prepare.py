from pathlib import Path
import scipy.io
import numpy as np
import argparse
import tqdm

fs = 500
# total_duration = 170 * fs
sample_len = 10 * fs
overlap = 0.8
intra_session_split_ratio = 0

sites = ['head', 'heart', 'wrist', 'neck']
sites_gt = ['bcg', 'scg', 'ppg', 'neck']

def intra_session_split(data: dict, label: dict, ratio: float):
    """
    Splits the given data and labels into training and testing sets based on the specified ratio.
    Args:
        data (dict): A dictionary where keys are session identifiers and values are lists of data points.
        label (dict): A dictionary where keys are session identifiers and values are lists of labels corresponding to the data points.
        ratio (float): A float value between 0 and 1 representing the proportion of data to be used for training.
    Returns:
        tuple: A tuple containing four dictionaries:
            - train_data (dict): Training data split from the input data.
            - train_label (dict): Training labels split from the input labels.
            - test_data (dict): Testing data split from the input data.
            - test_label (dict): Testing labels split from the input labels.
    """
    train_data = {}
    train_label = {}
    test_data = {}
    test_label = {}
    
    for key in data.keys():
        split = int(len(data[key]) * ratio)
        train_data[key] = data[key][:split]
        train_label[key] = label[key][:split]
        test_data[key] = data[key][split:]
        test_label[key] = label[key][split:]
    return train_data, train_label, test_data, test_label

def sliding_window_sampling(data: dict, label: dict, sample_len: int, overlap: float):
    """
    Splits the given data and labels into smaller samples using a sliding window approach.
    Args:
        data (dict): A dictionary where keys are session identifiers and values are lists of data points.
        label (dict): A dictionary where keys are session identifiers and values are lists of labels corresponding to the data points.
        sample_len (int): The length of each sample in number of data points.
        overlap (float): The proportion of overlap between consecutive samples.
    Returns:
        dict: A dictionary containing the sampled data and labels.
    """
    results = {}
    for key in data.keys():
        results[key + "_data"] = []
        results[key + "_label"] = []
        for i in range(0, len(data[key]) - sample_len, int(sample_len * (1 - overlap))):
            results[key + "_data"].append(data[key][i:i + sample_len])
            results[key + "_label"].append(label[key][i:i + sample_len])
    return results


def main(dataset_path):

    dataset_path = Path(dataset_path)
    exps = (dataset_path / 'raw').glob('*')

    train_path = dataset_path / 'train'
    train_path.mkdir(exist_ok=True)
    dev_path = dataset_path / 'dev'
    dev_path.mkdir(exist_ok=True)

    for exp in tqdm.tqdm(exps):
        gt_files = list(exp.glob('GT*.mat'))
        if len(gt_files) != 1:
            print(f"Warning: {exp} has {len(gt_files)} GT files")
        
        # Clear dictionaries at the start of each experiment
        site_files = {}
        site_data = {}
        site_labels = {}
        
        # Check which files exist for each site
        for site in sites:
            site_files[site] = list(exp.glob(f'{site}*.mat'))
        
        # Only process if we have GT files
        for gt_file in gt_files:
            exp_name = gt_file.name[3:].split('.')[0]
            train_file = train_path / f'{exp_name}.npz'
            dev_file = dev_path / f'{exp_name}.npz'
            
            if train_file.exists():
                continue
                
            # Load data for each available site
            data_len = None
            for site in sites:
                if site_files[site]:  # If files exist for this site
                    # Use memmap to load large files more efficiently
                    mat_contents = scipy.io.loadmat(site_files[site][0])
                    site_data[site] = mat_contents['data']
                    if data_len is None:
                        data_len = site_data[site].shape[0]
                    else:
                        assert site_data[site].shape[0] == data_len, f"Data length mismatch for {site}"
                    
                    # Explicitly delete mat_contents to free memory
                    del mat_contents
            
            print("Data shapes:", {site: site_data[site].shape for site in site_data})
            
            # Initialize labels more efficiently using numpy
            site_labels = {site: np.zeros((data_len, 1), dtype=np.int8) for site in site_data}
            
            # Load ground truth data with memmap
            gt_data = scipy.io.loadmat(gt_file)
            for i, site in enumerate(sites):
                if site in site_data:
                    peaks_gt = gt_data[f'{sites_gt[i]}_peaks_gt'].T
                    site_labels[site][peaks_gt] = 1
            
            data = {site: site_data[site].copy() for site in site_data}  # Create copies to avoid memory issues
            label = {site: site_labels[site].copy() for site in site_labels}
            
            # Clear original dictionaries to free memory
            site_data.clear()
            site_labels.clear()
            del gt_data
            
            train_data, train_label, test_data, test_label = intra_session_split(data, label, intra_session_split_ratio)
            sampled_train_data = sliding_window_sampling(train_data, train_label, sample_len, overlap)
            sampled_test_data = sliding_window_sampling(test_data, test_label, sample_len, overlap)
            
            print("Exp Name:", exp_name, "Train Data:", len(sampled_train_data[f'{list(data.keys())[0]}_data']), 
                  "Test Data:", len(sampled_test_data[f'{list(data.keys())[0]}_data']))

            # Save and clear data
            np.savez(train_file, **sampled_train_data)
            np.savez(dev_file, **sampled_test_data)
            
            # Clear all data after saving
            del data, label, train_data, train_label, test_data, test_label, sampled_train_data, sampled_test_data


def main_v2(dataset_path):
    dataset_path = Path(dataset_path)
    users = (dataset_path / 'raw').glob('*')

    processed_path = dataset_path / 'processed'
    processed_path.mkdir(exist_ok=True)

    for user in users:
        print(f"Processing user: {user.name}")
        user_path = processed_path / user.name
        user_path.mkdir(exist_ok=True)
        
        exps = user.glob('*')
        
        for exp in tqdm.tqdm(exps):
            gt_files = list(exp.glob('GT*.mat'))
            if len(gt_files) != 1:
                print(f"Warning: {exp} has {len(gt_files)} GT files")
            
            # Clear dictionaries at the start of each experiment
            site_files = {}
            site_data = {}
            site_labels = {}
            
            # Check which files exist for each site
            for site in sites:
                site_files[site] = list(exp.glob(f'{site}*.mat'))
            
            # Only process if we have GT files
            for gt_file in gt_files:
                exp_name = gt_file.name[3:].split('.')[0]
                processed_file = user_path / f'{exp_name}.npz'
                
                if processed_file.exists():
                    continue
                    
                # Load data for each available site
                data_len = None
                for site in sites:
                    if site_files[site]:  # If files exist for this site
                        # Use memmap to load large files more efficiently
                        mat_contents = scipy.io.loadmat(site_files[site][0])
                        site_data[site] = mat_contents['data']
                        if data_len is None:
                            data_len = site_data[site].shape[0]
                        else:
                            assert site_data[site].shape[0] == data_len, f"Data length mismatch for {site}"
                        
                        # Explicitly delete mat_contents to free memory
                        del mat_contents
                
                print("Data shapes:", {site: site_data[site].shape for site in site_data})
                
                # Initialize labels more efficiently using numpy
                site_labels = {site: np.zeros((data_len, 1), dtype=np.int8) for site in site_data}
                
                # Load ground truth data with memmap
                gt_data = scipy.io.loadmat(gt_file)
                for i, site in enumerate(sites):
                    if site in site_data:
                        peaks_gt = gt_data[f'{sites_gt[i]}_peaks_gt'].T
                        site_labels[site][peaks_gt] = 1
                
                data = {site: site_data[site].copy() for site in site_data}  # Create copies to avoid memory issues
                label = {site: site_labels[site].copy() for site in site_labels}
                
                # Clear original dictionaries to free memory
                site_data.clear()
                site_labels.clear()
                del gt_data
                
                sampled_data = sliding_window_sampling(data, label, sample_len, overlap)
                
                print("Exp Name:", exp_name, "Samples:", len(sampled_data[f'{list(data.keys())[0]}_data']))

                # Save and clear data
                np.savez(processed_file, **sampled_data)
                
                # Clear all data after saving
                del data, label, sampled_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process .mat files and generate training and testing set.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset directory')
    args = parser.parse_args()
    # main(args.dataset_path)
    main_v2(args.dataset_path)
    
    # main('/home/kyuan/RadarPulse/dataset/phase1_1212')