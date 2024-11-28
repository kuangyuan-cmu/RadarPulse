from pathlib import Path
import scipy.io
import numpy as np
import argparse


fs = 500
total_duration = 170 * fs
sample_len = 10 * fs
overlap = 0.75
intra_session_split_ratio = 0.8
    
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

    for exp in exps:
        gt_files = list(exp.glob('GT*.mat'))
        head_files = list(exp.glob('head*.mat'))
        heart_files = list(exp.glob('heart*.mat'))
        wrist_files = list(exp.glob('wrist*.mat'))

        for gt_file, head_file, heart_file, wrist_file in zip(gt_files, head_files, heart_files, wrist_files):
            exp_name = gt_file.name[3:].split('.')[0]

            head_data = scipy.io.loadmat(head_file)['data']
            heart_data = scipy.io.loadmat(heart_file)['data']
            wrist_data = scipy.io.loadmat(wrist_file)['data']
            assert head_data.shape[0] == heart_data.shape[0] == wrist_data.shape[0]
            print(head_data.shape, heart_data.shape, wrist_data.shape)
            data_len = head_data.shape[0]
            data = {
                'head': head_data,
                'heart': heart_data,
                'wrist': wrist_data,
            }

            offset = total_duration - data_len

            head_labels = np.zeros((data_len, 1), dtype=int)
            heart_labels = np.zeros((data_len, 1), dtype=int)
            wrist_labels = np.zeros((data_len, 1), dtype=int)

            gt_data = scipy.io.loadmat(gt_file)
            head_peaks_gt = gt_data['bcg_peaks_gt'].T - offset
            wrist_peaks_gt = gt_data['ppg_peaks_gt'].T - offset
            heart_peaks_gt = gt_data['scg_peaks_gt'].T - offset

            head_labels[head_peaks_gt] = 1
            wrist_labels[wrist_peaks_gt] = 1
            heart_labels[heart_peaks_gt] = 1

            label = {
                'head': head_labels,
                'heart': heart_labels,
                'wrist': wrist_labels
            }
            train_data, train_label, test_data, test_label = intra_session_split(data, label, intra_session_split_ratio)
            sampled_train_data = sliding_window_sampling(train_data, train_label, sample_len, overlap)
            sampled_test_data = sliding_window_sampling(test_data, test_label, sample_len, overlap)
            print("Exp Name:", exp_name, "Train Data:", len(sampled_train_data['head_data']), "Test Data:", len(sampled_test_data['head_data']))

            train_file = train_path / f'{exp_name}.npz'
            np.savez(train_file, **sampled_train_data)

            dev_file = dev_path / f'{exp_name}.npz'
            np.savez(dev_file, **sampled_test_data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process .mat files and generate training and testing set.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset directory')
    args = parser.parse_args()
    main(args.dataset_path)