import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import seizure_data_processing as sdp
import random

class RawDataset(Dataset):
    def __init__(self, path, channels):
        self.path = path
        self.channels = channels
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        file_list = []
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith('.edf'):
                    file_list.append(os.path.join(root, file))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        eeg_file = sdp.EEG(file, channels=self.channels)
        data = torch.tensor(eeg_file.data.astype(np.float32))
        labels = torch.tensor(eeg_file.get_labels())
        return data, labels
    
class SpectrogramDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        file_list = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.npz'):
                    file_list.append(os.path.join(root, file))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npz_file = np.load(self.file_list[idx])
        spectrogram = torch.from_numpy(npz_file['x'].astype(np.float32))
        
        label = torch.from_numpy(npz_file['y'].astype(np.float32))
        return spectrogram, label
    
    
def load_spectrograms_and_labels(load_dir):
    """
    Load spectrograms and labels from .npy files.
    
    :param load_dir: directory containing the .npy files
    :return: tuple of (spectrograms, labels)
    """
    data = np.load(os.path.join(load_dir, 'spectrograms.npz'), allow_pickle=True)

    # return spectrograms and labels per segment
    return data['x'], data['y']

def split_dataset(dataset, train_ratio=0.7, test_ratio=0.15, batch_size=32, shuffle=True):

    total_size = len(dataset)
    
    train_size = int(train_ratio * total_size)
    test_size = int(test_ratio * total_size)
    val_size = total_size - train_size - test_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_dataloaders(dataset,
                       train_ratio, 
                       test_ratio,
                       batch_size=32, 
                       shuffle=True
                       ):
    
    return split_dataset(dataset,
                         train_ratio=train_ratio,
                         test_ratio=test_ratio,
                         batch_size=batch_size, 
                         shuffle=shuffle)