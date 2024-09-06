import os
import torch
from torch.utils.data import Dataset, DataLoader
import seizure_data_processing as sdp

class PIDataset(Dataset):
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
        data = torch.tensor(eeg_file.data)
        labels = torch.tensor(eeg_file.get_labels())
        return data, labels

def get_pi_dataloader(path, channels, batch_size=32, shuffle=True):
    dataset = PIDataset(path, channels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)