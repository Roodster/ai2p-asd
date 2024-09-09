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
    
class SegmentsDataset(Dataset):
    def __init__(self, root_dir, mode='full', patient_id=None):
        """
        

        Args:
            root_dir: path to directory
            modes: 
                'full': uses data from all patients.
                'train': exclude data from given id
                'test': only include data from given id
            id: patient id to include/exlude data. Depends on mode.
        """
        
        self.root_dir = root_dir
        self.file_list = self._get_file_list(mode=mode, patient_id=patient_id)


    def _get_file_list(self, mode='full', patient_id=None):
        """
        Generates a list of file paths based on the mode and patient ID.

        Args:
            mode (str): Mode to specify the dataset type ('full', 'train', 'test').
            patient_id (str): Patient ID to include/exclude based on the mode.

        Returns:
            list: List of file paths matching the mode and patient ID criteria.
        """
        file_list = []
        
        for root, _, files in os.walk(self.root_dir):
            # Determine if the current directory matches the inclusion/exclusion criteria
            is_patient_dir = patient_id is not None and patient_id in root
            
            # Filter files based on the mode
            if (mode == 'train' and is_patient_dir):
                # Skip this directory if it's the patient to be excluded
                continue
            
            elif (mode == 'test' and not is_patient_dir):
                # Skip this directory if it doesn't match the patient to include
                continue

            # Add .npz files from the appropriate directories
            for file in files:
                if file.endswith('.npz'):
                    file_list.append(os.path.join(root, file))
                    
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npz_file = np.load(self.file_list[idx])
        segment = torch.from_numpy(npz_file['x'].astype(np.float32))
        
        label = torch.from_numpy(npz_file['y'].astype(np.float32))
        return segment, label
    

# ============================== UTILITIES ==============================

def split_dataset(dataset, train_ratio=0.7, test_ratio=0.15, batch_size=32, shuffle=True, num_workers=1):

    total_size = len(dataset)
    
    train_size = int(train_ratio * total_size)
    test_size = int(test_ratio * total_size)
    val_size = total_size - train_size - test_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_worker=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader    


def get_dataloaders(dataset,
                       train_ratio, 
                       test_ratio,
                       batch_size=32, 
                       shuffle=True,
                       num_workers=1
                       ):
    
    return split_dataset(dataset,
                         train_ratio=train_ratio,
                         test_ratio=test_ratio,
                         batch_size=batch_size, 
                         shuffle=shuffle, 
                         num_workers=num_workers)
    
    

         
    
if __name__ == "__main__":
   ds1 = SegmentsDataset("../data/preprocessed/", mode='train', patient_id='01')
   ds2 = SegmentsDataset("../data/preprocessed/", mode='test', patient_id='01')
