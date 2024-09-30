import os
import struct

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import seizure_data_processing as sdp
import random
import time
from tqdm import tqdm
from torchvision import transforms


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
    
class OnlineSegmentsDataset(Dataset):
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
        print(npz_file.shape)

        segment = torch.from_numpy(npz_file['x'].astype(np.float32))

        if len(segment.shape)== 2:
            segment = segment.reshape(1, segment.shape[0], segment.shape[1]) 
                
        label = torch.from_numpy(npz_file['y'].astype(np.float32))        
        return segment, label
    
    
class OfflineSegmentsDataset(Dataset):
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
        self.data = self._load_data()

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

    def _load_data(self):
        """
        Loads all data into RAM during initialization.

        Returns:
            list: A list of tuples where each tuple contains a segment tensor and a label tensor.
        """
        data = []
        pbar = tqdm(self.file_list)
        pbar.set_description("Loading dataset...")
        for file_path in pbar:
            npz_file = np.load(file_path)
            segment = torch.from_numpy(npz_file['x'].astype(np.float32))
                        
            if len(segment.shape)== 2:
                segment = segment.reshape(1, segment.shape[0], segment.shape[1]) 
                
            label = torch.from_numpy(npz_file['y'].astype(np.float32))
            data.append((segment, label))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the segment, label, and loading time for the specified index.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            tuple: (segment, label, load_time)
        """
        segment, label = self.data[idx]
        
        return segment, label
    
    
    
class OfflineFeaturesDataset(OfflineSegmentsDataset):
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
        super(OfflineFeaturesDataset, self).__init__(root_dir=root_dir, mode=mode, patient_id=patient_id)


    def _load_data(self):
        """
        Loads all data into RAM during initialization.

        Returns:
            list: A list of tuples where each tuple contains a segment tensor and a label tensor.
        """
        data = []
        pbar = tqdm(self.file_list)
        pbar.set_description("Loading dataset...")
        for file_path in pbar:
            pbar.set_description(f"Loading file: {file_path}")
            npz_file = np.load(file_path)
            segment = torch.from_numpy(npz_file['x'].real)
            
                
            segment = segment.flatten()

            label = torch.from_numpy(npz_file['y'].astype(np.float32))

            data.append((segment, label))

        return data
    

    def __getitem__(self, idx):
        """
        Returns the segment, label, and loading time for the specified index.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            tuple: (segment, label, load_time)
        """
        data = self.data[idx]
        return data
    
    
    
class MNISTDataset(Dataset):
    def __init__(self, images_file, labels_file, transform=None):
        with open(labels_file, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            self.labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_file, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            self.images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(self.labels), 784)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28).astype(np.float32) / 255
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label
    
    
class DummyDataset(Dataset):
    def __init__(self, num_classes, n_samples_per_class, x, y=None, z=None, seed=42, is_non_linear=True, noise_std=0.1):
        self.num_classes = num_classes
        self.n_samples_per_class = n_samples_per_class
        self.x = x
        self.y = y
        self.z = z
        self.seed = seed
        self.noise_std = noise_std
        self.is_non_linear = is_non_linear
        
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.dimensions = self._get_dimensions()
        self.data, self.labels = self._generate_data()

    def _get_dimensions(self):
        dims = [self.x]
        if self.y is not None:
            dims.append(self.y)
        if self.z is not None:
            dims.append(self.z)
        return tuple(dims)

    def _generate_data(self):
        data = []
        labels = []

        for class_idx in range(self.num_classes):
            # Generate initial random data
            class_samples = torch.randn(self.n_samples_per_class, *self.dimensions)
            
            # Apply non-linear transformation
            if self.is_non_linear: 
                class_samples = self._apply_nonlinear_transform(class_samples, class_idx)
            else: 
                offset = class_idx * torch.ones(self.dimensions)
                class_samples += offset
                
            data.append(class_samples)
            labels.extend([class_idx] * self.n_samples_per_class)

        data = torch.cat(data, dim=0)
        labels = torch.tensor(labels)

        return data, labels

    def _apply_nonlinear_transform(self, samples, class_idx):
        # Apply sine wave transformation
        freq = 1 + class_idx * 0.5  # Different frequency for each class
        amplitude = 2 + class_idx * 0.5  # Different amplitude for each class
        
        # Apply transformation to first dimension
        samples[:, 0] = amplitude * torch.sin(freq * samples[:, 0])
        
        # If more than one dimension, apply cosine to second dimension
        if samples.shape[1] > 1:
            samples[:, 1] = amplitude * torch.cos(freq * samples[:, 1])
        
        # Add some noise to make it more challenging
        samples += torch.randn_like(samples) * self.noise_std
        
        return samples

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
# ============================== UTILITIES ==============================


def load_mnist_data(train_images_file, train_labels_file, test_images_file, test_labels_file, batch_size=32, num_workers=4):
    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load the training data
    train_dataset = MNISTDataset(
        images_file=train_images_file,
        labels_file=train_labels_file,
        transform=transform
    )

    # Load the test data
    test_dataset = MNISTDataset(
        images_file=test_images_file,
        labels_file=test_labels_file,
        transform=transform
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

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
   ds1 = OnlineSegmentsDataset("../data/preprocessed/", mode='train', patient_id='01')
   ds2 = OnlineSegmentsDataset("../data/preprocessed/", mode='test', patient_id='01')
