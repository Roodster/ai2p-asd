import yaml
import os
import glob
import random as rnd
import numpy as np
import torch as th
import seizure_data_processing as sdp



def load_config(config_file=None):
    assert config_file is not None, "Error: config file not found."
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def parse_args(args_file):
    config = load_config(config_file=args_file)
    return config


def clean_path(full_path, substring_to_remove):
    # Remove the substring
    cleaned_path = full_path.replace(substring_to_remove, '')
    
    # Split the path into parts
    parts = cleaned_path.split(os.sep)
    
    # Remove the last part (filename)
    filename = parts[-1]
    
    
    parts = parts[:-1][0]
    
    
    return parts, filename


def load_spectrograms_and_labels(load_dir, filename):
    """
    Load spectrograms and labels from .npy files.
    
    :param load_dir: directory containing the .npy files
    :return: tuple of (spectrograms, labels)
    """
    data = np.load(os.path.join(load_dir, ), allow_pickle=True)

    return data


def load_seizure_edf_filepaths(root_path):
    edf_files = []
    
    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Find all .edf files in the current directory
        for filename in glob.glob(os.path.join(dirpath, '*.edf')):
            # Check if a corresponding .edf.seizures file exists
            seizure_file = filename + '.seizures'
            if os.path.exists(seizure_file):
                edf_files.append(r'{}'.format(filename))
    return edf_files


def load_edf_filepaths(root_path):
    edf_files = []
    
    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Find all .edf files in the current directory
        for filename in glob.glob(os.path.join(dirpath, '*.edf')):
            edf_files.append(r'{}'.format(filename))
    return edf_files


def load_eeg_file(file, channels="F7-T7;T7-P7;F8-T8;T8-P8"):
    # Load a single file
    channels = channels.split(";")
    try:
        eeg_file = sdp.EEG(file, channels=channels)
    except:
        eeg_file = None
    
    return eeg_file


def save_spectrograms_and_labels(spectrograms, labels, save_dir, filename):
    """
    Save spectrograms and labels to .npy files.
    
    :param spectrograms: numpy array of spectrograms
    :param labels: numpy array of labels
    :param save_dir: directory to save the files
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for index, (spectrogram, label) in enumerate(zip(spectrograms, labels)):
        a = filename.split('.')[0]
        save_file = f"{a}_{index}.npz"
         
        np.savez(os.path.join(save_dir, save_file.replace('+', '')), x=spectrogram, y=label, allow_pickle=True)
        
        
def save_signals_and_labels(X, y, save_dir, filename):
    """
    Save signals and labels to .npz files.
    
    :param X: numpy array of signals with shape (N, C, F)
    :param y: numpy array of labels with shape (F)
    :param save_dir: directory to save the files
    :param filename: base name for the saved files
    """
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Iterate over each signal (N dimension) and save with corresponding labels
    for index in range(X.shape[0]):
        signal = X[index]  # Shape (C, F)
        
        # Construct the file name with the index
        save_file = f"{filename.split('.')[0]}_{index}.npz"
        
        # Save the signal and labels using the keys 'x' and 'y'
        np.savez(os.path.join(save_dir, save_file), x=signal, y=y, allow_pickle=True)
        
    print(f"Saved {X.shape[0]} files in {save_dir}")



def set_seed(seed):
    """
    For seed to some modules.
    :param seed: int. The seed.
    :return:
    """
    th.manual_seed(seed)
    np.random.seed(seed)
    rnd.seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    th.Generator().manual_seed(seed)
    
    
def freeze_params(model):
    
    for param in model.parameters():
        param.required_grad = False
        
    return model
    