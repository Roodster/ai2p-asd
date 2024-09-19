# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import seizure_data_processing as sdp


def apply_bandpass_filter(eeg_data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to each channel in an EEG dataset.

    :param eeg_data: 2D numpy array of shape (channels, samples)
    :param lowcut: Lower cutoff frequency for the bandpass filter
    :param highcut: Upper cutoff frequency for the bandpass filter
    :param fs: Sampling frequency of the signal
    :param order: Order of the Butterworth filter (default is 5)
    :return: 2D numpy array of filtered EEG data of shape (channels, samples)
    """
    
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    threshold=2

    # Design the Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')

    # Initialize an array to store the filtered data
    filtered_data = np.zeros_like(eeg_data)

    # Loop over each channel and apply the filter
    for channel in range(eeg_data.shape[0]):  # Loop over channels
        filtered_data[channel] = signal.filtfilt(b, a, eeg_data[channel])
    
    return filtered_data



def plot_scatter_plot(eeg_data, channel):
    subsample_size = 5000  # Select a smaller subset, e.g., 5000 points
    samples = np.arange(eeg_data.shape[1]) 
    indices = np.random.choice(len(samples), subsample_size, replace=False)

    plt.scatter(samples[indices], eeg_data[channel][indices], s=2)
    plt.title(f'Subsampled Scatter Plot (Channel {channel})')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude (µV)')
    plt.show()


def normalize_eeg_data(eeg_data):
    """
    Normalize EEG data for each channel to have a mean of 0 and standard deviation of 1.
    
    :param eeg_data: 2D numpy array of shape (channels, num_samples)
    :return: Normalized EEG data of the same shape
    """
    # Initialize an array to store the normalized data
    normalized_data = np.zeros_like(eeg_data)
    
    # Normalize each channel independently
    for channel in range(eeg_data.shape[0]):
        mean = np.mean(eeg_data[channel])
        std = np.std(eeg_data[channel])
        normalized_data[channel] = (eeg_data[channel] - mean) / std
    
    return normalized_data


def remove_noise(eeg_data, fs, labels):
    """
    Apply noise reduction steps to the EEG data, including bandpass filtering, clipping extreme values (commented out), 
    and normalization. The function returns the processed EEG data and corresponding labels.
    
    :param eeg_data: 2D numpy array of shape (channels, num_samples) - Raw EEG data from multiple channels.
    :param fs: Integer - Sampling frequency of the EEG data in Hz.
    :param labels: 1D numpy array of size (num_samples) - Labels corresponding to each sample (time point) in the EEG data.
    
    :return:
        - normalized_data: 2D numpy array of shape (channels, num_samples) - Normalized EEG data after filtering.
        - labels: 1D numpy array of labels corresponding to the valid samples from the EEG data.
    """
    lowcut = 1.6
    highcut = 40.0
    filt_data = apply_bandpass_filter(eeg_data, lowcut, highcut, fs)
    print("Bandpass filtering done")

    normalized_data = normalize_eeg_data(filt_data)
    print("Normalization done ")

    return normalized_data


def segment_and_spectrogram(eeg_data, labels, fs, segment_length=1, overlap=0.5):
    """
    Segment EEG data into clips and convert to spectrograms.
    
    :param eeg_data: numpy array of shape (n_channels, n_samples)
    :param labels: numpy array of shape (n_samples,)
    :param fs: sampling frequency in Hz
    :param segment_length: length of each segment in seconds
    :param overlap: overlap between segments in seconds
    :return: tuple of (spectrograms, segment_labels)
    """
    n_channels, n_samples = eeg_data.shape
    
    # Calculate number of samples in each segment and step size
    samples_per_segment = int(segment_length * fs)
    step = int((segment_length - overlap) * fs)
    
    # Calculate number of segments
    n_segments = (n_samples - samples_per_segment) // step + 1
    
    # Initialize arrays to store spectrograms and labels
    spectrograms = []
    segment_labels = []
    for i in range(n_segments):

        start = i * step
        end = start + samples_per_segment
        
        # Extract segment
        segment = eeg_data[:, start:end]
        # Compute spectrogram for each channel
        segment_spectrograms = []
        for channel in range(n_channels):
            # Compute spectrogram on filtered data
            f, t, Sxx = signal.spectrogram(segment[channel], fs=fs, nperseg=samples_per_segment)
            segment_spectrograms.append(Sxx)
        
        # Stack channel spectrograms
        stacked_spectrogram = np.stack(segment_spectrograms, axis=0)
        spectrograms.append(stacked_spectrogram)
        
        # Determine label for the segment (majority vote)
        segment_label = int(np.sum(labels[start:end] > 0) > (end - start) / 2)
        segment_labels.append(segment_label)
    
    print('segment labels', segment_labels)
    return np.array(spectrograms), np.array(segment_labels)


# -

def load_eeg_file():
    # Load a single file
    file = (
            r".\ai2p-asd\data\chb13_21.edf")
    channels = "FP1-F7;F7-T7;T7-P7;P7-O1;FP1-F3;F3-C3;C3-P3;P3-O1;FP2-F4;F4-C4;C4-P4;P4-O2;FP2-F8;F8-T8;T8-P8;T8-P8;P8-O2;FZ-CZ;CZ-PZ;P7-T7;T7-FT9;FT9-FT10;FT10-T8;T8-P8".split(";")
    eeg_file = sdp.EEG(file, channels=channels)
    
    return eeg_file


def save_spectrograms_and_labels(spectrograms, labels, save_dir):
    """
    Save spectrograms and labels to .npy files.
    
    :param spectrograms: numpy array of spectrograms
    :param labels: numpy array of labels
    :param save_dir: directory to save the files
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    np.savez(os.path.join(save_dir, 'spectrograms'), x=spectrograms, y=labels, allow_pickle=True)
    print(f"Saved spectrograms and labels to {save_dir}")


def load_spectrograms_and_labels(load_dir):
    """
    Load spectrograms and labels from .npy files.
    
    :param load_dir: directory containing the .npy files
    :return: tuple of (spectrograms, labels)
    """
    data = np.load(os.path.join(load_dir, 'spectrograms.npz'), allow_pickle=True)
    print(f"Loaded spectrograms and labels from {load_dir}")

    return data# Example usage


if __name__ == "__main__":
    # Assume you have loaded your EEG data and labels
    # eeg_data shape: (n_channels, n_samples)
    # labels shape: (n_samples,)
    
    # Example data (replace with your actual data)
    eeg_file = load_eeg_file()
    # eeg_data = np.random.randn(n_channels, n_samples)
    # labels = np.random.randint(0, 2, n_samples)
    
    eeg_data = eeg_file.data
    labels = eeg_file.get_labels()
    
    
    fs = 256  # Sampling frequency in Hz

    denoised_data = remove_noise(eeg_data, fs, labels)
    # Segment and convert to spectrograms
    spectrograms, segment_labels = segment_and_spectrogram(denoised_data, labels, fs)
    
    print(f"Number of segments: {len(spectrograms)}")
    print(f"Spectrogram shape: {spectrograms[0].shape}")
    print(f"Segment labels shape: {segment_labels.shape}")
    

# Visualize a spectrogram for one channel of the first segment
def plot_spectrogram(X):
    plt.figure(figsize=(10, 4))
    plt.specgram(X[0][0], Fs=256, cmap="rainbow")
    plt.colorbar(label='Log Power')
    plt.title(f'Spectrogram of Channel 0, Segment 0 (Label: {segment_labels[0]})')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()


# plt.specgram(spectrograms[0][0], Fs=6, cmap="rainbow")
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()


save_spectrograms_and_labels(spectrograms, segment_labels, save_dir="./data/temp/")


data = load_spectrograms_and_labels("./data/temp/")
X = data['x']
y = data['y']
print(X.shape)
print(y.shape)


plot_spectrogram(X)

# +
print(type(X))
print(type(y))


# -

import torch as th

print(th.from_numpy(spectrograms.astype(np.complex128)).float())


