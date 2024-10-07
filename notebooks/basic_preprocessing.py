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
import torch
from imblearn.over_sampling import BorderlineSMOTE


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

    
def plot_channel_scatter(data, channel_idx, label):
    
    # Select the data for the given channel and remove the last singleton dimension
    channel_data = data[channel_idx, :].cpu().numpy()  # Convert to numpy for plotting

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(channel_data)), channel_data, c='blue', alpha=0.7)
    plt.title(f"Scatter Plot for Channel {channel_idx}, label, {label}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
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



def load_npz_files():
    root_dir = r".\output_spectrograms"  # Root directory with the npz files
    seiz_segments = []
    non_seiz_segments = []
    seiz_labels = []
    non_seiz_labels = []
    cnt  = 0
    # Walk through the root directory and subdirectories
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            cnt += 1
            if file.endswith(".npz"):
                file_path = os.path.join(subdir, file)
                try:
                    # Load the npz file
                    npz_file = np.load(file_path)
                    segment = torch.from_numpy(npz_file['x'].astype(np.float32))
                    label = torch.from_numpy(npz_file['y'].astype(np.float32))  # Ensure labels are float for comparison
                    print(segment.shape)
                    plot_channel_scatter(segment, 0, label)
                    # Append the segment and label to appropriate lists
                    if label.item() == 1.0:  # Seizure segments
                        seiz_segments.append(segment)
                        seiz_labels.append(label)
                    else:  # Non-seizure segments
                        non_seiz_segments.append(segment)
                        non_seiz_labels.append(label)
                
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    print("count, ", cnt)
    # Convert lists to tensors
    torch_seiz = torch.stack(seiz_segments)
    torch_non_seiz = torch.stack(non_seiz_segments)
    torch_seiz_labels = torch.stack(seiz_labels)
    torch_non_seiz_labels = torch.stack(non_seiz_labels)
    
    # Shuffle seizure and non-seizure segments along with their labels
    seiz_indices = torch.randperm(torch_seiz.size(0))
    non_seiz_indices = torch.randperm(torch_non_seiz.size(0))

    shuffled_seiz = torch_seiz[seiz_indices]
    shuffled_seiz_labels = torch_seiz_labels[seiz_indices]
    
    shuffled_non_seiz = torch_non_seiz[non_seiz_indices]
    shuffled_non_seiz_labels = torch_non_seiz_labels[non_seiz_indices]

    # Print counts and shapes
    print("Count of seizures: ", len(seiz_segments))
    print("Count of non-seizures: ", len(non_seiz_segments))
    print("Seizures shape: ", shuffled_seiz.shape)
    print("Non-seizures shape: ", shuffled_non_seiz.shape)

    # Return both shuffled segments and their respective labels
    return (shuffled_seiz, shuffled_seiz_labels), (shuffled_non_seiz, shuffled_non_seiz_labels)



def apply_SMOTE(shuffled_seiz, shuffled_seiz_labels, shuffled_non_seiz, shuffled_non_seiz_labels):
    # Step 1: Stack the channels for both seizure and non-seizure segments
    stacked_seiz = shuffled_seiz.view(shloaduffled_seiz.size(0), -1)
    stacked_non_seiz = shuffled_non_seiz.view(shuffled_non_seiz.size(0), -1)

    # Step 2: Remove half of the seizure segments and their labels
    seiz_half_idx = shuffled_seiz.size(0) // 2  # Find half of the seizure samples
    reduced_seiz = stacked_seiz[seiz_half_idx:]  # Keep only the second half of the seizure segments
    reduced_seiz_labels = shuffled_seiz_labels[seiz_half_idx:]  # Keep only the second half of the seizure labels

    # Step 3: Combine the reduced seizure data with non-seizure data
    combined_data = torch.cat((reduced_seiz, stacked_non_seiz), dim=0)
    combined_labels = torch.cat((reduced_seiz_labels, shuffled_non_seiz_labels), dim=0)

    # Step 4: Convert to numpy arrays for SMOTE
    combined_data_np = combined_data.numpy()
    combined_labels_np = combined_labels.numpy()

    # Step 5: Apply BorderlineSMOTE
    blsmote = BorderlineSMOTE(sampling_strategy="minority") 
    X_resampled, y_resampled = blsmote.fit_resample(combined_data_np, combined_labels_np)

    # Step 6: Reshape the resampled data from (N, C * B) back to (N, C, B)
    # N: number of samples, C: number of channels, B: number of features
    N = X_resampled.shape[0]
    C = shuffled_seiz.size(1)  # Number of channels
    B = shuffled_seiz.size(2)  # Number of features (time samples)
    
    X_resampled_reshaped = X_resampled.reshape(N, C, B)  # Reshape back to (N, C, B)

    # Convert the reshaped data and labels back to torch tensors
    X_resampled_torch = torch.from_numpy(X_resampled_reshaped).float()
    y_resampled_torch = torch.from_numpy(y_resampled).long()

    # Print the shapes for confirmation
    print(f"Resampled data shape (torch): {X_resampled_torch.shape}")
    print(f"Resampled labels shape (torch): {y_resampled_torch.shape}")
    
    print("old y = ", combined_labels_np)
    print("new y = ", y_resampled)

    # Return the resampled data and labels as PyTorch tensors
    return X_resampled_torch, y_resampled_torch


# Visualize a spectrogram for one channel of the first segment
def plot_spectrogram(X, label):
    print("Forma beledir! ", X.shape)
    plt.figure(figsize=(10, 4))
    plt.specgram(X[0][0], Fs=256, cmap="rainbow")
    plt.colorbar(label='Log Power')
    plt.title(f'Spectrogram of Channel 0, Segment 0 (Label: {label})')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == "__main__":
    # Assume you have loaded your EEG data and labels
    # eeg_data shape: (n_channels, n_samples)
    # labels shape: (n_samples,)
    
    (shuffled_seiz, shuffled_seiz_labels), (shuffled_non_seiz, shuffled_non_seiz_labels) = load_npz_files()
    X, Y = apply_SMOTE(shuffled_seiz, shuffled_seiz_labels, shuffled_non_seiz, shuffled_non_seiz_labels)

    print(torch.sum(Y[0:10951]))
    print(torch.sum(Y[10952:32856]))
    print(torch.sum(Y[32857:43807]))

    plot_channel_scatter(X[1222], 0)
    plot_channel_scatter(X[122], 1)
    plot_channel_scatter(X[2], 2)

    plot_channel_scatter(X[43605], 0)
    plot_channel_scatter(X[43704], 0)
    plot_channel_scatter(X[43507], 0)
    plot_channel_scatter(X[43403], 0)
    plot_channel_scatter(X[43201], 0)
    plot_channel_scatter(X[41802], 0)

    plot_channel_scatter(X[41802], 3)
    plot_channel_scatter(X[41802], 2)
    plot_channel_scatter(X[41802], 1)
    plot_channel_scatter(X[41802], 3)
    plot_channel_scatter(X[41802], 2)

    
    # # Example data (replace with your actual data)
    # eeg_file = load_eeg_file()
    # # eeg_data = np.random.randn(n_channels, n_samples)
    # # labels = np.random.randint(0, 2, n_samples)
    
    # eeg_data = eeg_file.data
    # labels = eeg_file.get_labels()
    
    
    # fs = 256  # Sampling frequency in Hz

    # denoised_data = remove_noise(eeg_data, fs, labels)
    # # Segment and convert to spectrograms
    # spectrograms, segment_labels = segment_and_spectrogram(denoised_data, labels, fs)
    
    # print(f"Number of segments: {len(spectrograms)}")
    # print(f"Spectrogram shape: {spectrograms[0].shape}")
    # print(f"Segment labels shape: {segment_labels.shape}")
    



# plt.specgram(spectrograms[0][0], Fs=6, cmap="rainbow")
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()


# def apply_SMOTE(segments):
    
# save_spectrograms_and_labels(spectrograms, segment_labels, save_dir="./data/temp/")


# data = load_spectrograms_and_labels("./data/temp/")
# X = data['x']
# y = data['y']
# print(X.shape)
# print(y.shape)

# print("Y budur qardash ", y)




# +
print(type(X))
print(type(y))


# -

import torch as th

print(th.from_numpy(spectrograms.astype(np.complex128)).float())


