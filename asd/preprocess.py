import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy import signal
import random as rnd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import librosa as lib
import librosa.feature as libf


class BandpassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=256, lowcut=0, highcut=128, order=6):
        """
        Custom sklearn transformer that applies a bandpass filter to EEG data.

        Parameters:
        sfreq (float): Sampling frequency of the EEG data.
        lowcut (float): Lower frequency bound (default 0 Hz).
        highcut (float): Upper frequency bound (default 128 Hz).
        order (int): Order of the filter (default 6).
        """
        self.sfreq = sfreq
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def fit(self, X, y=None):
        # No fitting necessary for this transformer, so just return self
        return self

    def transform(self, X, y=None):
        """
        Apply the bandpass filter to the EEG data.

        Parameters:
        X (numpy.ndarray): EEG data array (channels x samples).

        Returns:
        numpy.ndarray: Filtered EEG data.
        """
        
        X, y = (X)
        nyq = 0.5 * self.sfreq
        low = (self.lowcut / nyq) + 1e-6
        high = self.highcut / nyq - 1e-6
        sos = signal.butter(self.order, [low, high], analog=False, btype='band', output='sos')
        filtered_data = signal.sosfilt(sos, X)

        return (filtered_data, y)
    
    
class SegmentSignals(BaseEstimator, TransformerMixin):
    def __init__(self, fs=256, segment_length=1, overlap=0.5, mode='default'):
        """
        Segments EEG data into clips.

        Parameters:
        fs (float): Sampling frequency in Hz.
        segment_length (float): Length of each segment in seconds.
        overlap (float): Overlap between segments in seconds.
        """
        self.fs = fs
        self.segment_length = segment_length
        self.overlap = overlap

    def fit(self, X, y=None):
        # No fitting necessary for this transformer, so just return self
        return self

    def transform(self, X):
        """
        Segment EEG data into clips.

        Parameters:
        X (numpy.ndarray): EEG data array of shape (n_channels, n_samples).
        y (numpy.ndarray): Labels array of shape (n_samples,).

        Returns:
        tuple: (segmented_data, segment_labels)
        """
        eeg_data, labels = X
        n_channels, n_samples = eeg_data.shape
        samples_per_segment = int(self.segment_length * self.fs)
        step = int((self.segment_length - self.overlap) * self.fs)
        n_segments = (n_samples - samples_per_segment) // step + 1

        segmented_data = []
        segment_labels = []

        for i in range(n_segments):
            start = i * step
            end = start + samples_per_segment
            segment = eeg_data[:, start:end]
            segmented_data.append(segment)

            # Determine label for the segment (majority vote)
            segment_label = int(np.sum(labels[start:end] > 0) > (end - start) / 2)
            segment_labels.append(segment_label)
        return np.array(segmented_data), np.array(segment_labels)
        
    
class Spectrograms(BaseEstimator, TransformerMixin):
    def __init__(self, fs=256, nperseg=None, noverlap=None, window='tukey', use_multithreading=False, max_workers=None):
        """
        Converts segments of EEG data into spectrograms.

        Parameters:
        fs (float): Sampling frequency in Hz.
        nperseg (int): Number of samples per segment for the spectrogram.
        noverlap (int): Number of samples to overlap between segments for the spectrogram.
        use_multithreading (bool): Whether to use multithreading for processing channels.
        max_workers (int): Maximum number of worker threads. If None, uses the number of CPU cores.
        """
        self.fs = fs
        self.nperseg = int(nperseg * fs)
        self.noverlap = int(noverlap * fs)
        self.use_multithreading = use_multithreading
        self.max_workers = max_workers if max_workers is not None else multiprocessing.cpu_count()
        self.window = int(fs) // 4
    
    def fit(self, X, y=None):
        # No fitting necessary for this transformer, so just return self
        return self
    
    def _process_channel(self, channel_data):
        """Process a single channel of data."""
        # f, t, Sxx = signal.spectrogram(channel_data, fs=self.fs,
        #                                nperseg=self.nperseg,
        #                                noverlap=self.noverlap, 
        #                                window=self.window,
        #                                axis=1)
        S = libf.melspectrogram(y=channel_data, sr=self.fs, n_mels=256)

        S_db = lib.power_to_db(S)
        # Normalize the spectrogram to the range [0, 1]
        S_norm = np.average((S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db)), axis=3)

        # Expand to 1 grayscale channel (optional step to keep the dimensions consistent)
        S_gray = np.expand_dims(S_norm, axis=1)  # Shape becomes (1, height, width)

        return S_gray
        
    def transform(self, X, y=None):
        """
        Generate spectrograms for each EEG segment.

        Parameters:
        X (numpy.ndarray): Segmented EEG data array of shape (n_segments, n_channels, n_samples).

        Returns:
        numpy.ndarray: Array of spectrograms of shape (n_segments, n_channels, n_frequencies, n_times).
        """
        segmented_data, y = X
        spectrograms = self._process_channel(channel_data=segmented_data)
        return (np.array(spectrograms), y)
    
    
    
class ZScoreNormalization(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Custom sklearn transformer that applies z-score normalization to data.
        """
        pass

    def fit(self, X, y=None):
        # No fitting necessary for this transformer, so just return self
        return self

    def transform(self, X):
        """
        Apply z-score normalization to the data.

        Parameters:
        X (numpy.ndarray): Data array (channels x samples).

        Returns:
        numpy.ndarray: Normalized data.
        """
        
        X,y = X
        # Ensure data is of type float32 for precision
        X = X.astype(np.float32)
        
        # Compute mean and standard deviation along the channels axis
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        
        # Perform z-score normalization
        normalized_data = (X - mean) / std
        
        return (normalized_data, y)
    
    
class BalanceSeizureSegments(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=None):
        """
        Custom sklearn transformer that balances seizure and non-seizure segments
        by randomly sampling non-seizure segments.

        Parameters:
        random_state (int, optional): Seed for the random number generator for reproducibility.
        """
        self.random_state = random_state

    def fit(self, X, y=None):
        # No fitting necessary for this transformer, so just return self
        return self

    def transform(self, X, y=None):
        """
        Balance the number of seizure and non-seizure segments.

        Parameters:
        X (list): List of tuples, where each tuple contains (EEG segment, label, eeg_id).

        Returns:
        list: Balanced list of segments with equal number of seizure and non-seizure segments.
        """
        
        # Set the random seed if specified
        if self.random_state is not None:
            rnd.seed(self.random_state)
            
        X = list(zip(X[0], X[1]))

        # Separate seizure and non-seizure segments
        seizure_segments = [segment for segment in X if segment[1] == 1]
        non_seizure_segments = [segment for segment in X if segment[1] == 0]
        
        # Number of seizure segments
        num_seizures = len(seizure_segments)
        
        # Randomly sample non-seizure segments to match the number of seizure segments
        sampled_non_seizure = rnd.sample(non_seizure_segments, min(num_seizures, len(non_seizure_segments)))

        # Combine seizure and sampled non-seizure segments
        balanced_segments = seizure_segments + sampled_non_seizure

        # Shuffle the combined segments
        segments = np.array([signal for signal, _ in balanced_segments])
        labels = np.array([label for _, label in balanced_segments])
    
        return (segments, labels) 
    
    
    
    # Example usage
if __name__ == "__main__":

    from tqdm import tqdm
    from common.utils import load_edf_filepaths, clean_path, load_eeg_file, save_spectrograms_and_labels
    
    
    dataset_path = "./data/dataset/train/raw/temp"
    dataset_save_root_path = "./data/dataset/teste/spectrograms/"
    
    files_list = load_edf_filepaths(dataset_path)
    
    pbar = tqdm(files_list)
    for file in pbar: 
        # pbar.set_description(f"current file {file}")

        # channels="FZ-CZ;CZ-PZ;F8-T8;P4-O2;FP2-F8;F4-C4;C4-P4;P3-O1;FP2-F4;F3-C3;C3-P3;P7-O1;FP1-F3;F7-T7;T7-P7;FP1-F7"
        eeg_file = load_eeg_file(file)

        if eeg_file == None:
            continue
        
        eeg_data = eeg_file.data
        labels = eeg_file.get_labels()
        
        
        pipeline = Pipeline([('filters', BandpassFilter(sfreq=256,lowcut=1, highcut=40, order=6)),
                             ('normalizes',ZScoreNormalization()),
                             ('segments', SegmentSignals(fs=256, segment_length=4, overlap=0)),
                             ('spectrograms', Spectrograms(fs=256, nperseg=4, noverlap=0))
                             ]) 
        
        
        X, y = pipeline.transform(X=(eeg_data, labels))
        save_dir, filename = clean_path(file, dataset_path)
        save_dir = dataset_save_root_path + save_dir
        filename = filename.split('.')[0]        
        pbar.set_description(f"saved {filename} of size: {X.shape} to {save_dir}")
                

            
        save_spectrograms_and_labels(X, y, save_dir=save_dir, filename=filename)
    