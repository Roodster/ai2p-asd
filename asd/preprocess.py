import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy import signal
import random as rnd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import librosa as lib
import librosa.feature as libf
from scipy import stats, signal


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
        self.window = int(fs) // 16
    
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
        
        
        
        S = libf.melspectrogram(y=channel_data, sr=self.fs, win_length=self.window, hop_length=self.window, n_fft=self.window, n_mels=self.fs)
        # S_db = lib.power_to_db(S)
        # # Normalize the spectrogram to the range [0, 1]
        # S_norm = np.average((S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db)), axis=1)
        
        # # Expand to 1 grayscale channel (optional step to keep the dimensions consistent)
        # S_gray = np.expand_dims(S_norm, axis=1)  # Shape becomes (1, height, width)

        # return S_gray.transpose(0, 1, 3, 2)[:, :, :-1, :]
        print(S.shape)
        return S
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
    
class RMSAmplitudeFilter(BaseEstimator, TransformerMixin):
    def __init__(self, min_amplitude=0.5, max_amplitude=150):
        """
        Custom sklearn transformer that removes windows with RMS amplitude
        outside the specified range.

        Parameters:
        min_amplitude (float): Minimum RMS amplitude in microvolts (default: 0.5)
        max_amplitude (float): Maximum RMS amplitude in microvolts (default: 150)
        """
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def fit(self, X, y=None):
        # No fitting necessary for this transformer
        return self

    def transform(self, X):
        """
        Remove windows with RMS amplitude outside the specified range.

        Parameters:
        X (tuple): Tuple containing (data, labels)
            data (numpy.ndarray): Data array (windows x channels x samples)
            labels (numpy.ndarray): Labels array

        Returns:
        tuple: Filtered data and corresponding labels
        """
        data, labels = X

        # Ensure data is of type float32 for precision
        data = data.astype(np.float32)

        # Calculate RMS amplitude for each window
        rms_amplitude = np.sqrt(np.mean(np.square(data), axis=(1, 2)))

        # Create a mask for windows within the specified amplitude range
        mask = (rms_amplitude >= self.min_amplitude) & (rms_amplitude <= self.max_amplitude)

        # Apply the mask to both data and labels
        filtered_data = data[mask]
        filtered_labels = labels[mask]

        return (filtered_data, filtered_labels)

class EEGFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, sampling_rate=256):
        """
        Custom sklearn transformer that extracts features from EEG data and applies normalization.

        Parameters:
        sampling_rate (int): Sampling rate of the EEG data in Hz (default: 256)
        """
        self.sampling_rate = sampling_rate
        self.feature_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Extract features and apply normalization to the EEG data.

        Parameters:
        X (tuple): Tuple containing (data, labels)
            data (numpy.ndarray): Data array (windows x channels x samples)
            labels (numpy.ndarray): Labels array

        Returns:
        tuple: Extracted features and corresponding labels
        """
        data, labels = X
        features = []

        for window in data:
            window_features = []
            
            # Time domain features
            window_features.extend(self._time_domain_features(window))
            
            # Frequency domain features
            window_features.extend(self._frequency_domain_features(window))
            
            # Entropy-derived features
            window_features.extend(self._entropy_features(window))
            
            # Asymmetry features
            window_features.extend(self._asymmetry_features(window))
            
            features.append(window_features)

        features = np.array(features)
        
        # Apply normalization
        normalized_features = self._normalize_features(features)

        if not self.feature_names:
            self.feature_names = [
                "Zero crossings", "Line length", "Skewness", "Kurtosis",
                "Root mean square amplitude", "Total power", "Peak frequency",
                "Mean (0-4 Hz)", "Mean (4-8 Hz)", "Mean (8-13 Hz)", "Mean (13-30 Hz)", "Mean (30-80 Hz)",
                "Power (0-4 Hz)", "Power (4-8 Hz)", "Power (8-13 Hz)", "Power (13-30 Hz)", "Power (30-80 Hz)",
                "Sample entropy", "Shannon entropy", "Spectral entropy",
                "Power asymmetry (0-4 Hz)", "Power asymmetry (4-8 Hz)", "Power asymmetry (8-13 Hz)",
                "Power asymmetry (13-30 Hz)", "Power asymmetry (30-80 Hz)"
            ]

        return (normalized_features, labels)

    def _time_domain_features(self, window):
        zero_crossings = np.sum(np.diff(np.sign(window)) != 0, axis=1)
        line_length = np.sum(np.abs(np.diff(window, axis=1)), axis=1)
        skewness = stats.skew(window, axis=1)
        kurtosis = stats.kurtosis(window, axis=1)
        rms = np.sqrt(np.mean(np.square(window), axis=1))
        return np.concatenate([zero_crossings, line_length, skewness, kurtosis, rms])

    def _frequency_domain_features(self, window):
        freqs, psd = signal.welch(window, fs=self.sampling_rate, nperseg=min(256, window.shape[1]))
        total_power = np.sum(psd, axis=1)
        peak_freq = freqs[np.argmax(psd, axis=1)]
        
        freq_bands = [(0, 4), (4, 8), (8, 13), (13, 30), (30, 80)]
        mean_power = []
        band_power = []
        
        for low, high in freq_bands:
            band_mask = (freqs >= low) & (freqs < high)
            mean_power.extend(np.mean(psd[:, band_mask], axis=1))
            band_power.extend(np.sum(psd[:, band_mask], axis=1))
        
        return np.concatenate([total_power, peak_freq, mean_power, band_power])

    def _entropy_features(self, window):
        sample_entropy = np.apply_along_axis(self._sample_entropy, 1, window)
        shannon_entropy = stats.entropy(np.abs(window), axis=1)
        spectral_entropy = stats.entropy(np.abs(np.fft.fft(window)), axis=1)
        return np.concatenate([sample_entropy, shannon_entropy, spectral_entropy])

    def _asymmetry_features(self, window):
        left_channels = window[:len(window)//2]
        right_channels = window[len(window)//2:]
        
        freqs, left_psd = signal.welch(left_channels, fs=self.sampling_rate, nperseg=min(256, window.shape[1]))
        _, right_psd = signal.welch(right_channels, fs=self.sampling_rate, nperseg=min(256, window.shape[1]))
        
        freq_bands = [(0, 4), (4, 8), (8, 13), (13, 30), (30, 80)]
        asymmetry = []
        
        for low, high in freq_bands:
            band_mask = (freqs >= low) & (freqs < high)
            left_power = np.sum(left_psd[:, band_mask], axis=1)
            right_power = np.sum(right_psd[:, band_mask], axis=1)
            asymmetry.append((right_power - left_power) / (right_power + left_power))
        
        return asymmetry

    def _sample_entropy(self, x, m=2, r=0.2):
        n = len(x)
        r *= np.std(x)
        
        def _count_matches(template):
            return np.sum(np.max(np.abs(x[None, m:] - template[:, None]), axis=1) < r)
        
        A = np.mean([_count_matches(x[i:i+m]) for i in range(n-m)])
        B = np.mean([_count_matches(x[i:i+m+1]) for i in range(n-m)])
        
        return -np.log(B / A) if A > 0 and B > 0 else 0

    def _normalize_features(self, features):
        # Apply median decaying over time normalization
        time_constant = 0.1  # Adjust as needed
        normalized = np.zeros_like(features)
        running_median = np.median(features[0], axis=0)
        
        for i, window in enumerate(features):
            normalized[i] = (window - running_median) / (np.abs(running_median) + 1e-8)
            running_median = (1 - time_constant) * running_median + time_constant * np.median(window, axis=0)
        
        return normalized

    def get_feature_names(self):
        return self.feature_names
    
    
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
    dataset_save_root_path = "./data/dataset/train/visuals/"
    
    files_list = load_edf_filepaths(dataset_path)
    
    pbar = tqdm(files_list)
    print(files_list)
    for file in pbar: 
        # pbar.set_description(f"current file {file}")

        # channels="FZ-CZ;CZ-PZ;F8-T8;P4-O2;FP2-F8;F4-C4;C4-P4;P3-O1;FP2-F4;F3-C3;C3-P3;P7-O1;FP1-F3;F7-T7;T7-P7;FP1-F7"
        eeg_file = load_eeg_file(file)

        if eeg_file == None:
            print('its none')
            continue
        
        eeg_data = eeg_file.data
        labels = eeg_file.get_labels()
        
        
        pipeline = Pipeline([('filters', BandpassFilter(sfreq=256,lowcut=1, highcut=40, order=6)),
                             ('normalizes',ZScoreNormalization()),
                             ('segments', SegmentSignals(fs=256, segment_length=4, overlap=2)),
                             ('spectrograms', Spectrograms(fs=256, nperseg=4, noverlap=0)),
                             ('balancer', BalanceSeizureSegments())
                             ]) 
    
    
        
        X, y = pipeline.transform(X=(eeg_data, labels))

        print(X.shape)
        save_dir, filename = clean_path(file, dataset_path)
        save_dir = dataset_save_root_path + save_dir
        filename = filename.split('.')[0]        
        pbar.set_description(f"saved {filename} of size: {X.shape} to {save_dir}")
                

            
        save_spectrograms_and_labels(X, y, save_dir=save_dir, filename=filename)
    