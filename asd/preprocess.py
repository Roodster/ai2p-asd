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
from sklearn.preprocessing import normalize
from sklearn.impute import SimpleImputer
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from imblearn.over_sampling import BorderlineSMOTE
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from common.utils import load_edf_filepaths, clean_path, load_eeg_file, save_spectrograms_and_labels, save_signals_and_labels
from scipy.stats import mode



class MeanDownSampling(BaseEstimator, TransformerMixin):
    """
    Returns (new_X: downsampled EEG data, new_y: downsampled labels (obtained through majority vote))
    Processes data in chunks to avoid memory errors.
    """
    def __init__(self, down_freq=64, sfreq=256, chunk_size=1000000):
        self.down_freq = down_freq
        self.sfreq = sfreq
        self.chunk_size = chunk_size  # Process data in smaller chunks

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X, y = X
        # print("X shape before mean drop ", X.shape)

        step = int(self.sfreq / self.down_freq)
        num_channels, num_samples = X.shape
        
        # Initialize empty arrays to store results
        num_downsampled_samples = (num_samples // step)
        new_X = np.zeros((num_channels, num_downsampled_samples))
        new_y = np.zeros(num_downsampled_samples)

        # Process in chunks to reduce memory usage
        for start_idx in range(0, num_samples, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, num_samples)
            chunk_X = X[:, start_idx:end_idx]
            chunk_y = y[start_idx:end_idx]

            # Trim the chunk if necessary
            chunk_size = chunk_X.shape[1] - (chunk_X.shape[1] % step)
            chunk_X = chunk_X[:, :chunk_size]
            chunk_y = chunk_y[:chunk_size]

            # Downsample the chunk
            downsampled_X = chunk_X.reshape(num_channels, -1, step).mean(axis=2)
            downsampled_y = mode(chunk_y.reshape(-1, step), axis=1)[0].flatten()

            # Determine where to insert the downsampled chunk into new_X and new_y
            ds_start_idx = start_idx // step
            ds_end_idx = ds_start_idx + downsampled_X.shape[1]

            new_X[:, ds_start_idx:ds_end_idx] = downsampled_X
            new_y[ds_start_idx:ds_end_idx] = downsampled_y

        print("X shape after mean drop ", new_X.shape)
        return (new_X, new_y)
    

class BandpassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=256, lowcut=0, highcut=128, order=6):
        """
        Custom sklearn transformer that applies a bandpass filter to EEG data.

        Parameters:
        sfreq (float): Sampling frequency of the EEG data.
        lowcut (float): Lower frequency bound (default 0 Hz).ratio
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
        # print("X shape before segmenting ", eeg_data.shape)

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
            
        print("X shape after segmenting ", np.array(segmented_data).shape)
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
        
        
        
        S = libf.melspectrogram(y=channel_data.astype(np.float32), sr=self.fs, win_length=self.window, hop_length=self.window, n_fft=self.window, n_mels=self.fs)
        S_db = lib.power_to_db(S)
        # # Normalize the spectrogram to the range [0, 1]
        S_norm = np.average((S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db)), axis=1)
        
        # # Expand to 1 grayscale channel (optional step to keep the dimensions consistent)
        S_gray = np.expand_dims(S_norm, axis=1)  # Shape becomes (1, height, width)

        return S_gray.transpose(0, 1, 3, 2)[:, :, :-1, :]

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
    


class DropSegments(BaseEstimator, TransformerMixin):
    def __init__(self, drop_percentage=0.8):
        """
        Custom sklearn transformer that randomly drops a percentage of data.
        
        Parameters:
        drop_percentage (float): The percentage of data to drop (0 < drop_percentage < 1).
        """
        self.drop_percentage = drop_percentage

    def fit(self, X, y=None):
        # No fitting necessary for this transformer, so just return self
        return self
    
    def transform(self, X):
        """
        Randomly drop a percentage of the data.

        Parameters:
        X (tuple): Tuple containing:
            - X (numpy.ndarray): Data array with shape (N, C, F).
            - y (numpy.ndarray): Labels array with shape (N,).

        Returns:
        tuple: Tuple containing:
            - X (numpy.ndarray): Reduced data array.
            - y (numpy.ndarray): Corresponding reduced labels.
        """
        # print("Starting normalization and dropping segments.")
        
        X, y = X
        # Ensure data is of type float32 for precision
        X = X.astype(np.float32)
        
        # Calculate the number of segments to retain
        retain_percentage = 1 - self.drop_percentage
        num_retain = int(len(X) * retain_percentage)
        
        # Randomly select indices to retain
        retain_indices = np.random.choice(len(X), num_retain, replace=False)
        
        # Sort the indices to maintain order (optional)
        retain_indices = np.sort(retain_indices)
        
        # Select the retained samples
        X_reduced = X[retain_indices]
        y_reduced = y[retain_indices]
        
        print(f"Dropped {self.drop_percentage * 100}% of the segments. Retained {num_retain} out of {len(X)}.")
        print(f"Number of seizure samples: ", y_reduced.sum())

        return X_reduced, y_reduced
       

class ApplySMOTE(BaseEstimator, TransformerMixin):
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
        print("Starting normalizing")
        
        X, y = X
        # Ensure data is of type float32 for precision
        X = X.astype(np.float32)
        N, C, F = X.shape
        flattened = X.reshape(N, C * F)
        # Apply BorderlineSMOTE
        blsmote = BorderlineSMOTE(sampling_strategy=0.5)
        print(len(flattened), " ", y.sum())
        X_resampled, y_resampled = blsmote.fit_resample(flattened, y)

        # Reshape the resampled data from (N, C * B) back to (N, C, B)
        # N: number of samples, C: number of channels, F: number of features
        N = X_resampled.shape[0]
        X_resampled_reshaped = X_resampled.reshape(N, C, F)  # Reshape back to (N, C, F)
        # print("Size of dataset after SMOTE ", len(X_resampled_reshaped))
        return (X_resampled_reshaped, y)

         
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
        print("X shape before norm ", X.shape)
        # Ensure data is of type float32 for precision
        X = X.astype(np.float32)
        
        # Compute mean and standard deviation along the channels axis
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        
        # Perform z-score normalization
        normalized_data = (X - mean) / std
        
        # print("X shape after norm ", normalized_data.shape)

        return (normalized_data, y)



class RMSAmplitudeFilter(BaseEstimator, TransformerMixin):
    def __init__(self, min_amplitude=0.5, max_amplitude=150, return_id=False):
        """
        Custom sklearn transformer that removes windows with RMS amplitude
        outside the specified range.

        Parameters:
        min_amplitude (float): Minimum RMS amplitude in microvolts (default: 0.5)
        max_amplitude (float): Maximum RMS amplitude in microvolts (default: 150)
        """
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.return_id = return_id

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
        
        return (filtered_data, filtered_labels) if not self.return_id else (filtered_data, filtered_labels, mask)

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
        zero_crossings = slice(0,4)
        minimas = slice(4,8)
        maximas = slice(8,12)
        skewness = slice(12, 16)
        kurtosis = slice(16,20)
        rms = slice(20, 24)
        
        total_powers = slice(24, 28)
        peak_freqs = slice(28, 32)
        
        mean_freq_bands_delta = slice(32, 36)
        mean_freq_bands_theta = slice(36, 40)
        mean_freq_bands_alpha = slice(40, 44)
        mean_freq_bands_beta = slice(44, 48)
        
        norm_freq_bands_delta = slice(48, 52)
        norm_freq_bands_theta = slice(52, 56)
        norm_freq_bands_alpha = slice(56, 60)
        norm_freq_bands_beta = slice(60, 64)

        sample_entropies = slice(64, 68)
        shannon_entropies = slice(68, 72)
        spectral_entropies = slice(72, 76)
        
        power_asy_delta_1 = slice(76, 77)
        power_asy_delta_2 = slice(77, 78)

        power_asy_theta_1 = slice(78, 79)
        power_asy_theta_1 = slice(79, 80)

        power_asy_alpha_1 = slice(80, 81)
        power_asy_alpha_2 = slice(81, 82)

        power_asy_beta_1 = slice(82, 83)
        power_asy_beta_2 = slice(83, 84)
        
        amplitude_normalisation_slices = [
            rms, 
            total_powers, 
            mean_freq_bands_alpha, 
            mean_freq_bands_beta, 
            mean_freq_bands_delta, 
            mean_freq_bands_theta
            ]
        
        log_transformation_slices = [
            kurtosis, 
            rms,
            total_powers,
            mean_freq_bands_theta,
            mean_freq_bands_alpha,
            mean_freq_bands_beta,
            mean_freq_bands_delta,
            norm_freq_bands_theta,
            norm_freq_bands_alpha,
            norm_freq_bands_beta,
            shannon_entropies    
        ]
        
        for window in data:
            window_features = []
            
            # Time domain features
            window_features.extend(self._time_domain_features(window))


            # Frequency domain features
            window_features.extend(self._frequency_domain_features(window))

            # Entropy-derived features
            window_features.extend(self._entropy_features(window))
            
            # # Asymmetry features
            window_features.extend(self._asymmetry_features(window))
            
            features.append(window_features)


        features = np.array(features)
        features = np.nan_to_num(features)

        for _slice in amplitude_normalisation_slices:
            features[:, _slice] = self._median_memory_decay(features[:, _slice])
        
        for _slice in log_transformation_slices:
            features[:, _slice] = np.log(features[:, _slice])
            

        return (features, labels)

    def _time_domain_features(self, window):
        zero_crossings = np.sum(np.diff(np.sign(window)) != 0, axis=1)

        minima = np.min(window, axis=1)
        maxima = np.max(window, axis=1)
        skewness = stats.skew(window, axis=1)
        kurtosis = stats.kurtosis(window, axis=1)
        rms = np.sqrt(np.mean(np.square(window), axis=1))
        return np.concatenate([np.sort(zero_crossings), np.sort(minima), np.sort(maxima), np.sort(skewness), np.sort(kurtosis), np.sort(rms)])

    def _frequency_domain_features(self, window):
        freqs, psd = signal.welch(window, fs=self.sampling_rate, nperseg=min(256, window.shape[1]))
        total_power = np.sum(psd, axis=1)
        peak_freq = freqs[np.argmax(psd, axis=1)]
    
        # delta (1-3 Hz), theta (4-8 Hz), alph (9-13 Hz), beta (14-20 Hz), HF band (40-80 Hz)
        freq_bands = [(1, 3), (4, 8), (9, 13), (14, 20)]
        mean_power = []
        band_power = []
        for low, high in freq_bands:
            band_mask = (freqs >= low) & (freqs < high)
            mean_power.extend(np.mean(psd[:, band_mask], axis=1))
            band_power.extend(self._compute_normalized_power(psd[:, band_mask]))

        return np.concatenate([np.sort(total_power), np.sort(peak_freq), np.sort(mean_power), np.sort(band_power)])

    def _compute_normalized_power(self, psd):
        # Compute total power for each channel
        total_power = np.sum(psd, axis=1)
        
        # Normalize the power for each channel
        normalized_power = total_power / np.sum(total_power)
        
        return normalized_power

    def _entropy_features(self, window):
        sample_entropy = np.apply_along_axis(self._sample_entropy, 1, window)
        
        shannon_entropy = stats.entropy(np.abs(window), axis=1)
        spectral_entropy = stats.entropy(np.abs(np.fft.fft(window)), axis=1)
        
        return np.concatenate([np.sort(sample_entropy), np.sort(shannon_entropy), np.sort(spectral_entropy)])

    def _asymmetry_features(self, window):
        left_channels = window[:len(window)//2]
        right_channels = window[len(window)//2:]
        
        freqs, left_psd = signal.welch(left_channels, fs=self.sampling_rate, nperseg=min(256, window.shape[1]))
        _, right_psd = signal.welch(right_channels, fs=self.sampling_rate, nperseg=min(256, window.shape[1]))
        
        # delta (1-3 Hz), theta (4-8 Hz), alph (9-13 Hz), beta (14-20 Hz)
        freq_bands = [(1, 3), (4, 8), (9, 13), (14, 20)]
        asymmetry = []
        
        for low, high in freq_bands:
            band_mask = (freqs >= low) & (freqs < high)
            left_power = np.sum(left_psd[:, band_mask], axis=1)
            right_power = np.sum(right_psd[:, band_mask], axis=1)
            asymmetry.append((right_power - left_power) / (right_power + left_power))
        
        return np.sort(np.array(asymmetry).flatten())

    def _sample_entropy(self, x, m=2, r=0.2):
        n = len(x)
        r *= np.std(x)
        
        def _count_matches(template):
            return np.sum(np.max(np.abs(x[None, m:] - template[:, None]), axis=1) < r)
        
        A = np.mean([_count_matches(x[i:i+m]) for i in range(n-m)])
        B = np.mean([_count_matches(x[i:i+m+1]) for i in range(n-m)])
        
        return -np.log(B / A) if A > 0 and B > 0 else 0

    def _median_memory_decay(self, features):
        # Apply median decaying over time normalization
        time_constant = 0.1  # Adjust as needed
        normalized = np.zeros_like(features)
        running_median = np.median(features[0], axis=0)
        
        for i, window in enumerate(features):
            normalized[i] = (window - running_median) / (np.abs(running_median) + 1e-8)
            running_median = (1 - time_constant) * running_median + time_constant * np.median(window, axis=0)
        
        return normalized
    
    
    
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
        zero_crossings = slice(0,4)
        minimas = slice(4,8)
        maximas = slice(8,12)
        skewness = slice(12, 16)
        kurtosis = slice(16,20)
        rms = slice(20, 24)
        
        total_powers = slice(24, 28)
        peak_freqs = slice(28, 32)
        
        mean_freq_bands_delta = slice(32, 36)
        mean_freq_bands_theta = slice(36, 40)
        mean_freq_bands_alpha = slice(40, 44)
        mean_freq_bands_beta = slice(44, 48)
        
        norm_freq_bands_delta = slice(48, 52)
        norm_freq_bands_theta = slice(52, 56)
        norm_freq_bands_alpha = slice(56, 60)
        norm_freq_bands_beta = slice(60, 64)

        sample_entropies = slice(64, 68)
        shannon_entropies = slice(68, 72)
        spectral_entropies = slice(72, 76)
        
        power_asy_delta_1 = slice(76, 77)
        power_asy_delta_2 = slice(77, 78)

        power_asy_theta_1 = slice(78, 79)
        power_asy_theta_1 = slice(79, 80)

        power_asy_alpha_1 = slice(80, 81)
        power_asy_alpha_2 = slice(81, 82)

        power_asy_beta_1 = slice(82, 83)
        power_asy_beta_2 = slice(83, 84)
        
        amplitude_normalisation_slices = [
            rms, 
            total_powers, 
            mean_freq_bands_alpha, 
            mean_freq_bands_beta, 
            mean_freq_bands_delta, 
            mean_freq_bands_theta
            ]
        
        log_transformation_slices = [
            kurtosis, 
            rms,
            total_powers,
            mean_freq_bands_theta,
            mean_freq_bands_alpha,
            mean_freq_bands_beta,
            mean_freq_bands_delta,
            norm_freq_bands_theta,
            norm_freq_bands_alpha,
            norm_freq_bands_beta,
            shannon_entropies    
        ]
        
        for window in data:
            # print(window.shape)
            window_features = []
            
            # Time domain features
            window_features.extend(self._time_domain_features(window))


            # Frequency domain features
            window_features.extend(self._frequency_domain_features(window))

            # Entropy-derived features
            window_features.extend(self._entropy_features(window))
            
            # # Asymmetry features
            window_features.extend(self._asymmetry_features(window))
            
            features.append(window_features)
            # print(len(features))

            # # print("window_features: \n", np.array(window_features))
            
            # # import sys
            # # sys.exit(1)

        features = np.array(features)
        
        if len(features.shape) < 2:
            return "error", -1
        
        features = np.nan_to_num(features)
        # print('features shape: ', features.shape)

        for _slice in amplitude_normalisation_slices:
            features[:, _slice] = self._median_memory_decay(features[:, _slice])
        
        for _slice in log_transformation_slices:
            features[:, _slice] = np.log(features[:, _slice])
            

        return (features, labels)

    def _time_domain_features(self, window):
        zero_crossings = np.sum(np.diff(np.sign(window)) != 0, axis=1)

        minima = np.min(window, axis=1)
        maxima = np.max(window, axis=1)
        skewness = stats.skew(window, axis=1)
        kurtosis = stats.kurtosis(window, axis=1)
        rms = np.sqrt(np.mean(np.square(window), axis=1))
        return np.concatenate([np.sort(zero_crossings), np.sort(minima), np.sort(maxima), np.sort(skewness), np.sort(kurtosis), np.sort(rms)])

    def _frequency_domain_features(self, window):
        freqs, psd = signal.welch(window, fs=self.sampling_rate, nperseg=min(256, window.shape[1]))
        total_power = np.sum(psd, axis=1)
        peak_freq = freqs[np.argmax(psd, axis=1)]
    
        # delta (1-3 Hz), theta (4-8 Hz), alph (9-13 Hz), beta (14-20 Hz), HF band (40-80 Hz)
        freq_bands = [(1, 3), (4, 8), (9, 13), (14, 20)]
        mean_power = []
        band_power = []
        for low, high in freq_bands:
            band_mask = (freqs >= low) & (freqs < high)
            mean_power.extend(np.mean(psd[:, band_mask], axis=1))
            band_power.extend(self._compute_normalized_power(psd[:, band_mask]))

        return np.concatenate([np.sort(total_power), np.sort(peak_freq), np.sort(mean_power), np.sort(band_power)])

    def _compute_normalized_power(self, psd):
        # Compute total power for each channel
        total_power = np.sum(psd, axis=1)
        
        # Normalize the power for each channel
        normalized_power = total_power / np.sum(total_power)
        
        return normalized_power

    def _entropy_features(self, window):
        sample_entropy = np.apply_along_axis(self._sample_entropy, 1, window)
        
        shannon_entropy = stats.entropy(np.abs(window), axis=1)
        spectral_entropy = stats.entropy(np.abs(np.fft.fft(window)), axis=1)
        
        return np.concatenate([np.sort(sample_entropy), np.sort(shannon_entropy), np.sort(spectral_entropy)])

    def _asymmetry_features(self, window):
        left_channels = window[:len(window)//2]
        right_channels = window[len(window)//2:]
        
        freqs, left_psd = signal.welch(left_channels, fs=self.sampling_rate, nperseg=min(256, window.shape[1]))
        _, right_psd = signal.welch(right_channels, fs=self.sampling_rate, nperseg=min(256, window.shape[1]))
        
        # delta (1-3 Hz), theta (4-8 Hz), alph (9-13 Hz), beta (14-20 Hz)
        freq_bands = [(1, 3), (4, 8), (9, 13), (14, 20)]
        asymmetry = []
        
        for low, high in freq_bands:
            band_mask = (freqs >= low) & (freqs < high)
            left_power = np.sum(left_psd[:, band_mask], axis=1)
            right_power = np.sum(right_psd[:, band_mask], axis=1)
            asymmetry.append((right_power - left_power) / (right_power + left_power))
        
        return np.sort(np.array(asymmetry).flatten())

    def _sample_entropy(self, x, m=2, r=0.2):
        n = len(x)
        r *= np.std(x)
        
        def _count_matches(template):
            return np.sum(np.max(np.abs(x[None, m:] - template[:, None]), axis=1) < r)
        
        A = np.mean([_count_matches(x[i:i+m]) for i in range(n-m)])
        B = np.mean([_count_matches(x[i:i+m+1]) for i in range(n-m)])
        
        return -np.log(B / A) if A > 0 and B > 0 else 0

    def _median_memory_decay(self, features):
        # Apply median decaying over time normalization
        time_constant = 0.1  # Adjust as needed
        normalized = np.zeros_like(features)
        running_median = np.median(features[0], axis=0)
        
        for i, window in enumerate(features):
            normalized[i] = (window - running_median) / (np.abs(running_median) + 1e-8)
            running_median = (1 - time_constant) * running_median + time_constant * np.median(window, axis=0)
        
        return normalized
    
class HFBandFeatureExtraction(BaseEstimator, TransformerMixin):
    def __init__(self, low=40, high=80, sampling_rate=256):
        """
        Custom sklearn transformer that extracts features from EEG data and applies normalization.

        Parameters:
        sampling_rate (int): Sampling rate of the EEG data in Hz (default: 256)
        """
        self.low = low
        self.high = high
        self.sampling_rate = sampling_rate

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        
        X, y = data
        
        hf_features = []
        for window in X:
            hf_features.append(self._frequency_domain_features(window))
                    
        return (np.array(hf_features), y)
        
    def _frequency_domain_features(self, window):
        freqs, psd = signal.welch(window, fs=self.sampling_rate, nperseg=min(256, window.shape[1]))
    
        # HF band (40-80 Hz)
        freq_bands = [(self.low, self.high)]
        mean_power = []
        band_power = []
        for low, high in freq_bands:
            band_mask = (freqs >= low) & (freqs < high)
            mean_power.extend(np.mean(psd[:, band_mask], axis=1))
            band_power.extend(self._compute_normalized_power(psd[:, band_mask]))

        return np.concatenate([mean_power, band_power])

    def _compute_normalized_power(self, psd):
        # Compute total power for each channel
        total_power = np.sum(psd, axis=1)
        
        # Normalize the power for each channel
        normalized_power = total_power / np.sum(total_power)
        
        return normalized_power

    
class BalanceSeizureSegments(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=None, ratio=1):
        """
        Custom sklearn transformer that balances seizure and non-seizure segments
        by randomly sampling non-seizure segments.

        Parameters:
        random_state (int, optional): Seed for the random number generator for reproducibility.
        """
        self.random_state = random_state
        self.ratio = ratio

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
        num_seizures = len(seizure_segments) * self.ratio
        
        # Randomly sample non-seizure segments to match the number of seizure segments
        sampled_non_seizure = rnd.sample(non_seizure_segments, min(num_seizures, len(non_seizure_segments)))

        # Combine seizure and sampled non-seizure segments
        balanced_segments = seizure_segments + sampled_non_seizure

        # Shuffle the combined segments
        segments = np.array([signal for signal, _ in balanced_segments])
        labels = np.array([label for _, label in balanced_segments])
    
        return (segments, labels) 


class Pre_Post_Drop(BaseEstimator, TransformerMixin):
    def __init__(self, segment_length=4, overlap=2, random_state=None):
        self.random_state=random_state
        self.segment_length = segment_length
        self.overlap = overlap

    def fit(self, X, y=None):
        # No fitting necessary for this transformer, so just return self
        return self
    
    def transform(self, X, y=None):
        if self.random_state is not None:
            rnd.seed(self.random_state)

        X, y = X
        indices_to_remove = [] # We keep track of which indices of X need to be removed and only remove them all at once after, in order to prevent end-of-list issues with the next for-loop
        print("X shape before pre post drop ", X.shape)

        for i in range(1, len(y)):
            if y[i]== 1 and y[i-1] == 0: # Check if i-th segment is the start of a seizure

                indices_to_remove.extend([i-j for j in range(1, int(self.overlap * (60 / self.segment_length)) + 1)])

            elif y[i] == 1 and y[i+1] == 0: # Check if i-th segment is end of seizure

                indices_to_remove.extend([i+j for j in range(1, int(self.overlap * (60 / self.segment_length)) + 1)])
            else:
                continue

            
        #Now we actually remove the segments flagged earlier
        X = np.delete(X, indices_to_remove, axis=0)
        y = np.delete(y, indices_to_remove)
        # print("X shape after pre post drop ", X.shape)
        return X, y

class RatioPseudoUniformSampling(BaseEstimator, TransformerMixin):
    def __init__(self, ratio = 2.0, num_blocks=1000):
        """
        Custom sklearn transformer that randomly drops a percentage of data. The data it does not drop is evenly distributed in time.
        
        Parameters:
        drop_percentage (float): The percentage of data to drop (0 < drop_percentage < 1).
        """
        self.ratio = ratio
        self.num_blocks = num_blocks

    def fit(self, X, y=None):
        # No fitting necessary for this transformer, so just return self
        return self
    
    def transform(self, X):
        """
        Randomly drop a percentage of the data, accounting for the fact that the number of left over data per hour should be equal for every hour.

        Parameters:
        X (tuple): Tuple containing:
            - X (numpy.ndarray): Data array with shape (N, C, F).
            - y (numpy.ndarray): Labels array with shape (N,).

        Returns:
        tuple: Tuple containing:
            - X (numpy.ndarray): Reduced data array.
            - y (numpy.ndarray): Corresponding reduced labels.
        """
        print("Starting normalization and dropping segments.")
        

        X, y = X
        # Ensure data is of type float32 for precision
        X = X.astype(np.float32)

        z = list(range(len(y)))

        X = list(zip(X, y, z))

        seizure_segments = np.array([segment for segment in X if segment[1] == 1], dtype = object)
        non_seizure_segments = np.array([segment for segment in X if segment[1] == 0], dtype = object)

        # Useful constants
        num_retain = int(len(seizure_segments)*self.ratio)
        jump = len(non_seizure_segments) // self.num_blocks

        # Randomly select indices to retain in each block
        retain_indices = []
        for i in range(self.num_blocks - 1): 
            start = i * jump
            end = (i + 1) * jump
            to_sample_from = np.arange(start, end)
            if len(to_sample_from) > 0:
                selected_indices = np.random.choice(to_sample_from, 
                                                 size = num_retain // self.num_blocks, 
                                                 replace=False)
                retain_indices.extend(selected_indices)
        # Sort the indices to maintain order (optional)
        retain_indices.sort() 
        
        retain_indices = np.array(retain_indices, dtype = int)
        # Select the retained samples
        retained_non_seizure_segments = list(non_seizure_segments[retain_indices]) #Change to list to use extend later

        retained_non_seizure_segments.extend(seizure_segments)
        _ = retained_non_seizure_segments
        X = sorted(_, key = lambda x: x[2])

        X_reduced, y_reduced, z_reduced = list(zip(*X))
        # print("X shape after pseudouniform sampling ",  np.array(X_reduced).shape)
        return np.array(X_reduced), np.array(y_reduced)

class PercentagePseudoUniformSampling(BaseEstimator, TransformerMixin):
    def __init__(self, drop_percentage=0.8, num_blocks=1000):
        """
        Custom sklearn transformer that randomly drops a percentage of data. The data it does not drop is evenly distributed in time.
        
        Parameters:
        drop_percentage (float): The percentage of data to drop (0 < drop_percentage < 1).
        """
        self.drop_percentage = drop_percentage
        self.num_blocks = num_blocks

    def fit(self, X, y=None):
        # No fitting necessary for this transformer, so just return self
        return self
    
    def transform(self, X):
        """
        Randomly drop a percentage of the data, accounting for the fact that the number of left over data per hour should be equal for every hour.

        Parameters:
        X (tuple): Tuple containing:
            - X (numpy.ndarray): Data array with shape (N, C, F).
            - y (numpy.ndarray): Labels array with shape (N,).

        Returns:
        tuple: Tuple containing:
            - X (numpy.ndarray): Reduced data array.
            - y (numpy.ndarray): Corresponding reduced labels.
        """
        print("Starting normalization and dropping segments.")
        

        X, y = X
        # Ensure data is of type float32 for precision
        X = X.astype(np.float32)

        z = list(range(len(y)))

        X = list(zip(X, y, z))

        seizure_segments = np.array([segment for segment in X if segment[1] == 1], dtype = object)
        non_seizure_segments = np.array([segment for segment in X if segment[1] == 0], dtype = object)

        # Useful constants
        retain_percentage = 1 - self.drop_percentage
        num_retain = int(len(non_seizure_segments) * retain_percentage) 
        jump = len(non_seizure_segments) // self.num_blocks

        # Randomly select indices to retain in each block
        retain_indices = []
        for i in range(self.num_blocks - 1): 
            start = i * jump
            end = (i + 1) * jump
        
            to_sample_from = np.arange(start, end)

            if len(to_sample_from) > 0:
                selected_indices = np.random.choice(to_sample_from, 
                                                 size = num_retain // self.num_blocks, 
                                                 replace=False)
            
                retain_indices.extend(selected_indices)
        
        # Sort the indices to maintain order (optional)
        retain_indices.sort() 
        
        retain_indices = np.array(retain_indices)
        # Select the retained samples
        retained_non_seizure_segments = list(non_seizure_segments[retain_indices]) #Change to list to use extend later

        retained_non_seizure_segments.extend(seizure_segments)
        _ = retained_non_seizure_segments
        X = sorted(_, key = lambda x: x[2])

        X_reduced, y_reduced, z_reduced = list(zip(*X))

        return np.array(X_reduced), np.array(y_reduced)
    
def process_batch(batch: np.ndarray, y_batch: np.ndarray):
    """
    Process a batch of EEG segments through various transformations.

    Args:
        batch (np.ndarray): A batch of EEG segments.
        y_batch (np.ndarray): Corresponding labels for the batch.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Processed features, high-frequency features, and labels.
    """
    # Apply bandpass filter
    X_bf, y = BandpassFilter(sfreq=256, lowcut=1, highcut=25, order=6).transform((batch, y_batch))
    
    # Apply RMS amplitude filter
    X_rms, y, mask = RMSAmplitudeFilter(min_amplitude=13, max_amplitude=150, return_id=True).transform((X_bf, y))
    
    # Extract EEG features
    X_features, y = EEGFeatureExtractor(sampling_rate=256).transform((X_rms, y))
    
    if isinstance(X_features, str):
        return "error", -1, -1
    
    # Extract high-frequency band features
    X_hf, _ = HFBandFeatureExtraction().transform((batch, y_batch))
    X_hf = X_hf[mask]
    
    return X_features, X_hf, y

def get_svm_features(X: np.ndarray, y: np.ndarray, batch_size: int = 128, max_workers: int = None):
    """
    Extract SVM features from EEG data using parallel processing.

    Args:
        X (np.ndarray): Input EEG data.
        y (np.ndarray): Corresponding labels.
        batch_size (int): Number of segments to process in each batch.
        max_workers (int): Maximum number of worker threads.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed features and corresponding labels.
    """
    # Segment the input data
    X_segmented, y = SegmentSignals(fs=256, segment_length=2, overlap=0).transform((X, y))
    
    X_balanced, y = BalanceSeizureSegments(random_state=42).transform((X_segmented, y))
    all_features = []
    all_hf =  []
    all_y = []
    
    # Create batches
    num_segments = len(X_balanced)
    batches = [(X_balanced[i:i+batch_size], y[i:i+batch_size]) 
               for i in range(0, num_segments, batch_size)]

        
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch, y_batch in batches:
            future = executor.submit(process_batch, batch, y_batch)
            futures.append(future)
        # import sys
        # sys.exit(1)
        # Collect results as they complete
        for future in as_completed(futures):
            X_features, X_hf, y_processed = future.result()
            
            if isinstance(X_features, str):
                continue
            
            all_features.append(X_features)
            all_hf.append(X_hf)
            all_y.extend(y_processed)
    
    # Combine results from all batches
    X_features = np.vstack(all_features)
    X_hf = np.vstack(all_hf)
    y_final = np.array(all_y)
    
    # Concatenate features and handle NaN values
    X_final = np.concatenate([X_features, X_hf], axis=1)
    X_final = np.nan_to_num(X_final)
    
    return X_final, y_final

    

def load_npz_files(root_dir):
    seiz_segments = []
    non_seiz_segments = []
    seiz_labels = []
    non_seiz_labels = []
    cnt  = 0
    seizure_count = 0
    # Walk through the root directory and subdirectories
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            cnt += 1
            if file.endswith(".npz"):
                file_path = os.path.join(subdir, file)
                try:
                    # Load the npz file
                    npz_file = np.load(file_path, allow_pickle=True)
                    segment = torch.from_numpy(npz_file['x'].astype(np.float32))
                    label = torch.from_numpy(npz_file['y'].astype(np.float32))  # Ensure labels are float for comparison
                    seizure_count += npz_file['y'].astype(np.float32)
                    # print("Shape bro ", segment.shape)
                    if(cnt % 100 == 0):   
                        plot_channel_scatter(segment, 1, label)
                    # Append the segment and label to appropriate lists
                    if label.item() == 1.0:  # Seizure segments
                        seiz_segments.append(segment)
                        seiz_labels.append(label)
                    else:  # Non-seizure segments
                        non_seiz_segments.append(segment)
                        non_seiz_labels.append(label)
                
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    # print("count, ", cnt)
    # Convert lists to tensors
    
    print("number of seizures: ", seizure_count)
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

    # # Print counts and shapes
    # print("Count of seizures: ", len(seiz_segments))
    # print("Count of non-seizures: ", len(non_seiz_segments))
    # print("Seizures shape: ", shuffled_seiz.shape)
    # print("Non-seizures shape: ", shuffled_non_seiz.shape)

    # Return both shuffled segments and their respective labels
    return (shuffled_seiz, shuffled_seiz_labels), (shuffled_non_seiz, shuffled_non_seiz_labels)



def plot_channel_scatter(data, channel_idx, label):
    
    # Select the data for the given channel and remove the last singleton dimension
    channel_data = data[channel_idx, :].cpu().numpy()  # Convert to numpy for plotting

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(channel_data)), channel_data, c='blue', alpha=0.7)
    plt.title(f"Scatter Plot for Channel {channel_idx} , label {label}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def process_patient_folder(patient_folder, save_root):
    """
    Process all .edf files in a patient's folder, concatenate the data, and apply the pipeline.
    :param patient_folder: Path to the patient's folder (e.g., ./data/dataset/train/raw/chb01)
    :param save_root: Root directory where processed files will be saved.
    """
    # Initialize empty lists to accumulate data
    all_eeg_data = []
    all_labels = []

    # Load all .edf file paths for the current patient
    files_list = load_edf_filepaths(patient_folder)
    i = 0

    pbar = tqdm(files_list)
    for file in pbar: 
        # pbar.set_description(f"current file {file}")
        
        # Check if a corresponding .edf.seizures file exists, only do this if we use a validation set
       # #seizure_file = file + '.seizures'
       # #if i == 0:
       # #    if os.path.exists(seizure_file):
       # #        print(f"Skipped {file}")
       # #        i += 1
       # #        continue

        # channels="FZ-CZ;CZ-PZ;F8-T8;P4-O2;FP2-F8;F4-C4;C4-P4;P3-O1;FP2-F4;F3-C3;C3-P3;P7-O1;FP1-F3;F7-T7;T7-P7;FP1-F7"
        eeg_file = load_eeg_file(file)

        if eeg_file == None:
            print('its none')
            continue
        
        eeg_data = eeg_file.data
        labels = eeg_file.get_labels()

        print(f"Loaded file {file}: EEG data shape: {eeg_data.shape}, Labels shape: {labels.shape}")

        all_eeg_data.append(eeg_data)
        all_labels.append(labels)

    # Concatenate all EEG data and labels along the time axis
    combined_eeg_data = np.concatenate(all_eeg_data, axis=1)  # Combine along time axis
    combined_labels = np.concatenate(all_labels, axis=0)

    # print(f"Combined EEG data shape for {os.path.basename(patient_folder)}: {combined_eeg_data.shape}")
    # print(f"Combined labels shape for {os.path.basename(patient_folder)}: {combined_labels.shape}")

    # Apply the pipeline on the combined data
    pipeline = Pipeline([('filters', BandpassFilter(sfreq=256, lowcut=1, highcut=40, order=6)),
                        #  ('down sampling', MeanDownSampling(down_freq=64, sfreq=256)),
                         ('normalizes', ZScoreNormalization()),
                         ('segments', SegmentSignals(fs=256, segment_length=4, overlap=2)),
                        #  ('drop', Pre_Post_Drop()),
                        #  ('sample', RatioPseudoUniformSampling(ratio=15, num_blocks=1000))
                         ]) 
    X, y = pipeline.transform(X=(combined_eeg_data, combined_labels))
    
    
    print(f"Transformed data shape for {os.path.basename(patient_folder)}: {X.shape}")
    print(f"Transformed labels shape for {os.path.basename(patient_folder)}: {y.shape}")

    # Save the processed data
    save_dir = os.path.join(save_root, os.path.basename(patient_folder))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file =  os.path.basename(os.path.normpath(patient_folder))
    save_signals_and_labels(X, y, save_dir=save_dir, filename=save_file)
    
    print(f"Saved data for {os.path.basename(patient_folder)} with shape {X.shape}.")



def process_all_patients(dataset_path, save_root_path):
    """
    Process all patient folders in the dataset.
    :param dataset_path: Path to the root folder containing patient folders (e.g., ./data/dataset/train/raw/)
    :param save_root_path: Path where processed data will be saved (e.g., ./data/dataset/signals-per-patient/)
    """
    # Loop through all folders in the dataset path (assuming each folder is a patient)
    for patient_folder in os.listdir(dataset_path):
        full_patient_path = os.path.join(dataset_path, patient_folder)
        if os.path.isdir(full_patient_path):  # Only process directories (patient folders)
            process_patient_folder(full_patient_path, save_root_path)

    
# This function uses the previous approach, where only data from files with .seizure file are segmented. 
# Also pipeline is applied per file instead of applying it once when all data is concatenated. 
def process_seizure_files(dataset_path, dataset_save_root_path):
    
    from tqdm import tqdm
    from common.utils import load_edf_filepaths, clean_path, load_eeg_file, save_spectrograms_and_labels, load_seizure_edf_filepaths
    
    """
    WARNING: Current code computes features for svm.
    """

    files_list = load_seizure_edf_filepaths(dataset_path)
    
    pbar = tqdm(files_list)
    for file in pbar: 
        # pbar.set_description(f"current file {file}")

        # channels="FZ-CZ;CZ-PZ;F8-T8;P4-O2;FP2-F8;F4-C4;C4-P4;P3-O1;FP2-F4;F3-C3;C3-P3;P7-O1;FP1-F3;F7-T7;T7-P7;FP1-F7"
        eeg_file = load_eeg_file(file)

        if eeg_file == None:
            print('its none')
            continue
        
        eeg_data = eeg_file.data
        labels = eeg_file.get_labels()
        
        pipeline = Pipeline([('filters', BandpassFilter(sfreq=256, lowcut=1, highcut=40, order=6)),
                            #  ('Downsamples', MeanDownSampling(down_freq = 64, sfreq = 256)),
                         ('normalizes', ZScoreNormalization()),
                         ('segments', SegmentSignals(fs=256, segment_length=4, overlap=2)),
                         ('drop', Pre_Post_Drop()),
                         ('sample', RatioPseudoUniformSampling(ratio=15, num_blocks=200)),
                         ]) 
        
        X, y = pipeline.transform((eeg_data, labels))

        save_dir, filename = clean_path(file, dataset_path)
        save_dir = dataset_save_root_path + save_dir
        filename = filename.split('.')[0]        
        pbar.set_description(f"saved {filename} of size: {X.shape} to {save_dir}")
            
        save_spectrograms_and_labels(X, y, save_dir=save_dir, filename=filename)

def process_all_seizure_files(dataset_path, dataset_save_root_path):
    # Loop through all folders in the dataset path (assuming each folder is a patient)
    for patient_folder in os.listdir(dataset_path):
        full_patient_path = os.path.join(dataset_path, patient_folder)
        if os.path.isdir(full_patient_path):  # Only process directories (patient folders)
            process_seizure_files(full_patient_path, dataset_save_root_path)
    
if __name__ == "__main__":
    dataset_path = "./data/dataset/train/raw/chb04"
    dataset_path_single = "./data/dataset/train/raw/chb24"
    
    save_root_path = "./data/dataset/chb24_test_overlap"

    #print(f"Files in dataset_path_full: {load_edf_filepaths(dataset_path)}")
    #testing
    
    # If you want to use the previous dataset creation approach, use this:
    #process_seizure_files(dataset_path_single, save_root_path)

    # If you want to process all directories using previous dataset creation approach    
    # process_all_seizure_files(dataset_path, save_root_path)

    # If you want to process all directories
    # process_all_patients(dataset_path, save_root_path)


    # If you want to process a single patient (for test set), use this
    process_patient_folder(dataset_path_single, save_root_path)
    
    # load_npz_files("./data/dataset/parviz_train-15-1blya/full_train/chb03")