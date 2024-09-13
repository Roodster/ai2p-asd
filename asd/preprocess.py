import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy import signal
import random as rnd

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
        print('filtered data...')

        return (filtered_data, y)
    
    
class SegmentSignals(BaseEstimator, TransformerMixin):
    def __init__(self, fs=256, segment_length=1, overlap=0.5):
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
        print('segmented data...')

        return np.array(segmented_data), np.array(segment_labels)
        
    
class Spectrograms(BaseEstimator, TransformerMixin):
    def __init__(self, fs=256, nperseg=None, noverlap=None):
        """
        Converts segments of EEG data into spectrograms.

        Parameters:
        fs (float): Sampling frequency in Hz.
        nperseg (int): Number of samples per segment for the spectrogram.
        noverlap (int): Number of samples to overlap between segments for the spectrogram.
        """
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap

    def fit(self, X, y=None):
        # No fitting necessary for this transformer, so just return self
        return self

    def transform(self, X, y=None):
        """
        Generate spectrograms for each EEG segment.

        Parameters:
        X (numpy.ndarray): Segmented EEG data array of shape (n_segments, n_channels, n_samples).

        Returns:
        numpy.ndarray: Array of spectrograms of shape (n_segments, n_channels, n_frequencies, n_times).
        """
        segmented_data, y = X
        n_segments, n_channels, _ = segmented_data.shape
        spectrograms = []

        for segment in segmented_data:
            segment_spectrograms = []
            for channel in range(n_channels):
                f, t, Sxx = signal.spectrogram(segment[channel], fs=self.fs,
                                               nperseg=self.nperseg or int(self.fs * 0.5),
                                               noverlap=self.noverlap or int((self.nperseg or int(self.fs * 0.5)) // 2))
                segment_spectrograms.append(Sxx)
            
            stacked_spectrogram = np.stack(segment_spectrograms, axis=0)
            spectrograms.append(stacked_spectrogram)

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
        segments = np.array([signal for signal, _ in balanced_segments ])
        labels = np.array([label for _, label in balanced_segments])
    
        return (segments, labels) 
    
    
    
    # Example usage
if __name__ == "__main__":
    # Assume you have loaded your EEG data and labels
    # eeg_data shape: (n_channels, n_samples)
    # labels shape: (n_samples,)
    
    from tqdm import tqdm
    from common.utils import load_edf_filepaths, clean_path, load_eeg_file
    
    files_list = load_edf_filepaths("./data/dataset/train/raw/")
 
    print(files_list)   
    pbar = tqdm(files_list)
    for file in files_list: 
        pbar.set_description(f"current file {file}")
        # Example data (replace with your actual data)
        eeg_file = load_eeg_file(file)
        
        if eeg_file == None:
            continue
        # eeg_data = np.random.randn(n_channels, n_samples)
        # labels = np.random.randint(0, 2, n_samples)
        
        eeg_data = eeg_file.data
        labels = eeg_file.get_labels()
        
        
        pipeline = Pipeline([('normalizes',ZScoreNormalization()), 
                             ('filters', BandpassFilter()),
                             ('segments', SegmentSignals()),
                            ('samples', BalanceSeizureSegments())]) 
        
        
        X, y = pipeline.transform(X=(eeg_data, labels))
        print(X.shape, y.shape)
        
        save_dir, filename = clean_path(file, "../data/dataset/")
        save_dir = "../data/preprocessed/" + save_dir
        filename = filename.split('.')[0]
            
        # save_spectrograms_and_labels(X, y, save_dir=save_dir, filename=filename)


    # print(f"Number of segments: {len(spectrograms)}")
    # print(f"Spectrogram shape: {spectrograms[0].shape}")
    # print(f"Segment labels shape: {segment_labels.shape}")
    