import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import torch
from asd.dataset import get_dataloaders, SegmentsDataset

class Eval:
    def __init__(self, args, learner, results, n_patients=23):
        """
        Initializes the Eval class.

        :param model: Trained PyTorch model for seizure detection.
        :param segment_length: Length of each EEG segment.
        :param overlap: Overlap between segments.
        :param sample_freq: Sampling frequency of the input data.
        """
        
        self.args = args
        self.learner = learner
    
        self.n_patients = n_patients
        self.results = results
        # Initialize metrics as lists within the class
        self.metrics = {
            'sensitivity': [],
            'precision': [],
            'f1_score': [],
            'false_alarms_per_day': []
        }
        
    def evaluate(self, batch_size=64, num_workers=4):
        """
        Evaluates the model on the test data.

        :param data: List of patients' EEG data.
        :param labels: List of patients' labels.
        :param batch_size: Batch size for the DataLoader.
        :param num_workers: Number of workers for the DataLoader.
        :return: Dictionary with average sensitivity, precision, F1-score, and false alarms per day.
        """
        
        for i in range(self.n_patients):

            # Prepare DataLoader
            train_dataset = SegmentsDataset('./data/preprocessed/', mode='train', patient_id=f'{i}'.zfill(2))
            test_dataset = SegmentsDataset('./data/test/', mode='test', patient_id=f'{i}'.zfill(2))
            
            train_loader, _, _ = get_dataloaders(
                dataset=train_dataset,
                train_ratio=1,
                test_ratio=0,
                batch_size=batch_size,
                shuffle=True
            )
            test_loader, _, _ = get_dataloaders(
                dataset=test_dataset,
                train_ratio=1,
                test_ratio=0,
                batch_size=1,
                shuffle=False
            )
            
            y_test = np.array([labels for _, labels in test_loader])
            y_test = np.hstack(y_test, dtype=np.float32)
             
            self.learner.train(train_loader)
            
            predictions = self.learner.evaluate_consecutive_samples(test_loader)
            
            predictions = np.array(predictions)

            # Update metrics using the helper function
            self.get_scoring_metrics(y_pred=predictions, y_test=y_test)
            

    def get_aggregate_scoring_metrics(self):
        # Calculate average results
        avg_results = {metric: np.mean(values) for metric, values in self.metrics.items()}

        return avg_results
        
    def get_scoring_metrics(self, y_pred, y_test):
        """
        Calculates and appends performance metrics to the class's metrics dictionary.

        :param y_pred: Array of predicted labels.
        :param y_test: Array of true labels.
        """
        assert y_pred.shape == y_test.shape, "Error: y_pred and y_test do not have the same shape. "
        
        sensitivities = []
        precisions = []
        f1_scores = []
        false_alarms_per_days = []
        
        
        # RODY: I think we have to 
        for index in range(y_test.shape[0]):
                    # Compute performance metrics
            sensitivity, precision, f1_score, _ = precision_recall_fscore_support(
                y_test[index], y_pred[index], average='binary', zero_division=0)

            # Calculate false alarms per day
            tn, fp, fn, tp = confusion_matrix(y_test[index], y_pred[index]).ravel()
            false_alarms = fp
            duration_hours = len(y_test[index]) / self.sample_freq / 3600  # Total duration in hours
            false_alarms_per_day = false_alarms / (duration_hours / 24)

            sensitivities.append(sensitivity)
            precision.append(precision)
            f1_scores.append(f1_score)
            false_alarms_per_days.append(false_alarms_per_day)
            
        
        # Append to metrics dictionary
        self.results.train_loss = 0 # placeholders
        self.results.test_loss = 0 # placeholders
        self.results.sensitivity = np.mean(sensitivities)
        self.results.precision = np.mean(precisions)
        self.results.f1 = np.mean(f1_scores)
        self.results.fpd = np.mean(false_alarms_per_days)



if __name__ == "__main__":
    from asd.dataset import SegmentsDataset, get_dataloaders
    import matplotlib.pyplot as plt
    
    dataset = SegmentsDataset('../data/preprocessed/', mode='test', patient_id=1)
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset,
        train_ratio=1,
        test_ratio=0,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    def plot_spectrogram(X):
        plt.figure(figsize=(10, 4))
        plt.specgram(X, Fs=6, cmap="rainbow")
        plt.colorbar(label='Log Power')
        plt.title(f'Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()

    for batch_data, batch_labels in train_loader:
        plot_spectrogram(batch_data)   
        