from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np

class Results:
    def __init__(self, file=None):
        self._prev_results = None
        self._results = None
        # Initialize lists
        self._epochs = []
        self._train_losses = []
        self._test_losses = []
        self._accuracies = []
        self._sensitivities = []
        self._precisions = []
        self._aucs = []
        self._f1s = []

        # If a file is provided, read the data and populate the lists
        if file is not None:
            self._prev_results = pd.read_csv(file)

            # Populate the lists with data from the dataframe
            self._epochs = self._results['epoch'].tolist()
            self._train_losses = self._results['train_loss'].tolist()
            self._test_losses = self._results['test_loss'].tolist()
            self._accuracies = self._results['accuracy'].tolist()
            self._sensitivities = self._results['sensitivity'].tolist()
            self._precisions = self._results['precision'].tolist()
            self._aucs = self._results['auc'].tolist()
            self._f1 = self._results['f1'].tolist()

    def get(self):
        if self._prev_results is not None:
            # Create data array and add to dataframe
            data = np.empty((8, len(self._epochs)))
            data[0, :] = [epoch + len(self._results['epoch']) for epoch in self._epochs]
            data[1, :] = self._train_losses
            data[2, :] = self._test_losses
            data[3, :] = self._accuracies
            data[4, :] = self._sensitivities
            data[5, :] = self._precisions
            data[6, :] = self._aucs
            data[7, :] = self._f1s

            # Concatenate new data with existing DataFrame
            self._results = pd.concat([self._prev_results, pd.DataFrame(data.T, columns=self._prev_results.columns)], ignore_index=True)
        
        else:
            # If no previous results, create new DataFrame
            results = {
                'epoch': self._epochs,
                'train_loss': self._train_losses,
                'test_loss': self._test_losses,
                'accuracy': self._accuracies,
                'sensitivity': self._sensitivities,
                'precision': self._precisions,
                'auc': self._aucs,
                'f1': self._f1s
            }
            self._results = pd.DataFrame(results)
        
        return self._results
    
    # Property and setter for train_losses
    @property
    def train_losses(self):
        losses = np.array(self._train_losses).reshape(-1, 1)
        normalized_losses = 1 + (losses - np.min(losses)) / (np.max(losses) - np.min(losses))
        return normalized_losses.ravel()    
    @train_losses.setter
    def train_losses(self, value):
        self._train_losses.append(value)

    # Property and setter for test_losses
    @property
    def test_losses(self):
        losses = np.array(self._test_losses).reshape(-1, 1)
        normalized_losses = 1 + (losses - np.min(losses)) / (np.max(losses) - np.min(losses))
        return normalized_losses.ravel()
    
    @test_losses.setter
    def test_losses(self, value):
        self._test_losses.append(value)

    # Property and setter for epochs
    @property
    def epochs(self):
        return self._epochs
    
    @epochs.setter
    def epochs(self, value):
        self._epochs.append(value)

    # Property and setter for accuracies
    @property
    def accuracies(self):
        return self._accuracies
    
    @accuracies.setter
    def accuracies(self, value):
        self._accuracies.append(value)

    # Property and setter for sensitivities
    @property
    def sensitivities(self):
        return self._sensitivities
    
    @sensitivities.setter
    def sensitivities(self, value):
        self._sensitivities.append(value)

    # Property and setter for precisions
    @property
    def precisions(self):
        return self._precisions
    
    @precisions.setter
    def precisions(self, value):
        self._precisions.append(value)

    # Property and setter for aucs
    @property
    def aucs(self):
        return self._aucs
    
    @aucs.setter
    def aucs(self, value):
        self._aucs.append(value)

    # Property and setter for accuracy (overall accuracy)
    @property
    def f1s(self):
        return self._f1s
    
    @f1s.setter
    def f1s(self, value):
        self._f1s.append(value)