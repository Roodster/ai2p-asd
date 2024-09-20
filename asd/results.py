import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

class Results:
    def __init__(self, file=None):
        self._prev_results = None
        self._results = None
        # Initialize lists
        self._epochs = []
        self._train_losses = []
        self._test_losses = []
        self._tps = []
        self._fps = []
        self._tns = []
        self._fns = []

        # If a file is provided, read the data and populate the lists
        if file is not None:
            self._prev_results = pd.read_csv(file)

            # Populate the lists with data from the dataframe
            self._epochs = self._results['epoch'].tolist()
            self._train_losses = self._results['train_loss'].tolist()
            self._test_losses = self._results['test_loss'].tolist()
            self._tps = self._results['tp'].tolist()
            self._fps = self._results['fp'].tolist()
            self._tns = self._results['tn'].tolist()
            self._fns = self._results['fn'].tolist()

    def get(self):
        if self._prev_results is not None:
            # Create data array and add to dataframe
            data = np.empty((7, len(self._epochs)))
            data[0, :] = [epoch + len(self._results['epoch']) for epoch in self._epochs]
            data[1, :] = self._train_losses
            data[2, :] = self._test_losses
            data[3, :] = self._tps
            data[4, :] = self._fps
            data[5, :] = self._tns
            data[6, :] = self._fns

            # Concatenate new data with existing DataFrame
            self._results = pd.concat([self._prev_results, pd.DataFrame(data.T, columns=self._prev_results.columns)], ignore_index=True)
        
        else:
            # If no previous results, create new DataFrame
            results = {
                'epoch': self._epochs,
                'train_loss': self._train_losses,
                'test_loss': self._test_losses,
                'tp': self._tps,
                'fp': self._fps,
                'tn': self._tns,
                'fn': self._fns
            }
            self._results = pd.DataFrame(results)
        
        return self._results
    
    # Property and setter for train_losses
    @property
    def train_losses(self):
        return normalize(np.array(self._train_losses).reshape(-1, 1), axis=0).ravel()
    
    @train_losses.setter
    def train_losses(self, value):
        self._train_losses.append(value)

    # Property and setter for test_losses
    @property
    def test_losses(self):
        return normalize(np.array(self._test_losses).reshape(-1, 1), axis=0).ravel()
    
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

    # Property and setter for true positives (tp)
    @property
    def tps(self):
        return self._tps
    
    @tps.setter
    def tps(self, value):
        self._tps.append(value)

    # Property and setter for false positives (fp)
    @property
    def fps(self):
        return self._fps
    
    @fps.setter
    def fps(self, value):
        self._fps.append(value)

    # Property and setter for true negatives (tn)
    @property
    def tns(self):
        return self._tns
    
    @tns.setter
    def tns(self, value):
        self._tns.append(value)

    # Property and setter for false negatives (fn)
    @property
    def fns(self):
        return self._fns
    
    @fns.setter
    def fns(self, value):
        self._fns.append(value)