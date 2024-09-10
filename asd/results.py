import pandas as pd
import numpy as np

class Results:
    
    def __init__(self, file=None):
        
        self._results = None
        
        if file is not None:
            self._results = pd.read_csv(file)    


        self._epochs = []
        self._train_losses = []
        self._test_losses = []
        self._tps = []
        self._fps = []
        self._tns = []
        self._fns = []
    
    def get(self):
        
        if self._results is not None:
            
            data = np.empty((7, len(self._epochs)))
            
            data[0,:] = [epoch + len(self._results['epoch']) for epoch in self._epochs]
            data[1,:] = self._train_losses
            data[2,:] = self._test_losses
            data[3, :] = self._tps
            data[4, :] = self._fps
            data[5, :] = self._tns
            data[6, :] = self._fns
            
            self._results = pd.concat([self._results, pd.DataFrame(data.T, columns=self._results.columns)], ignore_index=True)
            
        else:
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
    def train_loss(self):
        return self._train_losses
    
    @train_loss.setter
    def train_loss(self, value):
        self._train_losses.append(value)

    # Property and setter for test_losses
    @property
    def test_loss(self):
        return self._test_losses
    
    @test_loss.setter
    def test_loss(self, value):
        self._test_losses.append(value)

    # Property and setter for epochs
    @property
    def epoch(self):
        return self._epochs
    
    @epoch.setter
    def epoch(self, value):
        self._epochs.append(value)

    # Property and setter for true positives (tp)
    @property
    def tp(self):
        return self._tps
    
    @tp.setter
    def tp(self, value):
        self._tps.append(value)

    # Property and setter for false positives (fp)
    @property
    def fp(self):
        return self._fps
    
    @fp.setter
    def fp(self, value):
        self._fps.append(value)

    # Property and setter for true negatives (tn)
    @property
    def tn(self):
        return self._tns
    
    @tn.setter
    def tn(self, value):
        self._tns.append(value)

    # Property and setter for false negatives (fn)
    @property
    def fn(self):
        return self._fns
    
    @fn.setter
    def fn(self, value):
        self._fns.append(value)