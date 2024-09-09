import pandas as pd


class Results:
    
    def __init__(self, file=None):
        
        self._results = None
        
        if file is not None:
            self.results = pd.read_csv(file)    

        self._epochs = []
        self._train_losses = []
        self._test_losses = []
        self._tps = []
        self._fps = []
        self._tns = []
        self._fns = []
    
    def get(self):
        
        if self._results is not None:
            self._results['epoch'] += self._epochs
            self._results['train_loss'] += self._train_losses
            self._results['test_loss'] += self._test_losses
            self._results['tp'] += self._tps
            self._results['fp'] += self._fps
            self._results['tn'] += self._tns
            self._results['fn'] += self._fns
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
            
            print(results)
            
            
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