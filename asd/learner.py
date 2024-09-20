import torch as th
from tqdm import tqdm
import os
import numpy as np

from torch.autograd import Variable

class BaseLearner:
    
    
    def __init__(self):
        pass
    
    def run(self):
        pass

    
class Learner(BaseLearner):


    def __init__(self, 
                 args=None, 
                 model=None, 
                 optimizer=None,
                 criterion=None                 
                 ):
        
        assert args is not None, "No args defined."
        assert model is not None, "No model defined."
        assert optimizer is not None, "No optimizer defined."
        assert criterion is not None, "No criterion defined."
        
        super().__init__()


        self.device = args.device

        # ===== DEPENDENCIES =====
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def compute_loss(self, y_pred, y_test):
        loss =  Variable(self.criterion(y_pred, y_test), requires_grad = True)
        return loss
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
    def predict(self, batch_data):
        outputs = self.model(batch_data).to(self.args.device)
        return outputs
                
    def reset(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()