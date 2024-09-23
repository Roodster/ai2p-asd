from base import BaseModel
import torch.nn as nn
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader

from asd.args import Args
from asd.dataset import OnlineSegmentsDataset, OfflineSegmentsDataset
from asd.experiment import Experiment
from asd.writer import Writer
from asd.plots import Plots
from asd.results import Results
from asd.learner import Learner


class ShallowDecoder(BaseModel):
    
    def __init__(self, args, input_dim, output_dim):
        super(ShallowDecoder, self).__init__(args)
        
        self.linear1 = nn.Linear(input_dim, output_dim, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        return x

class ShallowEncoder(BaseModel):
    
    
    
    def __init__(self, args, input_dim=256, hidden_dim=64):
        super(ShallowEncoder, self).__init__(args)
        
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2)


    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        return x
                

class ShallowAE(BaseModel):
    
    def __init__(self, args, input_dim=256, hidden_dim=64):
        super(ShallowAE, self).__init__(args)
        
        self.encoder = ShallowEncoder(args, input_dim=input_dim, hidden_dim=hidden_dim)
        self.decoder = ShallowDecoder(args, input_dim=hidden_dim, output_dim=input_dim)
        
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
    
class SoftMaxClassifier(BaseModel):
    
    def __init__(self, args, input_dim=256, hidden_dim=256, output_dim=2):
        super(SoftMaxClassifier, self).__init__(args)
        self.encoder = ShallowEncoder(args, input_dim, hidden_dim=hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_features=output_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.softmax(self.linear(x))
        return x