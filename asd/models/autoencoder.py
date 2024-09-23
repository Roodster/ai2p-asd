import torch.nn as nn
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader

from asd.models.base import BaseModel

class ShallowEncoder(BaseModel):
    def __init__(self, args, input_dim=256, hidden_dim=64):
        super(ShallowEncoder, self).__init__(args)
        
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        batch_size, channels, seq_len, _ = x.shape
        x = x.view(batch_size * channels * seq_len, -1)  # Reshape to 2D
        x = self.leaky_relu(self.linear1(x))
        return x

class ShallowDecoder(BaseModel):
    def __init__(self, args, input_dim, output_dim):
        super(ShallowDecoder, self).__init__(args)
        
        self.linear1 = nn.Linear(input_dim, output_dim, bias=True)
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
        print(f'x1.shape {x.shape}')
        original_shape = x.shape
        
        x = self.encoder(x)
        print(f'x2.shape {x.shape}')

        x = self.decoder(x)
        print(f'x3.shape {x.shape}')
        
        # Reshape back to original shape
        x = x.view(original_shape)
        print(f'x4.shape {x.shape}')
    
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