import torch.nn as nn
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader

from asd.models.base import BaseModel

#encoder
class ShallowEncoder(BaseModel):
    def __init__(self, args, in_features, latent_dims):
        super(ShallowEncoder, self).__init__(args=args)
        self.latent_dims = latent_dims
        self.in_features = in_features
        self.linear1 = nn.Linear(in_features=in_features, out_features=latent_dims)
        self.relu = nn.ReLU()


    def forward(self, x):

        x = th.flatten(x, start_dim=1)
        x = self.relu(self.linear1(x))
        return x

#decoder
class ShallowDecoder(BaseModel):
    def __init__(self, args, latent_dims, out_features, channels):
        super(ShallowDecoder, self).__init__(args=args)

        self.latent_dims = latent_dims
        self.out_features = out_features
        self.channels = channels
        self.linear1 = nn.Linear(in_features=latent_dims, out_features=out_features*channels)
        self.relu    = nn.ReLU()
    
    def forward(self, z):

        z = self.relu(self.linear1(z))
        z = z.reshape((-1, self.channels, self.out_features))

        return z

#autoencoder
class ShallowAE(BaseModel):
    def __init__(self, args, latent_dims, in_features, out_features, channels=4):
        super(ShallowAE, self).__init__(args=args)

        self.encoder = ShallowEncoder(in_features=in_features, latent_dims=latent_dims)
        self.decoder = ShallowDecoder(latent_dims=latent_dims, out_features=out_features, channels=channels)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y
    
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