import torch.nn as nn
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader

from asd.models.base import BaseModel

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
    

class BetaEncoder(BaseModel):
    
    
    def __init__(self, args, latent_dim=16, n_channels=4):
        super(BetaEncoder, self).__init__(args)
       
        self.conv1 = nn.Conv2d(n_channels, 32, 4, 2, 1)          # B,  32, 32, 32
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)          # B,  32, 16, 16
        self.conv3 = nn.Conv2d(32, 32, 4, 2, 1)          # B,  32,  8,  8
        self.conv4 = nn.Conv2d(32, 32, 4, 2, 1)          # B,  32,  4,  4
        # View((-1, 32*4*4)),                  # B, 512
        self.linear1 = nn.Linear(32*4*4, 256)              # B, 256
        self.linear2 = nn.Linear(256, 256)                 # B, 256
        self.linear3 = nn.Linear(256, latent_dim*2)             # B, z_dim*2

        self.relu = nn.ReLU(True)
        
    def reparameterize(self, mu, sig):
        return mu + sig*self.N.sample(mu.shape)
    
    def forward(self, x):
        
        z = self.relu(self.conv1(x))
        z = self.relu(self.conv2(z))
        z = self.relu(self.conv3(z))
        z = self.relu(self.conv4(z))
        z = z.view((-1, 32*4*4))
        z = self.relu(self.linear1(z))
        z = self.relu(self.linear2(z))
        z = self.relu(self.linear3(z))
        
        mu = z[:, :self.z_dim]
        logvar = z[:, self.z_dim:]
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.fill_(0)
                    

class BetaDecoder(BaseModel):
    
    
    def __init__(self, args, latent_dim=16, n_channels=4):
        super(BetaDecoder, self).__init__(args)

        
        self.linear1 = nn.Linear(latent_dim, 256)               # B, 256
        self.linear2 = nn.Linear(256, 256)                 # B, 256
        self.linear3 = nn.Linear(256, 32*4*4)              # B, 512
        self.conv1 = nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
        self.conv2 = nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
        self.conv3 = nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
        self.conv4 = nn.ConvTranspose2d(32, n_channels, 4, 2, 1), # B,  nc, 64, 64
        
        self.relu = nn.ReLU(True)        
        
        self.initialize_weights()
        
    def forward(self, z):
        x = self.relu(self.linear1(z))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        
        x = x.view((-1, 32, 4, 4))
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.fill_(0)
                    

class BetaVAE(BaseModel):
    
    def __init__(self, args,  latent_dim=16, n_channels=4):
        super(BetaVAE, self).__init__(args)

        self.encoder = BetaEncoder(args=args, latent_dim=latent_dim, n_channels=n_channels)
        self.decoder = BetaDecoder(args=args, latent_dim=latent_dim, n_channels=n_channels)
        
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)

        return x_recon, mu, logvar
    
    
    
