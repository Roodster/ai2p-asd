import torch.nn as nn
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader

from asd.models.base import BaseModel

#encoder
class ShallowEncoder(BaseModel):
    def __init__(self, args, input_dim, hidden_dim, channels=4):
        super(ShallowEncoder, self).__init__(args=args)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.channels = channels
        self.linear1 = nn.Linear(input_dim=input_dim*channels, out_features=hidden_dim)
        self.relu = nn.ReLU()


    def forward(self, x):

        x = th.flatten(x, start_dim=1)
        x = self.relu(self.linear1(x))
        return x

#decoder
class ShallowDecoder(BaseModel):
    def __init__(self, args, hidden_dim, out_features, channels):
        super(ShallowDecoder, self).__init__(args=args)

        self.hidden_dim = hidden_dim #in_features
        self.out_features = out_features
        self.channels = channels
        self.linear1 = nn.Linear(input_dim=hidden_dim, out_features=out_features*channels)
        self.relu    = nn.ReLU()
    
    def forward(self, z):

        z = self.relu(self.linear1(z))
        z = z.reshape((-1, 1, self.channels, self.out_features))

        return z

#autoencoder
class ShallowAE(BaseModel):
    def __init__(self, args, input_dim, hidden_dim, channels=4):
        super(ShallowAE, self).__init__(args=args)

        self.encoder = ShallowEncoder(args=args, input_dim=input_dim, hidden_dim=hidden_dim, channels=channels)
        self.decoder = ShallowDecoder(args=args, hidden_dim=hidden_dim, out_features=input_dim, channels=channels)

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
    

class BetaEncoder(BaseModel):
    
    
    def __init__(self, args, latent_dim=16, n_channels=4):
        super(BetaEncoder, self).__init__(args)
       
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(n_channels, 32, 2, 1, 1)          # B,  32, 32, 32
        self.conv2 = nn.Conv2d(32, 32, 2, 1, 1)          # B,  32, 16, 16
        self.conv3 = nn.Conv2d(32, 32, 2, 1, 1)          # B,  32,  8,  8
        self.conv4 = nn.Conv2d(32, 32, 2, 1, 1)          # B,  32,  4,  4
        # View((-1, 32*4*4)),                  # B, 512
        self.linear1 = nn.Linear(32*4*4, 256)              # B, 256
        self.linear2 = nn.Linear(256, 256)                 # B, 256
        self.linear3 = nn.Linear(256, latent_dim*2)             # B, z_dim*2

        self.relu = nn.ReLU(True)
        
        #distribution setup
        self.N = th.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(self.device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(self.device)
        
        
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
        
        mu = z[:, :self.latent_dim]
        logvar = z[:, self.latent_dim:]
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
        self.conv1 = nn.ConvTranspose2d(32, 32, 2, 1, 1) # B,  32,  8,  8
        self.conv2 = nn.ConvTranspose2d(32, 32, 2, 1, 1) # B,  32, 16, 16
        self.conv3 = nn.ConvTranspose2d(32, 32, 2, 1, 1) # B,  32, 32, 32
        self.conv4 = nn.ConvTranspose2d(32, n_channels, 2, 1, 1) # B,  nc, 64, 64
        
        self.relu = nn.ReLU(True)        
        
        self.initialize_weights()
        
    def forward(self, z):
        x = self.relu(self.linear1(z))
        x = self.relu(self.linear2(x))

        x = self.relu(self.linear3(x))
        x = x.view((-1, 32, 8, 260)) # 32 is hidden size, 8 is 4 +4 and 260 is input_dim + 4 for padding
        
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
    
    
    
