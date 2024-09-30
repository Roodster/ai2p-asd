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
        self.encoder = ShallowEncoder(args, input_dim, latent=hidden_dim)
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
        self.eps_w = 0.1
        self.N = th.distributions.Normal(0, self.eps_w)
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
    
    
    
class Encoder(nn.Module):
    def __init__(self, input_size=4096, hidden_size=1024, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        outputs, (hidden, cell) = self.lstm(x)
        return (hidden, cell)


class Decoder(nn.Module):
    def __init__(
        self, input_size=4, hidden_size=64, output_size=4, num_layers=2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)


class LSTMVAE(nn.Module):
    """LSTM-based Variational Auto Encoder"""

    def __init__(
        self, input_size, hidden_size, latent_size, device=th.device("cuda")
    ):
        """
        input_size: int, batch_size x sequence_length x input_dim
        hidden_size: int, output size of LSTM AE
        latent_size: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        super(LSTMVAE, self).__init__()
        self.device = device

        # dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = 1

        # lstm ae
        self.lstm_enc = Encoder(
            input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layers
        )
        self.lstm_dec = Decoder(
            input_size=latent_size,
            output_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
        )

        self.fc21 = nn.Linear(self.hidden_size, self.latent_size)
        self.fc22 = nn.Linear(self.hidden_size, self.latent_size)
        self.fc3 = nn.Linear(self.latent_size, self.hidden_size)

    def reparametize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        noise = th.randn_like(std).to(self.device)

        z = mu + noise * std
        return z

    def forward(self, x):
        batch_size, n_channels, feature_dim, seq_len = x.shape
        x = x.reshape(batch_size, seq_len, feature_dim)
        
#         print('x1: ', x.shape)
        # encode input space to hidden space
        enc_hidden = self.lstm_enc(x)

        enc_h = enc_hidden[0].view(batch_size, self.hidden_size).to(self.device)
#         print('enc_hidden1: ', enc_h.shape)

        # extract latent variable z(hidden space to latent space)
        mean = self.fc21(enc_h)
#         print('mean: ', mean.shape)

        logvar = self.fc22(enc_h)
#         print('logvar: ', logvar.shape)

        z = self.reparametize(mean, logvar)  # batch_size x latent_size

#         print('z1: ', z.shape)
        # initialize hidden state as inputs
        h_ = self.fc3(z)
#         print('h_: ', h_.shape)

        # decode latent space to input space
        z = z.repeat(1, seq_len, 1)
#         print('z2: ', z.shape)

        z = z.view(batch_size, seq_len, self.latent_size).to(self.device)
#         print('z3: ', z.shape)

        # initialize hidden state
        hidden = (h_.contiguous().unsqueeze(0), h_.contiguous().unsqueeze(0))

        reconstruct_output, hidden = self.lstm_dec(z, hidden)
#         print('reconstruct_output: ', reconstruct_output.shape)
        
        reconstruct_output = reconstruct_output.unsqueeze(1).reshape(-1, 1, feature_dim, seq_len)

        return reconstruct_output, mean, logvar



