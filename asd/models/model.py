import torch.nn as nn
import torch as th



class Model:
    
    @classmethod
    def create(cls, args):
        if args.model_name == 'dummy':
            return DummyModel(args)
        elif args.model_name == 'cnn-bilstm':
            return CNNBiLSTM(args)


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.device = args.device
        self.to(args.device)
    
class DummyModel(BaseModel):
    
    def __init__(self, args):
        super(DummyModel, self).__init__(args=args)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=(2, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(2, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(2, 1), padding=(1, 0))
        self.pool = nn.MaxPool2d((2, 1))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 1))
        self.fc1 = nn.Linear(64 * 2, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.to(self.device)
        x = th.relu(self.conv1(x))          
        x = th.relu(self.conv2(x))          
        x = th.relu(self.conv3(x))          
        x = self.adaptive_pool(x)            
        x = x.view(-1, 4 * 2)
        x = th.relu(self.fc1(x))
        x = self.dropout(x)
        x = th.sigmoid(self.fc2(x)).flatten()
        return x
    
    
    
class CNNBiLSTM(BaseModel):
        
    def __init__(self, args):
        super(CNNBiLSTM, self).__init__(args=args)
        
        # Define CNN layers
        self.conv1 = nn.Conv2d(4, 64, kernel_size=(3, 1), padding=(1, 0)).to(self.device)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(3, 1), padding=(1, 0)).to(self.device)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=(3, 1), padding=(1, 0)).to(self.device)
        self.pool = nn.MaxPool2d((3, 1)).to(self.device)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 1)).to(self.device)

        # LSTM parameters
        self.lstm_input_size = 16  # Based on CNN output channels
        self.hidden_size = 64     # LSTM hidden state size
        self.num_layers = 3       # Number of LSTM layers

        # Define Bidirectional LSTM
        self.bilstm = nn.LSTM(input_size=self.lstm_input_size, 
                              hidden_size=self.hidden_size, 
                              num_layers=self.num_layers, 
                              batch_first=True, 
                              bidirectional=True).to(self.device)

        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_size * 2, 32).to(self.device)  # Bi-directional, hence *2
        self.fc2 = nn.Linear(32, 1).to(self.device)
        self.dropout = nn.Dropout(0.5).to(self.device)

    def forward(self, x):
        x = x.to(self.device)

        # Pass input through CNN layers
        x = th.relu(self.conv1(x))
        x = th.relu(self.conv2(x))
        x = th.relu(self.conv3(x))
        x = self.adaptive_pool(x)  # Shape: (batch_size, channels, height, width)
        
        # Reshape the output to fit LSTM input (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3)  # Reshape to (batch_size, height, channels, width)
        x = x.view(batch_size, x.size(1), -1)  # Flatten to (batch_size, sequence_length, input_size)
        
        # Pass through LSTM layer
        lstm_out, _ = self.bilstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_size * 2)
        
        # Use the output from the last time step
        lstm_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size * 2)
        
        # Fully connected layers
        x = th.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = th.sigmoid(self.fc2(x)).flatten()
        
        return x
