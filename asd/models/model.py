import torch.nn as nn
import torch as th



class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
    
class DummyModel(BaseModel):
    
    def __init__(self, args):
        super(DummyModel, self).__init__(args=args)
        self.conv1 = nn.Conv2d(4, 4, kernel_size=(2, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(4, 4, kernel_size=(2, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(4, 4, kernel_size=(2, 1), padding=(1, 0))
        self.pool = nn.MaxPool2d((2, 1))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 1))
        self.fc1 = nn.Linear(4 * 2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = th.relu(self.conv1(x))          
        x = th.relu(self.conv2(x))          
        x = th.relu(self.conv3(x))          
        x = self.adaptive_pool(x)            
        x = x.view(-1, 4 * 2)
        x = th.relu(self.fc1(x))
        x = self.dropout(x)
        x = th.sigmoid(self.fc2(x)).flatten()
        return x