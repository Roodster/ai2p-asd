import torch.nn as nn
import torch as th



class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    
class DummyModel(BaseModel):
    
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0))
        self.pool = nn.MaxPool2d((2, 1))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 1))
        self.fc1 = nn.Linear(64 * 16, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = th.relu(self.conv1(x))          # Output: [32, 129, 1]
        x = th.relu(self.conv2(x))          # Output: [64, 129, 1]
        x = th.relu(self.conv3(x))          # Output: [64, 129, 1]
        x = self.adaptive_pool(x)              # Output: [64, 16, 1]
        x = x.view(-1, 64 * 16)
        x = th.relu(self.fc1(x))
        x = self.dropout(x)
        x = th.sigmoid(self.fc2(x)).flatten()
        return x