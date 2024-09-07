import torch.nn as nn
import torch as th



class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    
class DummyModel(BaseModel):
    
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 32, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(th.relu(self.conv1(x)))
        x = self.pool(th.relu(self.conv2(x)))
        x = th.relu(self.conv3(x))
        x = x.view(-1, 64 * 6 * 32)
        x = th.relu(self.fc1(x))
        x = self.dropout(x)
        x = th.sigmoid(self.fc2(x)).flatten()
        return x
