import math
import torch.nn as nn
import torch as th

from asd.models.base import BaseModel
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        
        # Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers for channel attention
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (B, C, W) -> B is batch, C is channels, W is input length
        
        # Step 1: Global Average Pooling across the temporal dimension
        avg_out = self.global_avg_pool(x)  # Shape: (B, C, 1)
        
        # Flatten the output from GAP to feed into fully connected layers
        avg_out = avg_out.view(avg_out.size(0), -1)  # Shape: (B, C)
        
        # Step 2: Fully connected layers with ReLU and Sigmoid activation
        out = self.fc1(avg_out)  # Shape: (B, C // reduction_ratio)
        out = self.relu(out)
        out = self.fc2(out)  # Shape: (B, C)
        out = self.sigmoid(out)  # Shape: (B, C)
        
        # Step 3: Reshape the channel attention weights and apply to the input
        out = out.unsqueeze(-1)  # Shape: (B, C, 1)
        
        # Multiply the attention weights with the original input x
        x = x * out  # Element-wise multiplication: (B, C, W)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # Second convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

        # Channel Attention
        self.attention = ChannelAttention(out_channels)


    def forward(self, x):
        residual = self.shortcut(x)

        # Forward pass through the layers
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add residual and apply ReLU
        out += residual
        out = self.relu(out)

        # Apply attention
        out = self.attention(out)

        return out
    

class DifferenceLayer(nn.Module):
    def __init__(self):
        super(DifferenceLayer, self).__init__()

    def forward(self, x):
        # Original input
        original_x = x  # Shape: (B, 4, 256)

        # First-order difference per channel
        first_order_diff = original_x[:, :, 1:] - original_x[:, :, :-1]  # Shape: (B, 4, 255)

        # Second-order difference per channel
        second_order_diff = first_order_diff[:, :, 1:] - first_order_diff[:, :, :-1]  # Shape: (B, 4, 254)

        # Pad the differences to match the original input size
        first_order_diff_padded = nn.functional.pad(first_order_diff, (1, 0))  # Shape: (B, 4, 256)
        second_order_diff_padded = nn.functional.pad(second_order_diff, (2, 0))  # Shape: (B, 4, 256)

        # Concatenate the original signal, first-order, and second-order differences along the channel dimension
        combined_output = th.cat((original_x, first_order_diff_padded, second_order_diff_padded), dim=1)  # Shape: (B, 12, 256)

        return combined_output  # Shape: (B, 12, 256)

class DARLNet(BaseModel):
    def __init__(self, args):
        
        super(DARLNet, self).__init__(args=args)
        # Difference Layer
        self.difference_layer = DifferenceLayer()

        # Backbone layers (Conv1, BatchNorm, ReLU, MaxPool)
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=7, stride=2, padding=0)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # Residual Blocks (Branch 1-1 to Branch 1-4)
        self.layer1 = ResidualBlock(64, 64, stride=1)    # Branch 1-1
        self.layer2 = ResidualBlock(64, 128, stride=2)   # Branch 1-2
        self.layer3 = ResidualBlock(128, 256, stride=2)  # Branch 1-3
        self.layer4 = ResidualBlock(256, 512, stride=1)  # Branch 1-4

        # Global Average Pooling (GAP) for Branch 1
        self.gap = nn.AdaptiveAvgPool1d(1)

        # LSTM cell for Branch 2 (input size = 64, hidden size = 64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)

        # Dropout layers
        self.dropout = nn.Dropout(p=0.6)  # Dropout probability is 50%

        # Fully connected layers after fusion
        self.fc1 = nn.Linear(512 + 64, 64)  # 512 from GAP, 64 from LSTM
        self.fc2 = nn.Linear(64, 2)  # num_classes = 2

    def average_channels(self, data):
        data = data.squeeze(dim=1)
        return data.mean(dim=1, keepdim=True)  # Average across channels, keep the dimension

    def forward(self, x):
        # Ensure the input is on the correct device
        x = x.squeeze(dim=1)
    
        # Backbone: Difference Layer + Conv, BN, ReLU, MaxPool
        x = self.difference_layer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # Output shape: (B, 64, 64)

        # Split the input into two branches

        # Branch 1: Pass through Residual blocks
        branch1 = self.layer1(x)  # Output: (B, 64, 45)
        branch1 = self.layer2(branch1)  # Output: (B, 128, 23)
        branch1 = self.layer3(branch1)  # Output: (B, 256, 12)
        branch1 = self.layer4(branch1)  # Output: (B, 512, 12)
        branch1 = self.gap(branch1)  # Output: (B, 512, 1)
        branch1 = branch1.view(branch1.size(0), -1)  # Flatten to (B, 512)

        # Branch 2: Pass through LSTM
        branch2 = x.permute(0, 2, 1)  # Change input to (B, 64, 64) -> (B, 64, 64) [B, T, C]
        branch2, _ = self.lstm(branch2)  # LSTM output: (B, 64, 64) -> Keep the last hidden state
        branch2 = branch2[:, -1, :]  # Use the last hidden state: (B, 64)

        # Fusion: Concatenate the outputs of both branches
        fused = th.cat((branch1, branch2), dim=1)  # Output: (B, 512 + 64)

        # Apply dropout before fully connected layers
        fused = self.dropout(fused)

        # Fully connected layers
        fused = self.fc1(fused)  # (B, 256)
        fused = self.dropout(fused)  # Dropout before final layer
        output = self.fc2(fused)  # (B, num_classes)
        return output
    
        
    
    
    
if __name__ == "__main__":
    from pytorch_model_summary import summary
    from asd.args import Args
    
    SAVE_TO_FILE = False
    
    device = "cuda" if th.cuda.is_available() else "cpu"
    
    args = Args("./data/configs/default.yaml")
    dummy_input = (th.randn(1, 4, 256))  # Replace with your input dimensions

    model = DARLNet(args=args)
        
    # Print the model summary
    print(summary(model, th.zeros((1, 1, 4, 256)), show_input=False, show_hierarchical=True))
    
    if SAVE_TO_FILE:    
        th.save(model.state_dict(), "./darlnet_model_untrained.pickle")
    
