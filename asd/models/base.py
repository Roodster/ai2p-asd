import torch as th
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.device = args.device
        self.to(args.device)
        
class DetectionModel(BaseModel):

    def __init__(self, args, layers):
        super(DetectionModel).__init__(args=args)
        
        if not isinstance(layers, list):
            layers = [layers]
            
        self.layers = layers
        
    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)
        
        return x




