import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.device = args.device
        self.to(args.device)