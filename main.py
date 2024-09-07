from asd.make_dataset import get_dataloaders, SpectrogramDataset
from asd.args import Args
from asd.models.model import DummyModel
from asd.learner import Learner
import torch as th
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    
    channels = "FP1-F7;F7-T7;T7-P7;P7-O1;FP1-F3;F3-C3;C3-P3;P3-O1;FP2-F4;F4-C4;C4-P4;P4-O2;FP2-F8;F8-T8;T8-P8;T8-P8;P8-O2;FZ-CZ;CZ-PZ;P7-T7;T7-FT9;FT9-FT10;FT10-T8;T8-P8".split(";")

    # Load arguments
    args = Args(file="./data/configs/default.yaml")
    # Load dataset
    
    dataset = SpectrogramDataset('./data/temp/')
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset,
        train_ratio=0.8,
        test_ratio=0.1,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    model = DummyModel()    

    optimizer = th.optim.Adam(params=model.parameters(), 
                              lr=args.learning_rate)
    
    criterion = nn.BCELoss()
        
    learner = Learner(args=args, 
                      model=model, 
                      train_loader=train_loader, 
                      val_loader=val_loader, 
                      test_loader=test_loader, 
                      optimizer=optimizer,
                      criterion=criterion
                )
    
    learner.train()