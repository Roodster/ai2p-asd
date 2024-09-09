from torch.utils.data import DataLoader


from asd.dataset import get_dataloaders, SegmentsDataset
from asd.args import Args
from asd.writer import Writer
from asd.results import Results
from asd.models.model import DummyModel
from asd.learner import Learner
from asd.experiment import Experiment
import torch as th
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    
    channels = "FP1-F7;F7-T7;T7-P7;P7-O1;FP1-F3;F3-C3;C3-P3;P3-O1;FP2-F4;F4-C4;C4-P4;P4-O2;FP2-F8;F8-T8;T8-P8;T8-P8;P8-O2;FZ-CZ;CZ-PZ;P7-T7;T7-FT9;FT9-FT10;FT10-T8;T8-P8".split(";")

    # Load arguments
    args = Args(file="./data/configs/default.yaml")
    # Load dataset
    
    train_dataset = SegmentsDataset("./data/dataset/train/preprocessed/", mode='train', patient_id='01')
    test_dataset = SegmentsDataset("./data/dataset/train/preprocessed/", mode='test', patient_id='01')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    writer = Writer(args=args)
    model = DummyModel(args=args)    

    optimizer = th.optim.Adam(params=model.parameters(), 
                              lr=args.learning_rate)
    
    criterion = nn.BCELoss()
    
    results = Results()
    
    learner = Learner(args=args, 
                      model=model, 
                      optimizer=optimizer,
                      criterion=criterion
                )
    
    experiment = Experiment(args=args, 
                            learner=learner,
                            writer=writer,
                            results=results)
    
    
    experiment.run(train_loader=train_loader, test_loader=test_loader)