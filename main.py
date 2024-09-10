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



def main():
        
    # Load arguments
    args = Args(file="./data/configs/default.yaml")
    
    # Load dataset
    train_dataset = SegmentsDataset("./data/dataset/train/preprocessed/", mode='train', patient_id=args.patient_id)
    test_dataset = SegmentsDataset("./data/dataset/train/preprocessed/", mode='test', patient_id=args.patient_id)
    
    # Instantiate dataloaders 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    writer = Writer(args=args)
    
    
    model = DummyModel(args=args)    
    # if u want to load an existing model. use this. Dont forget to add the corresponding results file to the results
    model.load_state_dict(th.load(".\logs\\run_dev_dummy\seed_1_eval_01\models\model_dummy_2.pickle", weights_only=True))
    
    optimizer = th.optim.Adam(params=model.parameters(), 
                              lr=args.learning_rate)
    
    criterion = nn.BCELoss()
    
    results = Results(file="./logs/run_dev_dummy/seed_1_eval_01/stats_dev_dummy.csv")
    
    
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


if __name__ == "__main__":

    # import cProfile
    # cProfile.run('main()')
    main()