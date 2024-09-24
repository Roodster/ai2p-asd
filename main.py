from torch.utils.data import DataLoader


from asd.dataset import get_dataloaders, OnlineSegmentsDataset, OfflineSegmentsDataset, DummyDataset
from asd.args import Args
from asd.writer import Writer
from asd.results import Results
from asd.models.model import Model
from asd.models.transformer import VisionTransformer
from asd.models.autoencoder import ShallowAE, BetaVAE
from asd.learner import Learner, AELearner
from asd.models.losses.beta_vae_loss import BetaVAELoss
from asd.labels import OHELabelTransformer
from asd.experiment import Experiment
import torch as th
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')



def main():
        
    # Load arguments
    args = Args(file="./data/configs/default.yaml")
    
    # # Load dataset
    train_dataset = OfflineSegmentsDataset("./data/dataset/train/partial-4-signals/", mode='train', patient_id=args.patient_id)
    test_dataset = OfflineSegmentsDataset("./data/dataset/test/partial-4-signals/chb01", mode='test', patient_id=args.patient_id)
    
    # # train_dataset = DummyDataset(num_classes=2, n_samples_per_class=4096, x=1, y=4, z=256)
    # test_dataset = DummyDataset(num_classes=2, n_samples_per_class=512, x=1, y=4, z=256)    
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = BetaVAE(args=args, latent_dim=16, n_channels=1)
    # if u want to load an existing model. use this. Dont forget to add the corresponding results file to the results
    # model.load_state_dict(th.load(".\logs\\run_dev_dummy\seed_1_eval_01\models\model_dummy_2.pickle", weights_only=True))
    
    optimizer = th.optim.Adam(params=model.parameters(), 
                              lr=args.learning_rate)
    
    criterion = BetaVAELoss(beta=1)
    
    
    # This transformer can be added to Experiment(..., label_transformer=transformer) to encode the labels to OHE
    # transformer = OHELabelTransformer()
    
    
    results = Results(verbose=False)
    learner = AELearner(args=args, 
                      model=model, 
                      optimizer=optimizer,
                      criterion=criterion
                )
    
    experiment = Experiment(args=args, 
                            learner=learner,
                            results=results,
                            verbose=False)
    
    
    experiment.run(train_loader=train_loader, test_loader=test_loader)


if __name__ == "__main__":

    # import cProfile
    # cProfile.run('main()')
    main()