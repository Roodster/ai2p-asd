import numpy as np
import torch as th
from tqdm import tqdm

from asd.common.utils import set_seed
from asd.writer import Writer
from asd.plots import Plots
from asd.results import Results


class Experiment:
    
    
    def __init__(self, args, learner, label_transformer=None, verbose=False):
        assert learner is not None, "NO learner"

        # ===== DEPENDENCIES =====
        self.args = args
        self.learner = learner
        self.writer = Writer(args=args)
        self.results = Results()
        self.plots = Plots()
        self.label_transformer = label_transformer
        
        self.start_epochs = len(self.results.epochs)
        
         # ===== TRAINING =====
        self.device = args.device
        self.n_epochs = args.n_epochs       
        
        
        # ===== EVALUATION =====
        assert args.eval_interval > 0, "Can't modulo by zero."
        self.eval_interval = args.eval_interval
        assert args.eval_save_model_interval > 0, "Can't modulo by zero."
        self.save_model_interval = args.eval_save_model_interval
        
        self.verbose = verbose
        
        # ===== SEEDING =====
        set_seed(args.seed)
    
    def run(self, train_loader, test_loader, mode='binary'):
        
        assert train_loader is not None, "Please, provide a training dataset : )."
        self.learner.model.train()
        
        self.writer.save_hyperparameters(self.args)

        pbar = tqdm(range(self.start_epochs, self.start_epochs + self.n_epochs))
        
        for epoch in pbar:
            
            loss = .0        

            self.results = self.learner.step(train_loader, results=self.results, verbose=self.verbose)
            
            if (epoch + 1) % self.eval_interval == 0: 
                self.results = self.learner.evaluate(dataloader=test_loader, 
                                                      results=self.results, 
                                                      verbose=self.verbose)
                self.results.epochs = epoch
            
            if (epoch + 1) % self.save_model_interval == 0:
                self.writer.save_model(self.learner.model, epoch+1)        
            
            self.plots.plot(results=self.results, update=True)
    
        self.plots.plot(results=self.results, update=False)
        self.writer.save_statistics(self.results.get())

            