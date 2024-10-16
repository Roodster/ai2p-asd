import numpy as np
import torch as th
from tqdm import tqdm

from asd.common.utils import set_seed
from asd.writer import Writer
from asd.plots import Plots, EventPlots
from asd.results import Results, EventResults


class Experiment:
    
    
    def __init__(self, args, learner, results, label_transformer=None, do_plot=True, verbose=False, event_scoring=False):
            assert learner is not None, "NO learner"

            # ===== DEPENDENCIES =====
            self.event_scoring=event_scoring
            self.args = args
            self.learner = learner
            self.results = results
            self.writer = Writer(args=args)
            self.plots = EventPlots() if event_scoring else Plots()
            self.label_transformer = label_transformer
            
            self.start_epochs = len(self.results.epochs)
            self.last_epoch = self.start_epochs
            
            # ===== TRAINING =====
            self.device = args.device
            self.n_epochs = args.n_epochs       
            
            
            # ===== EVALUATION =====
            assert args.eval_interval > 0, "Can't modulo by zero."
            self.eval_interval = args.eval_interval
            assert args.eval_save_model_interval > 0, "Can't modulo by zero."
            self.save_model_interval = args.eval_save_model_interval
            
            self.verbose = verbose
            self.do_plot = do_plot
            
            # ===== SEEDING =====
            set_seed(args.seed)
        
        def run(self, train_loader, test_loader):
            assert train_loader is not None, "Please, provide a training dataset :)."        
            assert test_loader is not None, "Please, provide a test dataset :)."        

            self.writer.save_hyperparameters(self.args)

            pbar = tqdm(range(self.last_epoch, self.last_epoch + self.n_epochs))
            
            for epoch in pbar:
                self.last_epoch += 1
                self.results = self.learner.step(train_loader, results=self.results, verbose=self.verbose)
                
                if (epoch + 1) % self.eval_interval == 0: 
                    self.results = self.learner.evaluate(dataloader=test_loader, 
                                                        results=self.results, 
                                                        verbose=self.verbose)
                    self.results.epochs = epoch
                
                if (epoch + 1) % self.save_model_interval == 0:
                    self.writer.save_model(self.learner.model, epoch+1)        
            
                if self.verbose:
                    self.results.print()
                    
                if self.do_plot:
                    self.plots.plot(results=self.results, update=True)
        
            if self.do_plot:
                self.plots.plot(results=self.results, update=False)
            self.writer.save_statistics(self.results.get())

                