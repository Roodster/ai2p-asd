"""
    Writer:
        Writes data to files.
"""
import os
import torch as th
import matplotlib.pyplot as plt
import json
import datetime as dt
import yaml
import pandas as pd

class Writer():
    """
        Based on folder structure:
            ./logs
               ./exp_<exp-name>_<model>
                    ./seed_<seed>_<datetime>
                        ./models
                    ./evaluation_<exp-name>_<model>
                    
    """
        
    def __init__(self, args):
        
        self.args = args
        self.datetime = str(dt.datetime.now().strftime("%d%m%Y%H%M"))
        self.root = args.log_dir
        self.base_dir = self.root + f"/run_{args.exp_name}_{args.model}"
        self.train_dir = self.base_dir + f"/seed_{args.seed}_{self.datetime}"
        self.eval_dir = self.base_dir + f"/evaluation_{args.exp_name}_{args.model}"
        self.model_dir = self.train_dir + "/models"
        
        self._create_directories(self.base_dir)
        self._create_directories(self.train_dir)
        self._create_directories(self.eval_dir)
        self._create_directories(self.model_dir)

    def _create_directories(self, path):
        do_exist = os.path.exists(path)
        if not do_exist:
            # Create a new directory because it does not exist
            os.makedirs(path)
    
    def save_model(self, model, epoch):
        _dir = os.path.join(self.model_dir)
        file = f"/model_{self.datetime}_{self.args.exp_name}_{self.args.model}_{epoch}.pickle"

        full_path = _dir + file
        th.save(model.state_dict(), full_path)
    
    def save_plot(self, plot, attribute):
        filepath = f"/plot_{self.args.exp_name}_{self.args.model}_{attribute}.png"
        
        plot_path = self.train_dir + filepath
        
        plot.savefig(plot_path)
    
    def save_statistics(self, statistics):
        filepath = f"/stats_{self.args.exp_name}_{self.args.model}.json"
        stats_path = self.train_dir + filepath
        statistics.to_csv(stats_path, index=False)
            
    def save_hyperparameters(self, hyperparameters):
        filepath = f"/hyperparameters_{self.args.exp_name}_{self.args.model}.yaml"

        hyperparams_path = self.train_dir + filepath
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparameters.__dict__,
                      f,
                      indent=4,
                      sort_keys=True,
                      default=str)
