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

class Writer():
    """
        Based on folder structure:
            ./logs
                T.B.D.
    """
        
    def __init__(self, args):
        
        self.args = args
        self.datetime = str(dt.datetime.now().strftime("%d%m%Y%H%M"))
        self.root = args.log_dir
        self.base_dir = self.root + f"/run_{args.exp_name}_{args.model}_{args.maze_id}_{args.trajectory_length}_{args.train_open_loop_probability}"
        self.train_dir = self.base_dir + f"/seed_{args.seed}_{self.datetime}"
        self.eval_dir = self.base_dir + f"/evaluation_{args.exp_name}"
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
    
    def save_model(self, model, step):
        pass
    
    def save_plot(self, plot, attribute):
        pass
    
    def save_statistics(self, statistics):
        pass
            
    def save_hyperparameters(self, hyperparameters):
        pass
