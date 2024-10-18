"""
    Args:
        Reads parameter file and returns arguments in class format.
"""

import torch as th
from asd.common.utils import parse_args

class Args():
    
    def __init__(self, file):
        # ===== Get the configuration from file =====
        self.config = parse_args(file)
        
        # ===== METADATA =====
        self.exp_name = self.config.get("exp_name", "exp")

        # ===== FILE HANDLING =====
        self.log_dir = self.config.get("log_dir", "./logs")
        
        # ===== MODEL =====
        self.model_name = self.config.get("model_name", "dummy")
        
        # ===== DATASET =====
        self.patient_id = str(self.config.get("patient_id", 1)).zfill(2)
        self.train_ratio = self.config.get("train_ratio", 0.7)
        self.test_ratio = self.config.get("test_ratio", 0.15)
        
        # ===== EXPERIMENT =====
        self.seed = self.config.get("seed", 1)
        self.device = self.config.get("device", "cpu")

        # ===== TRAINING ===== 
        self.batch_size = self.config.get("batch_size", 32)
        self.n_epochs = self.config.get("n_epochs", 100)
        self.learning_rate = self.config.get("learning_rate", 1e-3)
        
        # ===== EVALUATION =====
        self.eval_interval = self.config.get("eval_interval", 10)
        self.eval_save_model_interval = self.config.get("eval_save_model_interval", 10)
        self.eval_sample_rate = self.config.get("eval_sample_rate", 1)

        
        # ===== PLOTTING =====

    def default(self):
        return self.__dict__