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

        # ===== FILE HANDLING =====
        
        # ===== MODEL =====
        
        # ===== EXPERIMENT =====
        self.seed = self.config.get("seed", 1)

        # ===== TRAINING ===== 
        self.batch_size = self.config.get("batch_size", 32)
        
        # ===== EVALUATION =====
        
        # ===== PLOTTING =====

    def default(self):
        return self.__dict__