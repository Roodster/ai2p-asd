
class BaseExperiment:
    
    
    def __init__(self):
        pass
    
    def run(self):
        pass



    
class Experiment(BaseExperiment):


    def __init__(self, args, optimizer=None):
        assert optimizer is not None, "No optimizer defined."
        
        # ===== DEPENDENCIES =====
        self.args = args
        self.optimizer = optimizer
            
    def run(self):
        """
            METHOD: 
                run
            DESCRIPTION:
                Runs over number of epochs:
                - samples from dataset
                - makes predictions
                - update gradient based on residual 
                - evaluates on test set
        """
        pass
    
    def evaluate(self):
        pass