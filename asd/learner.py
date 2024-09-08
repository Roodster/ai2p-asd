import torch as th
from tqdm import tqdm
import os

class BaseLearner:
    
    
    def __init__(self):
        pass
    
    def run(self):
        pass

    
class Learner(BaseLearner):


    def __init__(self, 
                 args=None, 
                 model=None, 
                 train_loader=None, 
                 val_loader=None, 
                 test_loader=None, 
                 optimizer=None,
                 criterion=None,
                 ):
        
        assert args is not None, "No args defined."
        assert model is not None, "No model defined."
        assert train_loader is not None, "No train_loader defined."
        assert val_loader is not None, "No val_loader defined."
        assert test_loader is not None, "No test_loader defined."
        assert optimizer is not None, "No optimizer defined."
        assert criterion is not None, "No criterion defined."
        
        super().__init__()


        self.device = args.device

        # ===== DEPENDENCIES =====
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        
        # ===== TRAINING =====
        self.n_epochs = args.n_epochs
            
            
    def train(self):
        
        self.model.train()

        pbar = tqdm(range(self.n_epochs))
    
        for epoch in pbar:
            # Now you can use the dataloader in your training loop
            
            train_loss = 0
            
            for iteration, (batch_data, batch_labels) in enumerate(self.train_loader):
                pbar.set_description(f"batch_progress={iteration/len(self.train_loader):.3f}%")
                # Your training code here
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
            if (epoch + 1) % self.args.eval_save_model_interval == 0:
                if not os.path.exists(f'./logs/models/'):
                    os.makedirs(f'./logs/models/')

                th.save(self.model.state_dict(), f'./logs/models/model_weights_{epoch}.pth')
            
            pbar.set_description(f"progress={epoch+1}/{self.n_epochs}% loss={train_loss:.4f}")


    def evaluate(self, loader):
        self.model.eval()
        
        with th.no_grad():
            for batch_data, batch_labels in loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                _ = self.model(batch_data)



        