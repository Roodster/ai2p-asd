import torch as th
from tqdm import tqdm
import os
import numpy as np

from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

class BaseLearner:
    
    
    def __init__(self, 
                 args=None, 
                 model=None, 
                 optimizer=None,
                 criterion=None
                 ):
        self.device = args.device

        # ===== DEPENDENCIES =====
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def predict(self, batch_data):
        outputs = self.model(batch_data).to(self.args.device)
        return outputs

    def reset(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def compute_loss(self, y_pred, y_test):
        loss =  Variable(self.criterion(y_pred, y_test), requires_grad = True)
        return loss
    
class Learner(BaseLearner):


    def __init__(self, 
                 args=None, 
                 model=None, 
                 optimizer=None,
                 criterion=None,
                 label_transformer=None                 
                 ):
        
        assert args is not None, "No args defined."
        assert model is not None, "No model defined."
        assert optimizer is not None, "No optimizer defined."
        assert criterion is not None, "No criterion defined."
        
        super().__init__(args=args, model=model, optimizer=optimizer, criterion=criterion)

        # ===== DEPENDENCIES =====
        self.label_transformer = label_transformer
                
    def step(self, data_loader, results, verbose=False):
        self.model.train()
        train_loss = .0
        
        for batch_data, batch_labels in data_loader:
            batch_data, batch_labels = batch_data.to(self.args.device), batch_labels.to(self.args.device)
            
            if verbose:
                print(f"Shape of batch_data: {batch_data.shape}")
                print(f"Shape of batch_labels: {batch_labels.shape} ")
            
            outputs = self.predict(batch_data)
        
            if verbose:
                print(f"Shape of outputs: {outputs.shape}")
                
            if self.label_transformer != None:
                batch_labels = self.label_transformer(batch_labels)
                if verbose:
                    print(f"Shape of labels: {batch_labels.shape}")        
        
            loss = self.compute_loss(y_pred=outputs.float(), y_test=batch_labels)    
            self.update(loss)
            train_loss += loss.item()
        results.train_losses = train_loss
        return results
    
    
    def evaluate(self, dataloader, results, verbose=False):
        self.model.eval()
        test_loss = 0.0
        all_labels = []
        all_predictions = []

        with th.no_grad():
            for batch_data, batch_labels in dataloader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                if verbose:
                    print(f"Shape of batch_data: {batch_data.shape}")
                    print(f"Shape of batch_labels: {batch_labels.shape} ")
                
                outputs = self.predict(batch_data)
                
                if verbose:
                    print(f"Shape of outputs: {outputs.shape}")
                
                if self.label_transformer != None:
                    batch_labels = self.label_transformer(batch_labels)
                    if verbose:
                        print(f"Shape of labels: {batch_labels.shape}")  
                
                loss = self.criterion(outputs, batch_labels)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                
                all_labels.extend(batch_labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())


        # Calculate overall metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_predictions, average='macro')
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')


        print('auc: ', auc)
        print('prec: ', overall_precision)
        print('rec: ', overall_recall)
        print('f1: ', overall_f1)
        print('acc: ', accuracy)
        print('loss: ', test_loss)
        # Sum up the values for all classes
        results.aucs = auc
        results.precisions = overall_precision
        results.sensitivities = overall_recall
        results.f1s = overall_f1
        results.accuracies = accuracy

        results.test_losses = test_loss / len(dataloader)
        return results
    
    
class AELearner(Learner):

    def __init__(self, 
                 args=None, 
                 model=None, 
                 optimizer=None,
                 criterion=None
                 ): 
        assert args is not None, "No args defined."
        assert model is not None, "No model defined."
        assert optimizer is not None, "No optimizer defined."
        assert criterion is not None, "No criterion defined."
        
        super().__init__(args=args, 
                         model=model, 
                         optimizer=optimizer, 
                         criterion=criterion)
                
    def step(self, data_loader, results, verbose=False):
        train_loss = .0
        self.model.train()
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(self.args.device)
            
            if verbose:
                print(f"AELearner - Shape of batch_data: {batch_data.shape}")
            
            outputs = self.predict(batch_data)
        
            if verbose:
                print(f"AELearner - Shape of outputs: {outputs.shape}")
                    
            loss = self.compute_loss(y_pred=outputs, y_test=batch_data)    
            self.update(loss)
            train_loss += loss.item()

        
        results.train_losses = train_loss / len(data_loader)

        return results


    def evaluate(self, dataloader, results, verbose=False):
        self.model.eval()
        test_loss = .0
            
        with th.no_grad():
            for batch_data, _ in dataloader:
                batch_data = batch_data.to(self.device)
                
                if verbose:
                    print(f"Shape of batch_data: {batch_data.shape}")
                    
                outputs = self.predict(batch_data)

                if verbose:
                    print(f"Shape of outputs: {outputs.shape}")

                loss = self.criterion(outputs, batch_data)
                test_loss += loss.item()
            
        results.aucs = 0
        results.precisions = 0
        results.sensitivities = 0
        results.f1s = 0
        results.accuracies = 0
        results.test_losses = test_loss / len(dataloader)
        
        return results
        
        
        