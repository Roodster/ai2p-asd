import torch as th
from tqdm import tqdm
import os
import numpy as np

from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from asd.event_scoring.annotation import Annotation
from asd.event_scoring.scoring import EventScoring
from asd.plots import EventPlots

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
        self.model = model.to(self.args.device)
        self.optimizer = optimizer
        self.criterion = criterion
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def predict(self, batch_data):
        outputs = self.model(batch_data)
        return outputs

    def reset(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def compute_loss(self, y_pred, y_test):
        loss =  self.criterion(y_pred, y_test)
        return loss
    
class Learner(BaseLearner):
    mistakes = []

    def __init__(self, 
                 args=None, 
                 model=None, 
                 optimizer=None,
                 criterion=None,
                 label_transformer=None,
                 event_scoring=False
                 ):
        
        assert args is not None, "No args defined."
        assert model is not None, "No model defined."
        assert optimizer is not None, "No optimizer defined."
        assert criterion is not None, "No criterion defined."
        
        super().__init__(args=args, model=model, optimizer=optimizer, criterion=criterion)

        # ===== DEPENDENCIES =====
        self.label_transformer = label_transformer
        self.event_scoring = event_scoring
        
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

                    
            loss = self.compute_loss(y_pred=outputs, y_test=batch_labels)    
            self.update(loss=loss)
            train_loss += loss.item()

        results.train_losses = train_loss / len(data_loader)
        return results
    
    
    def evaluate(self, dataloader, results, verbose=False):
        self.model.eval()
        test_loss = 0.0
        all_labels = []
        all_predictions = []

        with th.no_grad():
            for batch_data, batch_labels in dataloader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                true_labels = batch_labels.detach().clone()
                
                if verbose:
                    print(f"Shape of batch_data: {batch_data.shape}")
                    print(f"Shape of batch_labels: {batch_labels.shape} ")
                
                outputs = self.predict(batch_data)
                
                if verbose:
                    print(f"Shape of outputs: {outputs.shape}")
                    print(f'Outputs: \n {outputs}')
                if self.label_transformer != None:
                    true_labels = self.label_transformer(batch_labels)
                    if verbose:
                        print(f"Shape of labels: {true_labels.shape}")  
                
        
                loss = self.compute_loss(y_pred=outputs, y_test=true_labels)
                test_loss += loss.item()

                if len(outputs.shape) == 1:
                    outputs = (outputs > 0.5).int()

                if len(outputs.shape) == 2:
                    _, outputs = outputs.max(1)
    
                if verbose:
                    print(f"Shape of outputs: {outputs.shape}")
                    print(f'Outputs: \n {outputs}')
                    
                all_labels.extend(batch_labels.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())
          
        if(self.event_scoring):
            scores = EventScoring(all_labels, all_predictions,  fs=self.args.eval_sample_rate)
            ref = Annotation(all_labels, fs=self.args.eval_sample_rate)
            hyp = Annotation(all_predictions, fs=self.args.eval_sample_rate)
            EventPlots().plotEventScoring(ref, hyp)
            # plotIndividualEvents(ref, hyp)
            results.fp_rates = scores.fpRate
            results.precisions = scores.precision
            results.sensitivities = scores.sensitivity
            results.f1s = scores.f1
            print("Any-overlap Performance Metrics:")
            print(f"Sensitivity: {scores.sensitivity:.4f}" if not np.isnan(scores.sensitivity) else "Sensitivity: NaN")
            print(f"Precision: {scores.precision:.4f}" if not np.isnan(scores.precision) else "Precision: NaN")
            print(f"F1 Score: {scores.f1:.4f}" if not np.isnan(scores.f1) else "F1 Score: NaN")
            print(f"False Positive Rate (FP/day): {scores.fpRate:.4f}")
            
        else:
            # Calculate overall metrics
            accuracy = accuracy_score(all_labels, all_predictions)
            auc = roc_auc_score(all_labels, all_predictions, average='macro')
            overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
            results.aucs = auc
            results.precisions = overall_precision
            results.sensitivities = overall_recall
            results.f1s = overall_f1
            results.accuracies = accuracy
    
        results.test_losses = test_loss / len(dataloader)
        return results
    
    

class DARLNetLearner(BaseLearner):
    mistakes = []

    def __init__(self, 
                 args=None, 
                 model=None, 
                 optimizer=None,
                 criterion=None,
                 label_transformer=None,
                 event_scoring=False
                 ):
        
        assert args is not None, "No args defined."
        assert model is not None, "No model defined."
        assert optimizer is not None, "No optimizer defined."
        assert criterion is not None, "No criterion defined."
        
        super().__init__(args=args, model=model, optimizer=optimizer, criterion=criterion)

        # ===== DEPENDENCIES =====
        self.label_transformer = label_transformer
        self.event_scoring = event_scoring
        
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

                    
            loss = self.compute_loss(y_pred=outputs, y_test=batch_labels.long())    
            self.update(loss=loss)
            train_loss += loss.item()

        results.train_losses = train_loss / len(data_loader)
        return results

    
    def evaluate(self, dataloader, results, verbose=False):
        self.model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        all_labels = []
        all_predictions = []
        all_raw_outputs = []
        best_weights = None  # Initialize to store best weights

        with th.no_grad():  # Disable gradient computation for evaluation
            for batch_data, batch_labels in dataloader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                true_labels = batch_labels.detach().clone()

                if verbose:
                    print(f"Shape of batch_data: {batch_data.shape}")
                    print(f"Shape of batch_labels: {batch_labels.shape} ")

                # Forward pass: Get predictions from the model
                outputs = self.model(batch_data)  # Outputs are already in [0, 1] due to sigmoid

                # Move outputs back to CPU for further processing
                outputs = outputs.cpu().numpy()

                all_raw_outputs.extend([row[1] for row in outputs])
                all_labels.extend(batch_labels.cpu().numpy())  # Collect true labels

                # Compute loss
                loss = self.compute_loss(y_pred=th.tensor(outputs).to(self.device), y_test=true_labels.long())
                test_loss += loss.item()

        # If event_scoring is True, find the best threshold based on F1-score using scores.f1
        if self.event_scoring:
            best_f1 = 0
            best_threshold = 0.5
            thresholds = np.arange(0.1, 1.0, 0.05)  # Range of thresholds to evaluate

            for threshold in thresholds:
                # Apply the threshold to raw outputs to get binary predictions
                predictions = (np.array(all_raw_outputs) > threshold).astype(float)

                # Calculate scores using your EventScoring class for the current threshold
                scores = EventScoring(all_labels, predictions, fs=self.args.eval_sample_rate)

                print(f"Threshold: {threshold:.2f}, F1 Score: {scores.f1:.4f}")

                # Keep track of the best threshold and save weights if F1 score is higher
                if scores.f1 >= best_f1:
                    best_f1 = scores.f1
                    best_threshold = threshold
                    best_weights = self.model.state_dict()  # Save the best weights
                    
            print(f"Best Threshold: {best_threshold:.2f}, F1 Score: {best_f1:.4f}")

            # Finalize predictions using the best threshold
            final_predictions = (np.array(all_raw_outputs) > best_threshold).astype(int)
            final_predictions = np.squeeze(final_predictions)  # Ensure predictions are binary and flat

            # Calculate final event scoring metrics based on the best threshold
            scores = EventScoring(all_labels, final_predictions, fs=self.args.eval_sample_rate)
            ref = Annotation(all_labels, fs=self.args.eval_sample_rate)
            hyp = Annotation(final_predictions, fs=self.args.eval_sample_rate)
            EventPlots().plotEventScoring(ref, hyp)
            results.fp_rates = scores.fpRate
            results.precisions = scores.precision
            results.sensitivities = scores.sensitivity
            results.f1s = scores.f1
            print("Any-overlap Performance Metrics:")
            print(f"Sensitivity: {scores.sensitivity:.4f}" if not np.isnan(scores.sensitivity) else "Sensitivity: NaN")
            print(f"Precision: {scores.precision:.4f}" if not np.isnan(scores.precision) else "Precision: NaN")
            print(f"F1 Score: {scores.f1:.4f}" if not np.isnan(scores.f1) else "F1 Score: NaN")
            print(f"False Positive Rate (FP/day): {scores.fpRate:.4f}")

        else:
            # Use default threshold (0.5) for binary classification if event_scoring is False
            final_predictions = (np.array(all_raw_outputs) > 0.5).astype(int)
            final_predictions = np.squeeze(final_predictions)  # Ensure predictions are binary and flat

            # Calculate overall metrics
            accuracy = accuracy_score(all_labels, final_predictions)
            auc = roc_auc_score(all_labels, final_predictions, average='macro')
            overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_labels, final_predictions, average='macro')
            results.aucs = auc
            results.precisions = overall_precision
            results.sensitivities = overall_recall
            results.f1s = overall_f1
            results.accuracies = accuracy

        results.test_losses = test_loss / len(dataloader)
        
        # Return results, best threshold, and best weights
        return results, best_threshold, best_weights



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

                loss = self.compute_loss(y_pred=outputs, y_test=batch_data)
                test_loss += loss.item()
            
        results.aucs = 0
        results.precisions = 0
        results.sensitivities = 0
        results.f1s = 0
        results.accuracies = 0
        results.test_losses = test_loss / len(dataloader)
        
        return results
        
        
        
class SSLLearner(AELearner):

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
        
        
    def compute_loss(self, outputs):
        # probably some more preprcocessing here such as splitting.
        return self.criterion(outputs)        
        
        
    def step(self, data_loader, results, verbose=False):
        train_loss = .0
        self.model.train()
        for batch_data, _ in data_loader:

            batch_data = batch_data.to(self.args.device)
            
            if verbose:
                print(f"Shape of batch_data: {batch_data.shape}")
            
            outputs = self.predict(batch_data)
                    
            loss = self.compute_loss(outputs=outputs)    
            self.update(loss)
            train_loss += loss.item()


        if verbose:
            print(f"Train loss: {train_loss}")
                    
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

                loss = self.compute_loss(outputs=outputs)
                test_loss += loss.item()

        if verbose:
            print(f"Test loss: {test_loss}")
            
        results.aucs = 0
        results.precisions = 0
        results.sensitivities = 0
        results.f1s = 0
        results.accuracies = 0
        results.test_losses = test_loss / len(dataloader)
        
        return results