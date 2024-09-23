import numpy as np
import torch as th
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

from asd.common.utils import set_seed


class Experiment:
    
    
    def __init__(self, args, learner, writer, results, plots, label_transformer=None, verbose=False):
        assert learner is not None, "NO learner"
        assert writer is not None, "Running an experiment without loggin is a futile endeavor."
        assert results is not None, "Running an experiment without logging is a futile endeavor."


        # ===== DEPENDENCIES =====
        self.args = args
        self.learner = learner
        self.writer = writer
        self.results = results
        self.plots = plots
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
            
            train_loss = .0          
            
            if mode in ('binary', 'bin', 'b'):
                train_loss =  self.classifier_step(train_loader=train_loader)
            elif mode in ('autoencoder', 'ae'):
                train_loss = self.autoencoder_step(train_loader=train_loader)  
                    
            self.results.train_losses = train_loss
            
            if (epoch + 1) % self.eval_interval == 0: 
                
                if mode in ('binary', 'bin', 'b'):
                    test_loss =  self.evaluate_metrics(dataloader=test_loader)
                elif mode in ('autoencoder', 'ae'):
                    test_loss = self.evaluate_autoencoder(dataloader=test_loader)  
                
                self.results.test_losses = test_loss
                self.results.epochs = epoch+1
            
            if (epoch + 1) % self.save_model_interval == 0:
                self.writer.save_model(self.learner.model, epoch+1)        
            self.plots.plot(self.results)
    
        self.plots.plot(self.results, update=False)
        self.writer.save_statistics(self.results.get())


    def autoencoder_step(self, train_loader):
        
        train_loss = .0
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(self.args.device)
            if self.verbose:
                print(f"Shape of batch_data: {batch_data.shape}")
            outputs = self.learner.predict(batch_data)        
            loss = self.learner.compute_loss(y_pred=outputs.float(), y_test=batch_data)    
            self.learner.update(loss)
            train_loss += loss.item()
        
        return train_loss 

    def classifier_step(self, train_loader):
        train_loss = .0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(self.args.device), batch_labels.to(self.args.device)
            
            if self.verbose:
                print(f"Shape of batch_data: {batch_data.shape}")
                print(f"Shape of batch_labels: {batch_labels.shape} ")
            
            outputs = self.learner.predict(batch_data)
        
            if self.verbose:
                print(f"Shape of outputs: {outputs.shape}")
                
            if self.label_transformer != None:
                labels = self.label_transformer(batch_labels)
                if self.verbose:
                    print(f"Shape of labels: {labels.shape}")        
        
            loss = self.learner.compute_loss(y_pred=outputs.float(), y_test=labels)    
            self.learner.update(loss)
            train_loss += loss.item()
            
        return train_loss
        
    
    def evaluate_autoencoder(self, dataloader):
        self.learner.model.eval()
        test_loss = .0
            
        with th.no_grad():
            for batch_data, _ in dataloader:
                batch_data = batch_data.to(self.device)
                
                if self.verbose:
                    print(f"Shape of batch_data: {batch_data.shape}")
                    
                outputs = self.learner.predict(batch_data)

                if self.verbose:
                    print(f"Shape of outputs: {outputs.shape}")

                loss = self.learner.criterion(outputs, batch_data)
                test_loss += loss.item()
        
        self.results.aucs = 0
        self.results.precisions = 0
        self.results.sensitivities = 0
        self.results.f1s = 0
        self.results.accuracies = 0
        self.results.test_losses = test_loss / len(dataloader)
        
        
    
    def evaluate_metrics(self, dataloader):
        self.learner.model.eval()
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        with th.no_grad():
            for batch_data, batch_labels in dataloader:
                batch_data, labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                if self.verbose:
                    print(f"Shape of batch_data: {batch_data.shape}")
                    print(f"Shape of batch_labels: {batch_labels.shape} ")
                
                outputs = self.learner.predict(batch_data)
                
                if self.verbose:
                    print(f"Shape of outputs: {outputs.shape}")
                
                if self.label_transformer != None:
                    labels = self.label_transformer(batch_labels)
                    if self.verbose:
                        print(f"Shape of labels: {labels.shape}")  
                
                loss = self.learner.criterion(outputs, labels.float())
                running_loss += loss.item()

                _, predicted = outputs.max(1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())


        # Calculate overall metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_predictions, average='macro')
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')

        # Sum up the values for all classes
        self.results.aucs = auc
        self.results.precisions = overall_precision
        self.results.sensitivities = overall_recall
        self.results.f1s = overall_f1
        self.results.accuracies = accuracy
        self.results.test_losses = running_loss / len(dataloader)