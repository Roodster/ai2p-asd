import numpy as np
import torch as th
import torch.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

from asd.common.utils import set_seed


class Experiment:
    
    
    def __init__(self, args, learner, writer, results, plots):
        assert learner is not None, "NO learner"
        assert writer is not None, "Running an experiment without loggin is a futile endeavor."
        assert results is not None, "Running an experiment without logging is a futile endeavor."


        # ===== DEPENDENCIES =====
        self.args = args
        self.learner = learner
        self.writer = writer
        self.results = results
        self.plots = plots
        
        self.start_epochs = len(self.results.epochs)
        
         # ===== TRAINING =====
        self.device = args.device
        self.n_epochs = args.n_epochs       
        
        
        # ===== EVALUATION =====
        assert args.eval_interval > 0, "Can't modulo by zero."
        self.eval_interval = args.eval_interval
        assert args.eval_save_model_interval > 0, "Can't modulo by zero."
        self.save_model_interval = args.eval_save_model_interval
        
        
        # ===== SEEDING =====
        set_seed(args.seed)
    
    
    def run(self, train_loader, test_loader):
        assert train_loader is not None, "Please, provide a training dataset : )."
        self.learner.model.train()
        
        
        self.writer.save_hyperparameters(self.args)

        pbar = tqdm(range(self.start_epochs, self.start_epochs + self.n_epochs))
        
        for epoch in pbar:
            
            train_loss = 0
            
            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(self.args.device), batch_labels.to(self.args.device)
                outputs = self.learner.predict(batch_data)
                loss = self.learner.compute_loss(y_pred=outputs.float(), y_test=batch_labels.unsqueeze(1))    
                self.learner.update(loss)
                train_loss += loss
            
            if (epoch + 1) % self.eval_interval == 0: 
                self.evaluate_metrics(test_loader)
                self.results.train_losses = train_loss.detach().cpu().numpy()
                self.results.epochs = epoch+1
            
            if (epoch + 1) % self.save_model_interval == 0:
                self.writer.save_model(self.learner.model, epoch+1)        
    
            self.plots.plot(self.results)
    
        self.plots.plot(self.results, update=False)
        self.writer.save_statistics(self.results.get())
        
        
    def evaluate_metrics(self, dataloader):
        self.learner.model.eval()
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        with th.no_grad():
            for images, labels in tqdm(dataloader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.unsqueeze(3)
                ohe_labels = F.one_hot(labels.to(th.int64), num_classes=2)  # Assuming MNIST with 10 classes
                outputs = self.learner.predict(images)

                loss = self.learner.criterion(outputs, ohe_labels.float())
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
        
    def evaluate_samples(self, test_loader):
        """
        Args:
            loader (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.learner.model.eval()
    
        metrics = {
            'tp': 0,
            'fp': 0,
            'tn': 0,
            'fn': 0,
            'loss': 0
        }
        with th.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                outputs = self.learner.model(batch_data)
                outputs, batch_labels = outputs.float().to(self.device), batch_labels.unsqueeze(1).to(self.device)
                
                loss = self.learner.compute_loss(y_pred=outputs, y_test=batch_labels)   
                # tn, fp, fn, tp = np.sum(multilabel_confusion_matrix(batch_labels, outputs),axis=0).ravel()
                binary_predictions = (outputs > 0.5).int()
                
                tn, fp, fn, tp = confusion_matrix(y_true=batch_labels.cpu().data.numpy(), y_pred=binary_predictions.cpu().data.numpy(), labels=[0,1]).ravel()
                metrics['tp'] += tp
                metrics['fp'] += fp
                metrics['fn'] += fn
                metrics['tn'] += tn   
                metrics['loss'] += loss.detach().cpu().numpy()
            
        self.results.tps = metrics['tp']
        self.results.fps = metrics['fp']
        self.results.fns = metrics['fn']
        self.results.tns = metrics['tn']
        self.results.test_losses = metrics['loss']
        
        self.learner.model.train()