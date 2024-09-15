import numpy as np
import torch as th
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from tqdm import tqdm


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
    
    
    def run(self, train_loader, test_loader):
        assert train_loader is not None, "Please, provide a training dataset : )."
        self.learner.model.train()
        
        
        self.writer.save_hyperparameters(self.args)

        pbar = tqdm(range(self.start_epochs, self.start_epochs + self.n_epochs))
        
        for epoch in pbar:
            
            train_loss = 0
            
            i = 0
            for batch_data, batch_labels in train_loader:
                outputs = self.learner.predict(batch_data)
                loss = self.learner.compute_loss(y_pred=outputs, y_test=batch_labels)    
                self.learner.update(loss)
                train_loss += loss
            
            if (epoch + 1) % self.eval_interval == 0: 
                self.evaluate_samples(test_loader)
                self.results.train_losses = train_loss.detach().numpy()
                self.results.epochs = epoch+1
            
            if (epoch + 1) % self.save_model_interval == 0:
                self.writer.save_model(self.learner.model, epoch+1)        
    
            self.plots.plot(self.results)
    
        self.plots.plot(self.results, update=False)
        self.writer.save_statistics(self.results.get())
                

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
                outputs = (self.learner.model(batch_data) > 0.5).float()
                loss = self.learner.compute_loss(y_pred=outputs, y_test=batch_labels)   
                # tn, fp, fn, tp = np.sum(multilabel_confusion_matrix(batch_labels, outputs),axis=0).ravel()
                tn, fp, fn, tp = confusion_matrix(y_true=batch_labels, y_pred=outputs, labels=[0,1]).ravel()
                metrics['tp'] += tp
                metrics['fp'] += fp
                metrics['fn'] += fn
                metrics['tn'] += tn   
                metrics['loss'] += loss
            
        self.results.tps = metrics['tp']
        self.results.fps = metrics['fp']
        self.results.fns = metrics['fn']
        self.results.tns = metrics['tn']
        self.results.test_losses = metrics['loss'].detach().numpy()
             