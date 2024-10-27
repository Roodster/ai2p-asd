import numpy as np
import torch as th
from tqdm import tqdm

from asd.common.utils import set_seed
from asd.writer import Writer
from asd.plots import Plots, EventPlots
from asd.results import Results, EventResults
from asd.event_scoring.scoring import EventScoring
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from asd.event_scoring.annotation import Annotation


class Experiment:
    
    def __init__(self, args, learner=None, results=None, label_transformer=None, do_plot=True, verbose=False, event_scoring=False):
            # ===== DEPENDNCIES =====
            self.event_scoring=event_scoring
            self.args = args
            self.learner = learner
            self.results = results
            self.writer = Writer(args=args)
            self.plots = EventPlots() if event_scoring else Plots()
            self.label_transformer = label_transformer
            
            self.start_epochs = len(self.results.epochs) if self.results else None
            self.last_epoch = self.start_epochs
            
            # ===== TRAINING =====
            self.device = args.device
            self.n_epochs = args.n_epochs       
            
            
            # ===== EVALUATION =====
            assert args.eval_interval > 0, "Can't modulo by zero."
            self.eval_interval = args.eval_interval
            assert args.eval_save_model_interval > 0, "Can't modulo by zero."
            self.save_model_interval = args.eval_save_model_interval
            
            self.verbose = verbose
            self.do_plot = do_plot
            
            # ===== SEEDING =====
            set_seed(args.seed)
        
    def run(self, train_loader, test_loader):
        assert train_loader is not None, "Please, provide a training dataset :)."
        assert test_loader is not None, "Please, provide a test dataset :)."
        
        self.writer.save_hyperparameters(self.args)

        # Initialize arrays to store thresholds and weights
        thresholds = []
        weights = []
        
        pbar = tqdm(range(self.last_epoch, self.last_epoch + self.n_epochs))
        
        for epoch in pbar:
            self.last_epoch += 1
            self.results = self.learner.step(train_loader, results=self.results, verbose=self.verbose)
            
            if (epoch + 1) % self.eval_interval == 0: 
                self.results, current_threshold, current_weights = self.learner.evaluate(
                    dataloader=test_loader, results=self.results, verbose=self.verbose
                )
                # Append current threshold and weights to arrays
                thresholds.append(current_threshold)
                weights.append(current_weights)
                self.results.epochs = epoch
            
            if (epoch + 1) % self.save_model_interval == 0:
                self.writer.save_model(self.learner.model, epoch + 1)
            
            if self.verbose:
                self.results.print()
                
            if self.do_plot:
                self.plots.plot(results=self.results, update=True)
        
        if self.do_plot:
            self.plots.plot(results=self.results, update=False)
        
        # Save the results statistics
        self.writer.save_statistics(self.results.get())

        # Find the index of the highest threshold
        max_threshold_idx = thresholds.index(max(thresholds))
        
        # Return the threshold and weights corresponding to the highest threshold
        return thresholds[max_threshold_idx], weights[max_threshold_idx]

    
    
    def evaluate_predictions(self, model, dataloader, thresholds=None, verbose=False):
        if thresholds is None:
            thresholds = [0.5]  # Default threshold if none provided

        model = model.to(self.device)
        model.eval()
        all_labels = []
        all_predictions = []

        best_threshold = None
        best_auc = -1  # Initialize with a low AUC to identify the best later
        threshold_scores = []  # To store (threshold, AUC) tuples
        evaluation_count = 0

        with th.no_grad():
            for threshold in thresholds:
                evaluation_count += 1
                current_labels = []
                current_predictions = []

                for batch_data, batch_labels in dataloader:
                    batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                    true_labels = batch_labels.detach().clone()

                    if verbose:
                        print(f"Shape of batch_data: {batch_data.shape}")
                        print(f"Shape of batch_labels: {batch_labels.shape}")

                    # Get model predictions
                    outputs = model(batch_data).to(self.device)

                    if verbose:
                        print(f"Shape of outputs: {outputs.shape}")
                        print(f'Outputs: \n {outputs}')

                    # Apply label transformer if present
                    if self.label_transformer is not None:
                        true_labels = self.label_transformer(batch_labels)
                        if verbose:
                            print(f"Shape of labels after transformation: {true_labels.shape}")

                    # Convert outputs to class predictions (for binary or multi-class)
                    if len(outputs.shape) == 1:  # Binary classification
                        outputs = (outputs > threshold).int()
                    if len(outputs.shape) == 2:  # Multi-class classification
                        outputs = th.tensor([1 if output[1] > threshold else 0 for output in outputs])
                    if verbose:
                        print(f"Shape of processed outputs: {outputs.shape}")
                        print(f'Processed outputs: \n {outputs}')

                    # Collect all predictions and labels for this threshold
                    current_labels.extend(batch_labels.cpu().numpy())
                    current_predictions.extend(outputs.cpu().numpy())

                # Calculate metrics for the current threshold
                auc = roc_auc_score(current_labels, current_predictions, average='macro')
                threshold_scores.append((threshold, auc))

                # Update best threshold after the 5th threshold
                if evaluation_count >= 5 and auc > best_auc:
                    best_auc = auc
                    best_threshold = threshold

                # Collect all labels and predictions for plotting
                all_labels.extend(current_labels)
                all_predictions.extend(current_predictions)

        # Final evaluation using best threshold
        scores = EventScoring(all_labels, all_predictions, fs=self.args.eval_sample_rate)
        ref = Annotation(all_labels, fs=self.args.eval_sample_rate)
        hyp = Annotation(all_predictions, fs=self.args.eval_sample_rate)
        self.plots.plotEventScoring(ref, hyp)
        self.plots.plotIndividualEvents(ref, hyp)
        print("Any-overlap Performance Metrics:")
        print(f"Sensitivity: {scores.sensitivity:.4f}" if not np.isnan(scores.sensitivity) else "Sensitivity: NaN")
        print(f"Precision: {scores.precision:.4f}" if not np.isnan(scores.precision) else "Precision: NaN")
        print(f"F1 Score: {scores.f1:.4f}" if not np.isnan(scores.f1) else "F1 Score: NaN")
        print(f"False Positive Rate (FP/day): {scores.fpRate:.4f}")

        # Calculate overall metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')

        print("Segment based evaluation:")
        print(f"AUC (Best Threshold = {best_threshold}): {best_auc}")
        print(f"Precision: {overall_precision}")
        print(f"Sensitivity (Recall): {overall_recall}")
        print(f"F1 Score: {overall_f1}")
        print(f"Accuracy: {accuracy}")

        return best_threshold
