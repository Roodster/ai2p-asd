"""
SVM Baseline code
"""

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

# Postprocessing: Group predictions in windows of 10, and apply the 8/10 rule for seizure detection
def postprocess_predictions(y_pred, window_size=10, threshold=8):
    final_predictions = []
    for i in range(0, len(y_pred) - window_size + 1, window_size):
        window = y_pred[i:i + window_size]
        seizure_votes = np.sum(window)
        if seizure_votes >= threshold:
            final_predictions.append(1)  # Seizure detected
        else:
            final_predictions.append(0)  # Non-seizure
    return np.array(final_predictions)

y_final_pred = postprocess_predictions(y_pred)

# Performance evaluation metrics
def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn)  # TP / (TP + FN)
    false_positive_rate = fp / (fp + tn)  # FP / (FP + TN)
    precision = tp / (tp + fp)  # TP / (TP + FP)
    f1 = f1_score(y_true, y_pred)
    detection_delay = None  # Not defined in detail, placeholder
    return sensitivity, false_positive_rate, precision, f1, detection_delay


if __name__ == "__main__":
    
    # Assuming X is a 2D array with shape (num_samples, num_features)
    # and y is a 1D array with labels (0 for non-seizure, 1 for seizure)
    # Load your data here
    # X = ...
    # y = ...

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Compute class weights to address class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # Define the SVM model with RBF kernel
    svm = SVC(kernel='rbf', class_weight=class_weight_dict)

    # Hyperparameter tuning using cross-validation
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Best model from cross-validation
    best_svm = grid_search.best_estimator_

    # Test set predictions
    y_pred = best_svm.predict(X_test)

    # Assuming y_test is already aligned for postprocessed windows
    metrics = calculate_metrics(y_test[:len(y_final_pred)], y_final_pred)

    print(f"Sensitivity: {metrics[0]}")
    print(f"False Positive Rate: {metrics[1]}")
    print(f"Precision: {metrics[2]}")
    print(f"F1 Score: {metrics[3]}")
    print(f"Detection Delay: {metrics[4]} (To be defined)")