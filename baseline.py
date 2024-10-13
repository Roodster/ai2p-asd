"""
SVM Baseline code
"""

import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

from torch.utils.data.dataloader import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from asd.dataset import OfflineSegmentsDataset, OfflineFeaturesDataset, OnlineSegmentsDataset, DummyDataset
from asd.args import Args
from asd.common.utils import set_seed

import warnings
warnings.filterwarnings('ignore')

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

# Performance evaluation metrics
def calculate_metrics(y_true, y_pred):
    
    accuracy = accuracy_score(y_true=y_true, 
                              y_pred=y_pred
                              )
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    sensitivity = tp / (tp + fn)  # TP / (TP + FN)
    false_positive_rate = fp / (fp + tn)  # FP / (FP + TN)
    f1 = f1_score(y_true, y_pred)
    return accuracy, sensitivity,precision, f1, auc, false_positive_rate



def process_patient(patient_id, model_path=None):
    print(f"Processing patient {patient_id}")
    train_dataset = OfflineFeaturesDataset(root_dir="./data/dataset/train/features-balanced/", mode='train', patient_id=str(patient_id).zfill(2))
    test_dataset = OfflineFeaturesDataset(root_dir="./data/dataset/test/features-all/", mode='test', patient_id=str(patient_id).zfill(2))

    train_data = train_dataset[:]
    test_data = test_dataset[:]

    X_train = np.nan_to_num(np.vstack([segment.numpy() for segment, _ in train_data]))
    y_train = np.vstack([label.item() for _, label in train_data]).flatten()
    X_test = np.nan_to_num(np.vstack([segment.numpy() for segment, _ in test_data]))
    y_test = np.vstack([label.item() for _, label in test_data]).flatten()

    if model_path is not None:
            # Load the pre-trained model from the given path
            with open(model_path, 'rb') as f:
                svm = pickle.load(f)
            print(f"Loaded model from {model_path}")
    else:        

        svm = SVC(kernel='rbf', degree=3, C=10, gamma='scale')
        svm.fit(X_train, y_train)

        with open(f'svm_patientid_{patient_id}.pkl', 'wb') as f:
            pickle.dump(svm, f)

    y_pred = svm.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)


    print(f"===== PATIENT {patient_id} =====")
    print(f"Accuracy: {metrics[0]}")
    print(f"Sensitivity: {metrics[1]}")
    print(f"Precision: {metrics[2]}")
    print(f"F1 Score: {metrics[3]}")
    print(f"AUC: {metrics[4]}")
    print(f"FPR: {metrics[5]}")

    return patient_id, metrics


if __name__ == "__main__":
    args = Args(file="./data/configs/default.yaml")
    set_seed(42)

    process_patient(16, model_path="D:\\tudelft\\ai2p-asd\logs\svm_results\svm_patientid_16.pkl")
    process_patient(18, model_path="D:\\tudelft\\ai2p-asd\logs\svm_results\svm_patientid_18.pkl")

    # patient_ids = list(range(9,10))
    # metrics = [None] * len(patient_ids)



    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     future_to_patient = {executor.submit(process_patient, patient_id, args): patient_id for patient_id in patient_ids}
        
    #     for future in as_completed(future_to_patient):
    #         patient_id, patient_metrics = future.result()
    #         metrics[patient_id - 1] = patient_metrics

    # metrics = np.array(metrics)
    # np.savetxt("svm_statistics_per_patient.csv", metrics, delimiter=",")
    

    
    # # Assuming X is a 2D array with shape (num_samples, num_features)
    # # and y is a 1D array with labels (0 for non-seizure, 1 for seizure)
    
    # # Load arguments
    # args = Args(file="./data/configs/default.yaml")
    # from asd.common.utils import set_seed
    
    # set_seed(42)
    
    # metrics = []
    
    
    # patient_ids = list(range(1, 25))
    # for patient_id in patient_ids:
    
    #     train_dataset = OfflineFeaturesDataset(root_dir="./data/dataset/train/features/", mode='train', patient_id=str(patient_id).zfill(2))
    #     test_dataset = OfflineFeaturesDataset(root_dir="./data/dataset/test/features-all/", mode='test', patient_id=str(patient_id).zfill(2))
    
        
    #     # Instantiate dataloaders 
    #     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    #     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)


    #     train_data = train_dataset[:]
        
    #     test_data = test_dataset[:]
        
    #     X_train = np.nan_to_num(np.vstack([segment.numpy() for segment, _ in train_data]))
            
    #     y_train = np.vstack([label.item() for _, label in train_data]).flatten()
    #     X_test = np.nan_to_num(np.vstack([segment.numpy() for segment, _ in test_data]))
    #     y_test = np.vstack([label.item() for _, label in test_data]).flatten()

    #     # Compute class weights to address class imbalance
    #     # Define the SVM model with RBF kernel
    #     svm = SVC(kernel='rbf', 
    #               degree=3,
    #               C=10, 
    #               gamma='scale',
    #               )
    #     svm.fit(X_train, y_train)
    #     # # Hyperparameter tuning using cross-validation
    #     # param_grid = {'C': [100], 'gamma': ['scale']}
    #     # grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='precision', verbose=4)
    #     # grid_search.fit(X_train, y_train)


    #     # # Best model from cross-validation
    #     # best_svm = grid_search.best_estimator_

    #     # Test set predictions
    #     y_pred = svm.predict(X_test)
        
    #     # y_final_pred = postprocess_predictions(y_pred)

    #     # Assuming y_test is already aligned for postprocessed windows
    #     metrics.append(calculate_metrics(y_test, y_pred))


    #     print(f"===== PATIENT {patient_id} =====")
    #     print(f"Accuracy: {metrics[patient_id-1][0]}")
    #     print(f"SEnsitivity: {metrics[patient_id-1][1]}")
    #     print(f"Precision: {metrics[patient_id-1][2]}")
    #     print(f"F1 Score: {metrics[patient_id-1][3]}")
    #     print(f"AUC: {metrics[patient_id-1][4]}")
    #     print(f"FPR: {metrics[patient_id-1][5]}")

    #     ##########################
    #     # SAVE-LOAD using pickle #
    #     ##########################
    #     import pickle

    #     # save
    #     with open(f'svm_patientid_{patient_id}.pkl','wb') as f:
    #         pickle.dump(svm,f)


    # metrics = np.array(metrics)
    
    # np.savetxt("svm_statistics_per_patient.csv", metrics, delimiter=",")