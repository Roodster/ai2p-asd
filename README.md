# AI2P: Automatic Seizure Detection (ai2p-asd)

This repository contains the codebase for the Advanced Interdisciplinary Artificial Intelligence Project (AI2P) at TU Delft. 

The primary goal of this project is to develop an automatic seizure detection model feasible for deployment on a wearable, behind-the-ear EEG device. To simulate this low-channel setup, the models utilize only 4 specific channels (F7-T7, T7-P7, F8-T8, T8-P8) extracted from standard scalp EEG data. The repository includes implementations of deep learning models, focusing on DARLNet, SLAM, and an Ensemble approach.

---

## Repository Overview

* **`data/`**: Stores configuration data and local datasets.
* **`logs/`**: Stores training and evaluation output files, such as hyperparameters, saved model weights, and visualization plots.
* **`notebooks/`**: Stores Jupyter notebooks used for training, validation, and testing workflows.

---

## Installation Instructions

To set up the project locally, ensure you have Python installed, and follow these steps:

1. Clone the repository:
   ```
   git clone [https://github.com/Roodster/ai2p-asd.git](https://github.com/Roodster/ai2p-asd.git)
   cd ai2p-asd
   ```
2. Create a virtual environment:
   ```
   python -m venv .venv
   ```
3. Activate the virtual environment:
   * **Mac/Linux**: `source .venv/bin/activate`
   * **Windows**: `.venv\Scripts\activate`
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

---

## Dataset Instructions

This project utilizes the publicly available CHB-MIT Scalp EEG Database.

1. Download the CHB-MIT dataset from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/).
2. Extract the downloaded files to `./data/dataset/`.

### Preprocessing

The preprocessing pipeline applies a 0.5-40Hz Butterworth bandpass filter and Z-score standardization. To prepare the data for the models, use the provided preprocessing script. 

To create a **testing or validation dataset**, run:
```
python preprocess.py --test_set True --dataset_path "./data/dataset/train/raw/chb01" --save_root "./data/dataset/testset_chb01"
```

To create a **training dataset**, run:
```
python preprocess.py --test_set False --dataset_path "./data/dataset/train/raw" --save_root "./data/dataset/trainSet"
```

---

## Training and Evaluation

Model training uses a modified leave-one-patient-out cross-validation strategy, separating one patient for testing and the immediate next patient for validation. Performance is evaluated using both segment-based metrics and clinically relevant event-based scoring (any-overlap method). 

**Jupyter Notebook Instructions:**
1. Navigate to the `./notebooks/train.ipynb` file.
2. For accelerated training, it is recommended to upload this notebook along with your preprocessed train, test, and validation datasets to Kaggle (the original experiments utilized Kaggle's T4x2 GPUs).
3. Run the notebook cells sequentially to execute training and testing pipelines.

---

## Project Results

Among the evaluated models, the best model achieved a mean event-based sensitivity of 0.588 and a daily false positive rate of 4.837. A custom post-processing step (sliding window and event merging) is applied to all model outputs to reduce fragmented false alarms and generate concise event detections.
