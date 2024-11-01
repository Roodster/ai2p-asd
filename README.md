# ai2p-asd
Repository for AI2P




# Repository overview

```
data: stores config data.
logs: stores training/evaluation output files, such as hyperparameters, models, plots etc.
notebooks: stores Jupyter notebooks
```



# Installation instructions
```
1. git clone <url-to-repo> 
2. cd ai2p-asd
3. python -m venv .venv
4. For mac `source venv/bin/activate`
5. pip install -r requirements.txt
```


# Branch instructions

When opening a new branch:
```
git branch -b <name-of-branch>
```

To check status:
```
git status
```

When adding files to commit:
```
git add .
```

When commiting to a branch:
```
git commit -m "<commit-message>"
```

When pushing (the first time u have to add '--set-upstream origin <name-of-branch>').
```
git push 
```


# Dataset instruction

```
1. Download dataset from https://physionet.org/content/chbmit/1.0.0/ 
2. Extract to `./data/dataset/`.
```
To create a testing dataset, run the command 
```
python preprocess.py --test_set True --dataset_path "./data/dataset/train/raw/chb01" --save_root "./data/dataset/testset_chb01"
```
To create a training dataset, run the command 
```
python preprocess.py --test_set False --dataset_path "./data/dataset/train/raw" --save_root "./data/dataset/trainSet"
```


# Jupyter notebook instructions

For now, we do not allow .ipynb file extensions. Since we have different machines and most likely different installs,
having jupyter notebooks become a hassle. Instead, we use jupytext.

To convert a .py file into jupyter notebook
```
jupytext --to ipynb <path-to-file>
```

To convert a .ipynbb file into python:
```
jupytext --to py <path-to-notebook>
```
