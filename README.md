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