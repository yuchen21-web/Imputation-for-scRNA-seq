# CL-Impute

An imputation method for scRNA-seq data based on contrastive learning

# Requirements

- python = 3.7
- sklearn
- numpy
- pandas
- torch


# Tutorial

We provide three ways to perform CL-Impute. Besides, we provided a saved trained model and Zeisel dataset to verify the effectiveness of the paper.

## 1. Run Impute.py on the project to execute a default imputation

```python
# change directory to src/CLIMP
python Impute.py
```

## 2. Follow the procedure below to perform CL-Impute on jupyter or on tutorial_CLImpte.ipynb

```python
import torch
from CLIMP import CLImputeUtils
import numpy as np
device=torch.device('cpu')
dataset_name=Zeisel
```
```python
## Step1: reading dataset
groundTruth_data, cells, genes = CLImputeUtils.load_data('data/Zeisel/Zeisel_top2000.csv')
```
```python
## Step2: simulate dropout-events
drop_data = CLImputeUtils.impute_dropout(groundTruth_data, drop_rate=0.4)
```
```python
## Step3: training embedding
X = torch.FloatTensor(np.copy(drop_data)).to(device)
# Step3.1: loading the provided trained model
model = CLImputeUtils.load_pretained_model(X, load_path='data/Zeisel/Zeisel_saved_model.pkl')
# or Step3.1: training a model
# model = CLImputeUtils.training(X, hidden_size=128, epoch=100, aug_rate=0.4)
```
```python
## Step4: select k similiar cells and imputation
choose_cell = CLImputeUtils.select_neighbours(model, X, k=20)
imputed_data = CLImputeUtils.LS_imputation(drop_data, choose_cell, device)
```
```python
## Step5: evaluation
print('dropout data PCCs:', CLImputeUtils.pearson_corr(drop_data, groundTruth_data))
print('imputed data PCCs:', CLImputeUtils.pearson_corr(imputed_data, groundTruth_data))
print('dropout data L1:', CLImputeUtils.l1_distance(drop_data, groundTruth_data))
print('imputed data L1:', CLImputeUtils.l1_distance(imputed_data, groundTruth_data))
```

## 3.Use CL-Impute as a python function

Package CL-Impute as a python function with setup.py for use in other code

3.1 Package CL-Impute utils in shells
```shell
src/CLIMP$ python3 setup.py bdist
src/CLIMP$ sudo python3 setup.py install --record installed.txt
```
3.2 Use CL-Impute utils function in python
```python
import CLImputeUtils
import pandas as pd

device=torch.device('cpu') # or you can use cuda

# load data that need to be imputed, shape=[cells, genes]
drop_data, cells, genes = CLImputeUtils.load_data(datapath)

# or you can load a groundTruth data to test imputation performance
# groundTruth_data, cells, genes = CLImputeUtils.load_data(datapath)
# drop_data = CLImputeUtils.impute_dropout(groundTruth_data, drop_rate=0.4)


# contrastive learning the embedding
model = CLImputeUtils.training(torch.FloatTensor(drop_data).to(device), hidden_size=128, epoch=100, aug_rate=0.4)

## imputation
choose_cell = CLImputeUtils.select_neighbours(model, X, k=20)
imputed_data = CLImputeUtils.LS_imputation(drop_data, choose_cell, device)

# saved file
pd.DataFrame(imputed_data, index=cells, columns=genes).to_csv('saved path')
```
