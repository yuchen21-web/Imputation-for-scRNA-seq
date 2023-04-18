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

1.change directory to src/CLIMP and run Impute.py.

2.You can perform CL-Impute on jupyter notebook or follow the process:


```python
import torch
from CLIMP import CLImputeUtils as utils
import numpy as np
device=torch.device('cpu')
dataset_name=Zeisel

## Step1: reading dataset
groundTruth_data, cells, genes = utils.load_data('data/Zeisel/Zeisel_top2000.csv')

## Step2: simulate dropout-events
drop_data = utils.impute_dropout(groundTruth_data, drop_rate=0.4)

## Step3: training embedding
X = torch.FloatTensor(np.copy(drop_data)).to(device)
# Step3.1: loading the provided trained model
model = utils.load_pretained_model(X, load_path='data/Zeisel/Zeisel_saved_model.pkl')
# or Step3.2: training a model
# model = utils.training(X, hidden_size=128, epoch=100, aug_rate=0.4)

## Step4: select k similiar cells and imputation
choose_cell = utils.select_neighbours(model, X, k=20)
imputed_data = utils.LS_imputation(drop_data, choose_cell, device)

## Step5: evaluation
print('dropout data PCCs:', utils.pearson_corr(drop_data, groundTruth_data))
print('imputed data PCCs:', utils.pearson_corr(imputed_data, groundTruth_data))
print('dropout data L1:', utils.l1_distance(drop_data, groundTruth_data))
print('imputed data L1:', utils.l1_distance(imputed_data, groundTruth_data))

```

3.Package CL-Impute as a python function with setup.py for use in other code

3.1 package CL-Impute utils in shells
```shell
src/CLIMP$ python3 setup.py bdist
src/CLIMP$ sudo python3 setup.py install --record installed.txt
```
3.2 use CL-Impute utils in python
```python
import CLImputeUtils
```
