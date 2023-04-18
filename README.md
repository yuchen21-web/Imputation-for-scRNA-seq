# clustering-for-scRNA-seq

a imputation method for scRNA-seq data based on contrastive learning

# Requirements

- python = 3.7
- sklearn
- numpy
- pandas
- torch


# Tutorial

We provide three ways for you to use CL-Impute:

1.You can perform CL-Impute on jupyter notebook or follow the process:


```
import torch
from CLIMP import CLImputeUtils as utils
import numpy as np

## Step1: reading dataset
groundTruth_data, cells, genes = utils.load_data('data/Zeisel/Zeisel_top2000.csv')

## Step2: simulate dropout-events
drop_data, i, j, ix = utils.impute_dropout(groundTruth_data, drop_rate=drop_rate)

## Step3: training embedding
X = torch.FloatTensor(np.copy(drop_data)).to(device)
# Step3.1: loading the provided trained model
model = utils.load_pretained_model(X, load_path='data/Zeisel/Zeisel_saved_model.pkl')
# or Step3.2: training a model
# model = utils.training(X, hidden_size=128, epoch=100, aug_rate=0.4)

## Step4: select k similiar cells and imputation
choose_cell = utils.select_neighbours(model, X, k=20)
imputed_data = utils.LS_imputation(drop_data, choose_cell, device, filter_noise=5)

## Step5: evaluation
print('dropout data PCCs:', utils.pearson_corr(drop_data, groundTruth_data))
print('imputed data PCCs:', utils.pearson_corr(imputed_data, groundTruth_data))
print('dropout data L1:', utils.l1_distance(drop_data, groundTruth_data))
print('imputed data L1:', utils.l1_distance(imputed_data, groundTruth_data))

```





2.change directory to src/CLIMP and run Impute.py

3.Package CL-Impute as a Python function package with setup.py for use in other code

3.1 package CL-Impute utils
```shell
src/CLIMP$ python3 setup.py bdist
src/CLIMP$ sudo python3 setup.py install --record installed.txt
```
3.2 use CL-Impute utils in other code
```python
import CLImputeUtils
```
