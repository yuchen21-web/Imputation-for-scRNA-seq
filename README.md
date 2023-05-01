# CL-Impute

An imputation method for scRNA-seq data based on contrastive learning

# Requirements

- python = 3.7
- sklearn
- numpy
- pandas
- torch


# Tutorial

There are three ways to perform CL-Impute. Moreover, we provided a saved trained model and Zeisel dataset to verify the effectiveness of the paper.

## 1.Use CL-Impute as a python function

Package CL-Impute as a python function with setup.py for use in other code

1.1 Package CL-Impute utils in shells
```shell
src/CLIMP$ python3 setup.py bdist
src/CLIMP$ sudo python3 setup.py install --record installed.txt
```
1.2 Use CL-Impute utils function in python
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
# pd.DataFrame(imputed_data.T, index=genes, columns=cells).to_csv('saved path')
```

## 2. Follow the procedure below to perform CL-Impute with the jupyter or on tutorial_CLImpte.ipynb

### 2.1 Verify th experimental results of Zeisel dataset in this paper

```python
import torch
from CLIMP import CLImputeUtils
import numpy as np
device=torch.device('cpu')
dataset_name=Zeisel
```
```python
# Verify experimental results of Zeisel dataset in this paper

## 1. loading dataset and the simulated dropout events data with 40% used in our experiment
groundTruth_data, cells, genes = utils.load_data('CLIMP/data/Zeisel/Zeisel_top2000.csv')
drop_data, _, _ = utils.load_data('CLIMP/data/Zeisel/Zeisel_d40.csv')
X = torch.FloatTensor(np.copy(drop_data)).to(device)
drop_rate = (len(groundTruth_data.nonzero()[0])-len(drop_data.nonzero()[0]))/len(groundTruth_data.nonzero()[0])
print('drop rate: {:.2f}'.format(drop_rate))

## 2. loading the saved model
model = utils.load_pretained_model(X, load_path='CLIMP/data/Zeisel/Zeisel_saved_model.pkl')

## 3.imputation
choose_cell = utils.select_neighbours(model, X, k=20)
imputed_data = utils.LS_imputation(drop_data, choose_cell, device, filter_noise=2)

# saved
# imputed_saved = pd.DataFrame(imputed_data.T, index=genes, columns=cells)
# imputed_saved.to_csv('CLIMP/data/Zeisel/Zeisel_Imputed.csv')

print('dropout data PCCs: {:.4f}, imputed data PCCs: {:.4f}'.
      format(utils.pearson_corr(drop_data, groundTruth_data), 
             utils.pearson_corr(imputed_data, groundTruth_data)))
print('dropout data L1: {:.4f}, imputed data L1: {:.4f}'.
      format(utils.l1_distance(drop_data, groundTruth_data), 
             utils.l1_distance(imputed_data, groundTruth_data)))
print('dropout data RMSE: {:.4f}, imputed data RMSE: {:.4f}'.
      format(utils.RMSE(drop_data, groundTruth_data), 
             utils.RMSE(imputed_data, groundTruth_data)))
```
drop rate: 0.40
loading pre-train model
dropout data PCCs: 0.7688, imputed data PCCs: 0.9483
dropout data L1: 1.8372, imputed data L1: 1.0810
dropout data RMSE: 17.2358, imputed data RMSE: 8.5091

```python
# Verify clustering results of Zeisel dataset
clusterResults = pd.read_csv('CLIMP/data/Zeisel/Zeisel_d40_Clustering.csv', index_col=0)
clusterResults = clusterResults.values.squeeze()
labels = pd.read_csv('CLIMP/data/Zeisel/Zeisel_cell_label.csv', index_col=0)
labels = labels.values.squeeze()
print('ARI: {:.3f}, NMI: {:.3f}, NMI: {:.3f}'.
      format(utils.adjusted_rand_score(clusterResults, labels), 
             utils.normalized_mutual_info_score(clusterResults, labels),
             utils.getPurityScore(clusterResults, labels)))
```
ARI: 0.879, NMI: 0.841, NMI: 0.938

### 2.2 Perform CL-Impute with the jupyter notebook


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

# saved file
# pd.DataFrame(imputed_data.T, index=genes, columns=cells).to_csv('saved path')
```
```python
## Step5: evaluation
print('dropout data PCCs:', CLImputeUtils.pearson_corr(drop_data, groundTruth_data))
print('imputed data PCCs:', CLImputeUtils.pearson_corr(imputed_data, groundTruth_data))
print('dropout data L1:', CLImputeUtils.l1_distance(drop_data, groundTruth_data))
print('imputed data L1:', CLImputeUtils.l1_distance(imputed_data, groundTruth_data))
```

## 3. Run Impute.py programs directly

change directory to src/CLIMP and run Impute.py
