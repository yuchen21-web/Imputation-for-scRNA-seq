# CL-Impute

CL-Impute is an imputation method for scRNA-seq data based on contrastive learning.

Overall architecture of the CL–Impute algorithm pipeline:

![Model](https://github.com/yuchen21-web/Imputation-for-scRNA-seq/blob/main/src/Model.eps)

CL–Impute generates two augmented cells for each original cell by randomly masking non–zero values to simulate dropout events. The self–attention network is then designed to automatically capture the potential cell relationships and learn the latent cell representations. Finally, the two augmented cells are then considered positive pairs for contrastive learning. When the model training is completed, the learned cell representations are utilized to select k–nearest cells for the imputation task using the least square method.

# Requirements

- python = 3.7
- sklearn
- numpy
- pandas
- torch

# Tutorial

This is a running guide for CLImputeto perform CL-Impute. Moreover, we provided a saved trained model and Zeisel dataset to verify the effectiveness of the paper.

## 1. Follow the procedure below to perform CL-Impute with the jupyter or on [tutorial_CLImpte.ipynb](https://github.com/yuchen21-web/Imputation-for-scRNA-seq/blob/main/src/tutorial_CLImpte.ipynb)

### 1.1 Perform CL-Impute with the jupyter notebook

```python
import torch
from CLIMP import CLImputeUtils
import numpy as np
import pandas as pd
device=torch.device('cpu')
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

### 1.2 Verify the experimental results of Zeisel dataset in this paper

```python
# Verify experimental results of Zeisel dataset in this paper

## 1. loading dataset and the simulated dropout events data with 40% used in our experiment
groundTruth_data, cells, genes = CLImputeUtils.load_data('CLIMP/data/Zeisel/Zeisel_top2000.csv')
drop_data, _, _ = CLImputeUtils.load_data('CLIMP/data/Zeisel/Zeisel_d40.csv')
X = torch.FloatTensor(np.copy(drop_data)).to(device)
drop_rate = (len(groundTruth_data.nonzero()[0])-len(drop_data.nonzero()[0]))/len(groundTruth_data.nonzero()[0])
print('drop rate: {:.2f}'.format(drop_rate))

## 2. loading the saved model
model = CLImputeUtils.load_pretained_model(X, load_path='CLIMP/data/Zeisel/Zeisel_saved_model.pkl')

## 3.imputation
choose_cell = CLImputeUtils.select_neighbours(model, X, k=20)
imputed_data = CLImputeUtils.LS_imputation(drop_data, choose_cell, device, filter_noise=2)

# saved
# imputed_saved = pd.DataFrame(imputed_data.T, index=genes, columns=cells)
# imputed_saved.to_csv('CLIMP/data/Zeisel/Zeisel_Imputed.csv')

print('dropout data PCCs: {:.4f}, imputed data PCCs: {:.4f}'.
      format(CLImputeUtils.pearson_corr(drop_data, groundTruth_data), 
             CLImputeUtils.pearson_corr(imputed_data, groundTruth_data)))
print('dropout data L1: {:.4f}, imputed data L1: {:.4f}'.
      format(CLImputeUtils.l1_distance(drop_data, groundTruth_data), 
             CLImputeUtils.l1_distance(imputed_data, groundTruth_data)))
print('dropout data RMSE: {:.4f}, imputed data RMSE: {:.4f}'.
      format(CLImputeUtils.RMSE(drop_data, groundTruth_data), 
             CLImputeUtils.RMSE(imputed_data, groundTruth_data)))
'''
drop rate: 0.40
loading pre-train model
dropout data PCCs: 0.7688, imputed data PCCs: 0.9483
dropout data L1: 1.8372, imputed data L1: 1.0810
dropout data RMSE: 17.2358, imputed data RMSE: 8.5091
'''
```

```python
# Verify clustering results of Zeisel dataset
clusterResults = pd.read_csv('CLIMP/data/Zeisel/Zeisel_d40_Clustering.csv', index_col=0)
clusterResults = clusterResults.values.squeeze()
labels = pd.read_csv('CLIMP/data/Zeisel/Zeisel_cell_label.csv', index_col=0)
labels = labels.values.squeeze()
print('ARI: {:.3f}, NMI: {:.3f}, NMI: {:.3f}'.
      format(CLImputeUtils.adjusted_rand_score(clusterResults, labels), 
             CLImputeUtils.normalized_mutual_info_score(clusterResults, labels),
             CLImputeUtils.getPurityScore(clusterResults, labels)))
'''
ARI: 0.879, NMI: 0.841, NMI: 0.938
'''
```

## 2. Run Impute.py program directly

change directory to src/CLIMP and run Impute.py

# Quick start

Package CL-Impute as a python function with setup.py for imputation in other python program

### 1 Package CL-Impute utils in shells

```shell
src/CLIMP$ python3 setup.py bdist
src/CLIMP$ sudo python3 setup.py install --record installed.txt
```

### 2 Use CL-Impute utils function in python

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
model = CLImputeUtils.training(torch.FloatTensor(drop_data).to(device), hidden_size=64, epoch=100, aug_rate=0.4)

## imputation
choose_cell = CLImputeUtils.select_neighbours(model, X, k=20)
imputed_data = CLImputeUtils.LS_imputation(drop_data, choose_cell, device)

# saved file
# pd.DataFrame(imputed_data.T, index=genes, columns=cells).to_csv('saved path')
```


