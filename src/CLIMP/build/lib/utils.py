import torch
import numpy as np
import pandas as pd
from .model import SelfAttention
from .model import ConstrastiveLoss
from torch import optim
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

def load_data(data_path):
    data_csv = pd.read_csv(data_path, index_col=0)
    cells = data_csv.columns.values
    genes = data_csv.index.values
    data = data_csv.values.T

    return data, cells, genes

def impute_dropout(X, seed=None, drop_rate=0.1):
    """
    X: original testing set
    ========
    returns:
    X_zero: copy of X with zeros
    i, j, ix: indices of where dropout is applied
    """

    X_zero = np.copy(X)
    i, j = np.nonzero(X_zero)
    if seed is not None:
        np.random.seed(seed)

    ix = np.random.choice(range(len(i)), int(
        np.floor(drop_rate * len(i))), replace=False)
    X_zero[i[ix], j[ix]] = 0.0

    return X_zero, i, j, ix

def load_pretained_model(X, load_path):
    print("loading pre-train model")
    model = SelfAttention(input_size=X.shape[-1], hidden_size=128)
    model.load_state_dict(torch.load(load_path))
    return model.to(X.device)

def training(X, hidden_size=128, epoch=100, aug_rate=0.4):

    model = SelfAttention(input_size=X.shape[-1], hidden_size=hidden_size).to(X.device)
    criterion_instance = ConstrastiveLoss(X.shape[0], 1.5)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    loss_min = 100000

    for i in range(epoch):
        model.train()
        optimizer.zero_grad()

        y1 = model(data_augmentations(X, rate=aug_rate))
        y2 = model(data_augmentations(X, rate=aug_rate))

        loss = criterion_instance(y1, y2)
        if (i + 1) % 100 == 0 or i == 0:
            print(loss.item())
        loss.backward()
        optimizer.step()

    return model


def data_augmentations(X, rate=0.4):

    X_aug = X.clone().to(X.device)

    select_index = X_aug.nonzero()
    ix = np.random.choice(range(len(select_index)), int(
        np.floor(rate * len(select_index))), replace=False)
    X_aug[select_index.T[0][ix], select_index.T[1][ix]] = 0.0

    return X_aug

def select_neighbours(model, X, k):
    model.eval()
    with torch.no_grad():
        hidden = model(X)
        sim = torch.cosine_similarity(hidden.unsqueeze(1), hidden.unsqueeze(0), dim=-1)
        sim = sim.fill_diagonal_(0.0)
        choose_cell = sim.argsort()[:, -k:].to('cpu').numpy()
    return choose_cell

def LS_imputation(drop_data, choose_cell, device, filter_noise=5):

    original_data = torch.FloatTensor(np.copy(drop_data)).to(device)
    dataImp = original_data.clone().to(device)
    for i in range(dataImp.shape[0]):
        y = original_data[i, dataImp[i].nonzero()]  # [datas, 1]
        x = original_data[choose_cell[i], dataImp[i].nonzero()]  # [datas, 20]
        w = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(x.T, x)), x.T), y)  # 20,1
        impute_data = torch.matmul(original_data[choose_cell[i], (dataImp[i] == 0).nonzero()], w).squeeze()
        # filter noise
        impute_data[impute_data <= filter_noise] = 0
        dataImp[i, dataImp[i] == 0] = impute_data

    return dataImp.detach().cpu().numpy()

def cos_simility(imputed_data, original_data):

    return np.mean(np.cosine_similarity(original_data, imputed_data))

def l1_distance(imputed_data, original_data):

    return np.mean(np.abs(original_data-imputed_data))

def RMSE(imputed_data, original_data):
    return np.sqrt(np.mean((original_data - imputed_data)**2))

 # 计算两个向量person相关系数
def pearson_corr(imputed_data, original_data):
    Y = original_data
    fake_Y = imputed_data
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = np.mean(fake_Y), np.mean(Y)
    corr = (np.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
            np.sqrt(np.sum((fake_Y - fake_Y_mean) ** 2)) * np.sqrt(np.sum((Y - Y_mean) ** 2)))
    return corr

def getARI(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)

def getNMI(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred)

def getPurityScore(y_true, y_pred):
    contingency = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

# Pos scores
def getPosScore(traj):
    scoreorder = 0
    for i in range(len(traj)):
        for j in range(i, len(traj)):
            scoreorder += np.sum((traj[j:] - traj[j]))

    optscoreorder=0
    sort_traj= np.sort(traj)
    for i in range(len(sort_traj)):
        for j in range(i, len(sort_traj)):
            optscoreorder += np.sum((sort_traj[j:] - sort_traj[j]))
    return scoreorder/optscoreorder