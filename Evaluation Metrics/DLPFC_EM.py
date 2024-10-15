import numpy as np
import pickle
import os
from scipy.stats import pearsonr
import pandas as pd

def normalize_data(data):

    min_vals = np.min(data, axis=0, keepdims=True)
    max_vals = np.max(data, axis=0, keepdims=True)

    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

def get_R(x1, x2, dim=1, func=pearsonr):

    r1, p1 = [], []
    for g in range(x1.shape[dim]):
        if dim == 1:
            r, pv = func(x1[:, g], x2[:, g])
        elif dim == 0:
            r, pv = func(x1[g, :], x2[g, :])
        r1.append(r)
        p1.append(pv)
    r1 = np.array(r1)
    p1 = np.array(p1)
    return r1, p1

def get_RSME(x1, x2):
    rsme = []
    for i in range(x1.shape[1]):
        errors = np.sqrt(np.mean((x1[:, i] - x2[:, i]) ** 2))
        rsme.append(errors)
    return np.array(rsme)

def get_MAE(x1, x2):
    mae = []
    for i in range(x1.shape[1]):
        errors = np.mean(np.abs(x1[:, i] - x2[:, i]))
        mae.append(errors)
    return np.array(mae)

source_path = "Please replace it with your address"
genenames = f'{source_path}gene-names.txt'
true_path = f'{source_path}cnts.csv'

pre_path = f'{source_path}\\cnts-pseudo-ours.csv'
folder = f'{source_path}metrics\\ours'

words = []
with open(genenames, 'r', encoding='utf-8') as f:
    for line in f:

        word = line.strip()

        words.append(word)

true_data = pd.read_csv(true_path, index_col=0, header=0)
true_data = true_data[words]
pre_data = pd.read_csv(pre_path, index_col=0, header=0)

pcc, _ = get_R(normalize_data(pre_data.values), normalize_data(true_data.values))
rmse = get_RSME(normalize_data(pre_data.values), normalize_data(true_data.values))
mae = get_MAE(normalize_data(pre_data.values), normalize_data(true_data.values))

np.savetxt(f'{folder}PCC.txt', pcc)
print(pcc.mean())
np.savetxt(f'{folder}RMSE.txt', rmse)
print(rmse.mean())
np.savetxt(f'{folder}MAE.txt', mae)
print(mae.mean())

