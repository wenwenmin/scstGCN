import numpy as np
import pickle
import os

def normalize_data(data):

    min_vals = np.min(data, axis=(1, 2), keepdims=True)
    max_vals = np.max(data, axis=(1, 2), keepdims=True)

    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

predicted_data = []

prefix = ''
folder_path = f"{prefix}\\cnts-super\\"
file_list = os.listdir(folder_path)
for file_name in file_list:
    if file_name.endswith(".pickle"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'rb') as f:
            matrix = pickle.load(f)
            predicted_data.append(matrix)
predicted_data = np.array(predicted_data)

with open(f"{prefix}\\genes_3D.pkl", 'rb') as f:
    gene = pickle.load(f)
    actual_data = np.transpose(gene, (2, 0, 1))

# mask - false
mask = np.array(Image.open(f"{prefix}\mask.png")) > 0
mask = mask[:, :, 0]  # If mask_image is single channel, please delete this code.

predicted_data = normalize_data(predicted_data)
actual_data = normalize_data(actual_data)
predicted_data = np.where(mask, predicted_data, np.nan)
actual_data = np.where(mask, actual_data, np.nan)

squared_errors = [(predicted - actual) ** 2 for predicted, actual in zip(predicted_data, actual_data)]

rmse_per_sample = [np.sqrt(np.nanmean(errors)) for errors in squared_errors]

folder = f"{prefix}\\RMSE\\"
np.savetxt(f'{folder}scstGCN+HIPT.txt', rmse_per_sample)