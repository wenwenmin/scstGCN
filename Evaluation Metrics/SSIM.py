from skimage.metrics import structural_similarity as ssim
import numpy as np
import pickle
import os
from utils import load_pickle, save_image, read_lines, load_image

def normalize_data(data):

    min_vals = np.min(data, axis=(1, 2), keepdims=True)
    max_vals = np.max(data, axis=(1, 2), keepdims=True)

    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

predicted_data = []

prefix= ''
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


predicted_data = normalize_data(predicted_data)
actual_data = normalize_data(actual_data)

valid_mask = ~np.isnan(actual_data[0])

SSIM = []
for i in range(predicted_data.shape[0]):
    gray_image1 = np.where(valid_mask, predicted_data[i], 0)
    gray_image2 = np.where(valid_mask, actual_data[i], 0)
    ssim_index = ssim(gray_image1, gray_image2, data_range=1.0)
    SSIM.append(ssim_index)

x = SSIM
print('Mean:',np.nanmean(x))
print('Var:',np.nanvar(x))

folder = f"{prefix}\\SSIM\\"
np.savetxt(f'{folder}scstGCN+HIPT.txt', x)