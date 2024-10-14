import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from matplotlib.patches import Rectangle

prefix = 'E:\human_HD'

def normalize_data(data):

    min_vals = np.min(data, axis=(0, 1), keepdims=True)
    max_vals = np.max(data, axis=(0, 1), keepdims=True)


    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

# #  Prediction
# with open(f"{prefix}\cnts-super\\Slc17a7.pickle", 'rb') as f:
#     matrix = pickle.load(f)
#     matrix = normalize_data(matrix)

# #  True
with open(f"{prefix}\genes_3D.pkl", 'rb') as f:
    gene = pickle.load(f)
    matrix = normalize_data(gene[:, :, 880])

mask = np.array(Image.open(f"{prefix}\mask.png")) > 0
mask = mask[:, :, 0]
display_matrix = np.where(mask, matrix, np.nan)

cmap = plt.cm.viridis
cmap.set_bad(color='white')

fig, ax = plt.subplots()

im = ax.imshow(display_matrix, cmap=cmap, vmin=matrix.min(), vmax=matrix.max())

plt.imshow(display_matrix, cmap='viridis', vmin=0., vmax=0.5)
plt.axis('off')

rect = Rectangle((0, 0), display_matrix.shape[1], display_matrix.shape[0],
                 fill=False, edgecolor='black', linewidth=2)
ax.add_patch(rect)

plt.show()



