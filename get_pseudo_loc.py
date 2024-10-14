import numpy as np
import cv2
import pandas as pd
from PIL import Image

# 10X Visium
prefix = 'E:\human_HD'
diameter = 55
distance = 100
pixel_size = 0.5
he = cv2.imread(f"{prefix}\\he.jpg")
list = []
for i in range(0, int(he.shape[1] * pixel_size - 27.5), distance):
    if i/100 % 2 == 0:
        k = 0
    else:
        k = 50
    for j in range(0, int(he.shape[0] * pixel_size - 27.5 - (127.5-k)), distance):
        x = i + 27.5
        y = j + 27.5 + k
        list.append([x, y])
list = np.array(list).astype(np.float32)
list = list // pixel_size

df = pd.DataFrame(list, columns=['x', 'y'])

mask = np.array(Image.open(f"{prefix}\mask.png")) > 0
mask = mask[:, :, 0]


rows_to_delete = []
H, W = mask.shape
for i in range(H):
    for j in range(W):
        if mask[i, j] == False:
            row_to_delete = df[(df['x'] // 16 - 3 == j) & (df['y'] // 16 - 3 == i)]
            if not row_to_delete.empty:
                rows_to_delete.append(row_to_delete.index[0])

df = df.drop(rows_to_delete)
df.to_csv(f"{prefix}\\locs.csv", index=False, float_format='%.2f')