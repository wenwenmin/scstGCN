import numpy as np
import pickle
import pandas as pd
from utils import get_disk_mask
from PIL import Image

Image.MAX_IMAGE_PIXELS = 300000000

prefix = 'E:\human_HD'

def get_locs(prefix, target_shape=None):

    locs = pd.read_csv(f'{prefix}locs.csv', header=0, index_col=0)


    locs = np.stack([locs['x'], locs['y']], -1)

    if target_shape is not None:
        wsi = np.array(Image.open(f'{prefix}he.jpg'))
        current_shape = np.array(wsi.shape[:2])
        rescale_factor = current_shape // target_shape
        locs = locs.astype(float)
        locs /= rescale_factor

    locs = locs.round().astype(int)

    return locs

def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)
    x_list = []
    for s in locs:
        patch = img[
                s[1]+r[0][0]:s[1]+r[0][1],
                s[0]+r[1][0]:s[0]+r[1][1]]
        if mask.all():
            x = patch
        else:
            x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list

with open(f"{prefix}\genes_3D.pkl", 'rb') as f:
    img = pickle.load(f)

radius = 55 / 16

mask = get_disk_mask(radius)

locs =get_locs(f"{prefix}\\", np.array([img.shape[0], img.shape[1]]))

x = get_patches_flat(img, locs, mask)

cnts = np.sum(x, axis=1)

with open(f"{prefix}\gene_names.txt", 'r') as file:
    gene_names = [line.strip() for line in file]

cnts_df = pd.DataFrame(data=cnts, columns=gene_names)
cnts_df.to_csv(f'{prefix}\\cnts.csv')
