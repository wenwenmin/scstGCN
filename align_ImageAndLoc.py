import numpy as np
from PIL import Image
from skimage.transform import rescale
import pandas as pd

def main(prefix):
    with open(prefix + 'pixel-size-raw.txt', 'r') as file:
        raw_pix_size = float([line.rstrip() for line in file][0])
    scale_pix_size = 0.5
    scale = raw_pix_size / scale_pix_size

    loc = pd.read_csv(prefix+'locs-raw.csv', header=0, index_col=0)
    loc = loc * scale
    loc = loc.round().astype(int)
    loc.to_csv(prefix+'locs.csv')

    img = np.array(Image.open(prefix+'he-raw.jpg'))
    img = rescale(img, [scale, scale, 1], preserve_range=True)
    img = img.astype(np.uint8)
    H, W, _ = img.shape
    img = img[:H // 224 * 224, :W // 224 * 224]
    Image.fromarray(img).save(prefix+'he.jpg')