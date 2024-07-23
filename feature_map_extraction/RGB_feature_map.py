from einops import reduce
import numpy as np
from utils import load_image

def main(prefix):
    img = load_image(f'{prefix}he.jpg')
    rgb_emb = np.stack([
        reduce(
            img[..., i].astype(np.float16) / 255.0,
            '(h1 h) (w1 w) -> h1 w1', 'mean',
            h=16, w=16).astype(np.float32)
        for i in range(3)])
    return rgb_emb