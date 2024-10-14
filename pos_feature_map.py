import numpy as np
from utils import smoothen

def normalize_data(data, k=1):

    min_vals = np.min(data, axis=(0, 1), keepdims=True)
    max_vals = np.max(data, axis=(0, 1), keepdims=True)

    normalized_data = (data - min_vals) / (max_vals - min_vals) * k
    return normalized_data

def embeddings(
        embs, size, kernel,
        method='cv', device='cuda'):
    smoothened = [
        smoothen(
            c[..., np.newaxis], size=size,
            kernel=kernel, backend=method,
            device=device)[..., 0]
        for c in embs]
    return smoothened

def create_coordinates_matrix(h, w):

    coordinates = np.zeros((2, h, w), dtype=int)

    for i in range(h):
        coordinates[0, i, :] = i

    for j in range(w):
        coordinates[1, :, j] = j

    return coordinates

def main(prefix, img_emb):
    h, w = img_emb[0].shape[0], img_emb[0].shape[1]
    coordinates_matrix = create_coordinates_matrix(h, w)
    pos_emb = [[], []]
    pos_emb[0] = coordinates_matrix[0].astype('float32')
    pos_emb[0] = normalize_data(pos_emb[0])
    pos_emb[1] = coordinates_matrix[1].astype('float32')
    pos_emb[1] = normalize_data(pos_emb[1])

    pos_emb = embeddings(
        pos_emb, size=4, kernel='uniform',
        method='cv',
        device='cuda')

    return pos_emb