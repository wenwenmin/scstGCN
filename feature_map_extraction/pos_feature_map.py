import numpy as np
import pickle
from image import upscale, smoothen

def normalize_data(data,k=1):
    """
    对样本数据进行 Min-Max 标准化，将数据标准化到 [0, 1] 范围内。

    参数：
    data: ndarray，包含样本数据的数组，形式为 (n, h, w)，其中 n 是样本数，h 是矩阵高度，w 是矩阵宽度。

    返回值：
    normalized_data: ndarray，标准化后的样本数据数组，形式与 data 相同。
    """
    # 计算每个样本的最小值和最大值
    min_vals = np.min(data, axis=(0, 1), keepdims=True)
    max_vals = np.max(data, axis=(0, 1), keepdims=True)

    # 将样本数据标准化到 [0, 1] 范围内
    normalized_data = (data - min_vals) / (max_vals - min_vals) * k
    return normalized_data

def smoothen_embeddings(
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
    # 创建一个形状为 (2, h, w) 的零矩阵
    coordinates = np.zeros((2, h, w), dtype=int)

    # 填充行索引
    for i in range(h):
        coordinates[0, i, :] = i

    # 填充列索引
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

    pos_emb = smoothen_embeddings(
        pos_emb, size=4, kernel='uniform',
        method='cv',
        device='cuda')

    return pos_emb