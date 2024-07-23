import sys
from utils import load_pickle, save_pickle, sort_labels, load_mask
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from utils import save_image, load_pickle
from connected_components import relabel_small_connected
from image import crop_image

def prepare_for_clustering(embs, location_weight):
    mask = np.all([np.isfinite(c) for c in embs], axis=0)
    embs = np.stack([c[mask] for c in embs], axis=-1)

    if location_weight is None:
        x = embs
    else:
        embs -= embs.mean(0)
        embs /= embs.var(0).sum()**0.5
        # get spatial coordinates
        locs = np.meshgrid(
                *[np.arange(mask.shape[i]) for i in range(mask.ndim)],
                indexing='ij')
        locs = np.stack(locs, -1).astype(float)
        locs = locs[mask]
        locs -= locs.mean(0)
        locs /= locs.var(0).sum()**0.5

        # balance embeddings and coordinates
        embs *= 1 - location_weight
        locs *= location_weight
        x = np.concatenate([embs, locs], axis=-1)
    return x, mask

def cluster(
        embs, n_clusters, method='mbkm', location_weight=None,
        sort=True):

    x, mask = prepare_for_clustering(embs, location_weight)

    print(f'Clustering pixels using {method}...')
    if method == 'mbkm':
        model = MiniBatchKMeans(
                n_clusters=n_clusters,
                # batch_size=x.shape[0]//10, max_iter=1000,
                # max_no_improvement=100, n_init=10,
                random_state=0, verbose=0)
    elif method == 'km':
        model = KMeans(
                n_clusters=n_clusters,
                random_state=0, verbose=0)

    elif method == 'agglomerative':
        model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward', compute_distances=True)
    else:
        raise ValueError(f'Method `{method}` not recognized')
    print(x.shape)
    labels = model.fit_predict(x)
    print('n_clusters:', np.unique(labels).size)

    if sort:
        labels = sort_labels(labels)[0]

    labels_arr = np.full(mask.shape, labels.min()-1, dtype=int)
    labels_arr[mask] = labels

    return labels_arr, model

def remove_margins(embs, mar):
    for ke, va in embs.items():
        embs[ke] = [
                v[mar[0][0]:-mar[0][1], mar[1][0]:-mar[1][1]]
                for v in va]


def get_mask_embeddings(embs, mar=16, min_connected=4000):

    n_clusters = 2

    # remove margins to avoid border effects 移除边距以避免边框效果
    remove_margins(embs, ((mar, mar), (mar, mar)))

    # get features
    x = np.concatenate(list(embs.values()))

    # segment image
    labels, __ = cluster(x, n_clusters=n_clusters, method='km')
    labels = relabel_small_connected(labels, min_size=min_connected)

    # select cluster for foreground
    rgb = np.stack(embs['rgb'], -1)
    i_foreground = np.argmax([
        rgb[labels == i].std() for i in range(n_clusters)])
    mask = labels == i_foreground

    # restore margins
    extent = [(-mar, s+mar) for s in mask.shape]
    mask = crop_image(
            mask, extent,
            mode='constant', constant_values=mask.min())

    return mask


def main(prefix):

    inpfile = f'{prefix}embeddings-hist.pickle'
    outfile = f'{prefix}mask-small.png'

    embs = load_pickle(inpfile)
    mask = get_mask_embeddings(embs)
    save_image(mask, outfile)

