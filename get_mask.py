from utils import sort_labels
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from utils import save_image, load_pickle
from connected_components import relabel_small_connected

def prepare_for_clustering(embs, location_weight):
    mask = np.all([np.isfinite(c) for c in embs], axis=0)
    embs = np.stack([c[mask] for c in embs], axis=-1)

    if location_weight is None:
        x = embs
    else:
        embs -= embs.mean(0)
        embs /= embs.var(0).sum()**0.5

        locs = np.meshgrid(
                *[np.arange(mask.shape[i]) for i in range(mask.ndim)],
                indexing='ij')
        locs = np.stack(locs, -1).astype(float)
        locs = locs[mask]
        locs -= locs.mean(0)
        locs /= locs.var(0).sum()**0.5

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


def crop_image(img, extent, mode='edge', constant_values=None):
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    if (pad != 0).any():
        kwargs = {}
        if mode == 'constant' and constant_values is not None:
            kwargs['constant_values'] = constant_values
        img = np.pad(img, pad, mode=mode, **kwargs)
        extent += pad[:extent.shape[0], [0]]
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
    return img

def get_mask_embeddings(embs, mar=16, min_connected=4000):

    n_clusters = 2

    remove_margins(embs, ((mar, mar), (mar, mar)))

    x = np.concatenate(list(embs.values()))

    labels, __ = cluster(x, n_clusters=n_clusters, method='km')
    labels = relabel_small_connected(labels, min_size=min_connected)

    rgb = np.stack(embs['rgb'], -1)
    i_foreground = np.argmax([
        rgb[labels == i].std() for i in range(n_clusters)])
    mask = labels == i_foreground

    extent = [(-mar, s+mar) for s in mask.shape]
    mask = crop_image(
            mask, extent,
            mode='constant', constant_values=mask.min())

    return mask


def main(prefix):

    inpfile = f'{prefix}Multimodal_feature_map.pickle'
    outfile = f'{prefix}mask.png'

    embs = load_pickle(inpfile)
    mask = get_mask_embeddings(embs)
    save_image(mask, outfile)

