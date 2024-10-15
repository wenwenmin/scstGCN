from PIL import Image
import pickle
import os
import cv2 as cv
import numpy as np
import pandas as pd
Image.MAX_IMAGE_PIXELS = None

def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)


def load_image(filename):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    return img


def save_image(img, filename):
    mkdir(filename)
    Image.fromarray(img).save(filename)


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def read_string(filename):
    return read_lines(filename)[0]


def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)


def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    return x


def load_tsv(filename, index=True):
    if index:
        index_col = 0
    else:
        index_col = None
    df = pd.read_csv(filename, header=0, index_col=index_col)
    return df


def get_most_frequent(x):
    # return the most frequent element in array
    uniqs, counts = np.unique(x, return_counts=True)
    return uniqs[counts.argmax()]


def sort_labels(labels, descending=True):
    labels = labels.copy()
    isin = labels >= 0
    labels_uniq, labels[isin], counts = np.unique(
            labels[isin], return_inverse=True, return_counts=True)
    c = counts
    if descending:
        c = c * (-1)
    order = c.argsort()
    rank = order.argsort()
    labels[isin] = rank[labels[isin]]
    return labels, labels_uniq[order]

def get_HVG_genes(prefix, n_top):

    cnts = load_tsv(f'{prefix}cnts.csv')
    order = cnts.var().to_numpy().argsort()[::-1]
    names = cnts.columns.to_list()
    names_all = [names[i] for i in order]
    names_top = names_all[:n_top]
    mkdir(f'{prefix}gene-names.txt')
    with open(f'{prefix}gene-names.txt', 'w') as file:
        for s in names_top:
            file.write(f'{s}\n')

def impute_missing(x, mask, radius=3, method='ns'):

    method_dict = {
            'telea': cv.INPAINT_TELEA,
            'ns': cv.INPAINT_NS}
    method = method_dict[method]

    x = x.copy()
    if x.dtype == np.float64:
        x = x.astype(np.float32)

    x[mask] = 0
    mask = mask.astype(np.uint8)

    expand_dim = np.ndim(x) == 2
    if expand_dim:
        x = x[..., np.newaxis]
    channels = [x[..., i] for i in range(x.shape[-1])]
    y = [cv.inpaint(c, mask, radius, method) for c in channels]
    y = np.stack(y, -1)
    if expand_dim:
        y = y[..., 0]

    return y


def smoothen(
        x, size, kernel='uniform', backend='cv', mode='mean',
        impute_missing_values=True, device='cuda'):

    if x.ndim == 3:
        expand_dim = False
    elif x.ndim == 2:
        expand_dim = True
        x = x[..., np.newaxis]
    else:
        raise ValueError('ndim must be 2 or 3')

    mask = np.isfinite(x).all(-1)
    if (~mask).any() and impute_missing_values:
        x = impute_missing(x, ~mask)

    if kernel == 'uniform':
        if backend == 'cv':
            kernel = np.ones((size, size), np.float32) / size**2
            y = cv.filter2D(
                    x, ddepth=-1, kernel=kernel,
                    borderType=cv.BORDER_REFLECT)
            if y.ndim == 2:
                y = y[..., np.newaxis]
        else:
            raise ValueError('backend must be cv')
    else:
        raise ValueError('kernel must be uniform')

    if not mask.all():
        y[~mask] = np.nan

    if expand_dim and y.ndim == 3:
        y = y[..., 0]
    return y

def get_disk_mask(radius, boundary_width=None):
    radius_ceil = np.array(radius).astype(int)
    locs = np.meshgrid(
            np.arange(-radius_ceil, radius_ceil+1),
            np.arange(-radius_ceil, radius_ceil+1),
            indexing='ij')
    locs = np.stack(locs, -1)
    distsq = (locs**2).sum(-1)
    isin = distsq <= radius**2
    if boundary_width is not None:
        isin *= distsq >= (radius-boundary_width)**2
    return isin
