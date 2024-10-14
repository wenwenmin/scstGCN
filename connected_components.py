import numpy as np
from scipy.ndimage import label as label_connected
from utils import sort_labels, get_most_frequent

def get_adjacent(ind):
    adj = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing='ij')
    adj = np.stack(adj, -1)
    adj = adj.reshape(-1, adj.shape[-1])
    adj = adj[(adj != 0).any(-1)]
    adj += ind
    return adj


def split_by_connected_size_single(labels, min_size):
    labels = label_connected(labels)[0]
    labels -= 1
    labels = sort_labels(labels)[0]
    counts = np.unique(labels[labels >= 0], return_counts=True)[1]
    cut = np.sum(counts >= min_size)
    small = labels - cut
    small[small < 0] = -1
    large = labels.copy()
    large[labels >= cut] = -1
    return small, large


def split_by_connected_size(labels, min_size):
    labs_uniq = np.unique(labels[labels >= 0])
    small = np.full_like(labels, -1)
    large = np.full_like(labels, -1)
    for lab in labs_uniq:
        isin = labels == lab
        sma, lar = split_by_connected_size_single(isin, min_size)
        issma = sma >= 0
        islar = lar >= 0
        small[issma] = sma[issma] + small.max() + 1
        large[islar] = lar[islar] + large.max() + 1
    return small, large


def relabel_small_connected(labels, min_size):
    labels = labels.copy()
    small, __ = split_by_connected_size(labels, min_size)
    small = sort_labels(small, descending=False)[0]
    small_uniq = np.unique(small[small >= 0])
    lab_na = min(-1, labels.min() - 1)
    for lab_small in small_uniq:

        isin = small == lab_small
        lab = labels[isin][0]

        indices = np.stack(np.where(isin), -1)
        labs_adj = []
        labs_small_adj = []
        for ind in indices:
            adj = get_adjacent(ind)
            is_within = np.logical_and(
                    (adj < labels.shape).all(-1),
                    (adj >= 0).all(-1))
            adj[~is_within] = 0
            la = labels[adj[:, 0], adj[:, 1]]
            lsa = small[adj[:, 0], adj[:, 1]]
            la[~is_within] = lab_na
            lsa[~is_within] = lab_na
            labs_adj.append(la)
            labs_small_adj.append(lsa)
        labs_adj = np.stack(labs_adj)
        labs_small_adj = np.stack(labs_small_adj)
        is_other = (labs_adj >= 0) * (labs_adj != lab)
        if is_other.any():
            lab_new = get_most_frequent(labs_adj[is_other])
            i_new, i_adj_new = np.stack(
                    np.where(labs_adj == lab_new), -1)[0]
            ind_new = get_adjacent(indices[i_new])[i_adj_new]
            lab_small_new = small[ind_new[0], ind_new[1]]
        else:
            lab_new = lab
            lab_small_new = lab_small
        labels[isin] = lab_new
        small[isin] = lab_small_new

    return labels

