import multiprocessing
import torch
from torch.utils.data import Dataset
import numpy as np
from utils import read_lines, read_string, save_pickle, load_pickle, load_tsv, load_image, get_disk_mask
from train import get_model as train_load_model
from scipy.spatial.distance import cdist
from model import scstGCN

def pad(emd, num):
    h, w = emd.shape[0], emd.shape[1]

    pad_h = (num - h % num) % num
    pad_w = (num - w % num) % num

    padded_matrix = np.pad(emd,
                           ((0, pad_h), (0, pad_w), (0, 0)),
                           'constant', constant_values=0)

    new_h, new_w = padded_matrix.shape[:2]
    assert new_h % num == 0 and new_w % num == 0
    return padded_matrix

class SpotDataset(Dataset):

    def __init__(self, x_all, y, locs, radius):
        super().__init__()
        mask = get_disk_mask(radius)
        x = get_patches_flat(x_all, locs, mask)

        self.x = x
        self.y = y
        self.locs = locs
        self.size = x_all.shape[:2]
        self.radius = radius
        self.mask = mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    mask = np.ones_like(mask, dtype=bool)
    # 2-D  shape=[201,201]  r=[[-101, 101], [-101, 101]]
    center = shape // 2  #
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    for s in locs:
        patch = img[
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]

        x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list

def get_locs(prefix, target_shape=None):

    locs = load_tsv(f'{prefix}locs.csv')

    locs = np.stack([locs['y'], locs['x']], -1)

    if target_shape is not None:
        wsi = load_image(f'{prefix}he.jpg')
        current_shape = np.array(wsi.shape[:2])
        rescale_factor = current_shape // target_shape
        locs = locs.astype(float)
        locs /= rescale_factor

    locs = locs.round().astype(int)

    return locs

def get_data(prefix):
    gene_names = read_lines(f'{prefix}gene-names.txt')

    cnts = load_tsv(f'{prefix}cnts.csv')
    cnts = cnts.iloc[:, cnts.var().to_numpy().argsort()[::-1]]
    cnts = cnts[gene_names]

    embs = load_pickle(f'{prefix}Multimodal_feature_map.pickle')
    embs = np.concatenate([embs['his'], embs['rgb'], embs['pos']]).transpose(1, 2, 0)

    locs = get_locs(prefix, target_shape=embs.shape[:2])
    return embs, cnts, locs

def get_model_kwargs(kwargs):
    return get_model(**kwargs)

def get_model(
        x, y, locs, radius, prefix, batch_size, epochs, lr,
        load_saved=False, device='cuda'):

    print('x:', x.shape, ', y:', y.shape)

    x = x.copy()

    dataset = SpotDataset(x, y, locs, radius)
    model = train_load_model(
            model_class=scstGCN,
            model_kwargs=dict(
                n_inp=x.shape[-1],
                n_out=y.shape[-1],
                lr=lr),
            dataset=dataset, prefix=prefix,
            batch_size=batch_size, epochs=epochs,
            load_saved=load_saved, device=device)
    model.eval()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return model, dataset

def normalize(embs, cnts):

    embs = embs.copy()
    cnts = cnts.copy()

    # TODO: check if adjsut_weights in extract_features can be skipped
    embs_mean = np.nanmean(embs, (0, 1))
    embs_std = np.nanstd(embs, (0, 1))
    embs -= embs_mean
    embs /= embs_std + 1e-12

    cnts_min = cnts.min(0)
    cnts_max = cnts.max(0)
    cnts -= cnts_min
    cnts /= (cnts_max - cnts_min) + 1e-12

    return embs, cnts, (embs_mean, embs_std), (cnts_min, cnts_max)

def predict_single_out(model, z, indices, names, y_range):
    z = torch.tensor(z, device=model.device)
    y = model.lat_to_out(z, indices=indices)
    y = y.cpu().detach().numpy()

    y *= y_range[:, 1] - y_range[:, 0]
    y += y_range[:, 0]
    return y


def predict_single_lat(model, x):
    x = torch.tensor(x, device=model.device)
    z = model.inp_to_lat(x)

    z = z.cpu().detach().numpy()
    return z

def predict(h,w,
        model_states, x_batches, name_list, y_range, prefix,
        device='cuda', gra_size=7):

    batch_size_outcome = 100

    model_states = [mod.to(device) for mod in model_states]

    z_states_batches_1 = [
            [predict_single_lat(mod, x_bat) for mod in model_states]
            for x_bat in x_batches]
    z_point = np.concatenate([
        np.median(z_states, 0)
        for z_states in z_states_batches_1])

    z_1 = np.zeros((h + (gra_size - h % gra_size), w + (gra_size - w % gra_size), z_point.shape[-1]))
    k = 0
    for i in range(0, h, gra_size):
        for j in range(0, w, gra_size):
            z_1[i:i + gra_size, j:j + gra_size, :] = z_point[k]
            k = k + 1
    z_1 = z_1[0:h, 0:w, :]

    z_dict = dict(cls=z_1.transpose(2, 0, 1))
    save_pickle(
            z_dict,
            prefix+'embeddings-gene.pickle')
    del z_point

    idx_list = np.arange(len(name_list))
    n_groups_outcome = len(idx_list) // batch_size_outcome + 1
    idx_groups = np.array_split(idx_list, n_groups_outcome)
    for idx_grp in idx_groups:
        name_grp = name_list[idx_grp]
        y_ran = y_range[idx_grp]
        y_grp = np.concatenate([
            np.median([
                predict_single_out(mod, z, idx_grp, name_grp, y_ran)
                for mod, z in zip(model_states, z_states)], 0)
            for z_states in z_states_batches_1])

        z_1 = np.zeros((h + (gra_size - h % gra_size), w + (gra_size - w % gra_size), y_grp.shape[-1]))
        k = 0
        for i in range(0, h, gra_size):
            for j in range(0, w, gra_size):
                z_1[i:i + gra_size, j:j + gra_size, :] = y_grp[k]
                k = k + 1
        z_1 = z_1[0:h, 0:w, :]
    print(f'All genes have been saved in {prefix}cnts-super/..')


def impute(
        embs, cnts, locs, radius, epochs, batch_size, prefix,
        n_states=1, load_saved=False, device='cuda', n_jobs=1):

    names = cnts.columns
    cnts = cnts.to_numpy()
    cnts = cnts.astype(np.float32)

    __, cnts, __, (cnts_min, cnts_max) = normalize(embs, cnts)


    kwargs_list = [
            dict(
                x=embs, y=cnts, locs=locs, radius=radius,
                batch_size=batch_size, epochs=epochs, lr=1e-4,
                prefix=f'{prefix}states/{i:02d}/',
                load_saved=load_saved, device=device)
            for i in range(n_states)]

    if n_jobs is None or n_jobs < 1:
        n_jobs = n_states
    if n_jobs == 1:
        out_list = [get_model_kwargs(kwargs) for kwargs in kwargs_list]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            out_list = pool.map(get_model_kwargs, kwargs_list)

    model_list = [out[0] for out in out_list]
    dataset_list = [out[1] for out in out_list]
    mask_size = dataset_list[0].mask.sum()

    cnts_range = np.stack([cnts_min, cnts_max], -1)
    cnts_range /= mask_size

    ####################################--######---------
    gra_size = 7
    h, w = embs.shape[0], embs.shape[1]
    embs_1 = pad(embs, gra_size)

    batch_size_row = gra_size
    n_batches_row = embs_1.shape[0] // batch_size_row

    batch_size_col = gra_size
    n_batches_col = embs_1.shape[1] // batch_size_col

    embs_batches = np.array_split(embs_1, n_batches_row, axis=0)

    embs_batches = [np.array_split(i, n_batches_col, axis=1) for i in embs_batches]
    del embs_1
    del embs

    predict(h, w,
            model_states=model_list, x_batches=embs_batches,
            name_list=names, y_range=cnts_range,
            prefix=prefix, device=device, gra_size=gra_size)



def main(prefix, epoch=500, device='cuda', n_states=5, load_saved=False):
    embs, cnts, locs = get_data(prefix)

    factor = 16
    radius = int(read_string(f'{prefix}radius.txt'))
    radius = radius / factor

    n_train = cnts.shape[0]
    batch_size = min(128, n_train//16)

    impute(
            embs=embs, cnts=cnts, locs=locs, radius=radius,
            epochs=epoch, batch_size=batch_size,
            n_states=n_states, prefix=prefix,
            load_saved=load_saved,
            device=device, n_jobs=1)
