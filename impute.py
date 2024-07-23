import argparse
import multiprocessing
import math
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import numpy as np
import torch.nn.functional as F
from impute_by_basic import get_gene_counts, get_embeddings, get_locs
from utils import read_lines, read_string, save_pickle
from image import get_disk_mask
from train import get_model as train_load_model
from scipy.spatial.distance import cdist


def get_odj_5(h, w):

    # 创建节点的坐标
    coordinates = np.array([[i, j] for i in range(h) for j in range(w)])

    # 计算节点之间的欧氏距离
    distances = cdist(coordinates, coordinates)

    # 初始化邻接矩阵
    adj_matrix = np.zeros((h*w, h*w), dtype=int)

    # 对于每个节点，找到距离最近的五个节点，并在邻接矩阵中标记为相连
    for i in range(h*w):
        # 获取当前节点与其他节点的距离，并按距离排序
        distances_from_node = distances[i]
        sorted_indices = np.argsort(distances_from_node)

        # 获取距离最近的五个节点的索引（排除自己）
        nearest_indices1 = sorted_indices[1:5]

        # 在邻接矩阵中标记相连的节点
        for idx in nearest_indices1:
            adj_matrix[i][idx] = 1
            adj_matrix[idx][i] = 1  # 对称地设置连接

        # for i in range(adj_matrix.shape[0]):
        #     adj_matrix[i,i] = 1
    #adj_matrix = symmetric_normalized_laplacian(adj_matrix)
    return adj_matrix

def pad(emd, num):
    h, w = emd.shape[0], emd.shape[1]

    # 计算高度和宽度需要填充到最接近的7的倍数
    pad_h = (num - h % num) % num  # 高度需要填充的零的数量
    pad_w = (num - w % num) % num  # 宽度需要填充的零的数量

    # 对矩阵进行零填充
    padded_matrix = np.pad(emd,
                           ((0, pad_h), (0, pad_w), (0, 0)),
                           'constant', constant_values=0)

    # 验证结果
    new_h, new_w = padded_matrix.shape[:2]
    assert new_h % num == 0 and new_w % num == 0
    return padded_matrix

class ELU(nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta

class Feed(nn.Module):
    def __init__(self,input_features,output_features,func=None, bias=False):
        super(Feed,self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weights = nn.Parameter(torch.FloatTensor(input_features,output_features))
        if func is None:
            # TODO: change activation to LeakyRelu(0.01)
            func = nn.LeakyReLU(0.1, inplace=True)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias',None)
        self.func = func
        self.reset_parameters()

    def reset_parameters(self):
        std = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std,std)
        if self.bias is not None:
            self.bias.data.uniform_(-std,std)
    def forward(self,x, indices=None):

        if indices is None:
            output = torch.matmul(x, self.weights)
            if self.bias is not None:
                output = output + self.bias
        else:
            weight = self.weights[:, indices]
            output = torch.matmul(x, weight)
            if self.bias is not None:
                output = output + self.bias[indices]
        output = self.func(output)
        return output

class FeedForward(nn.Module):
    def __init__(self,input_features,output_features,func=None, bias=False):
        super(FeedForward,self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weights = nn.Parameter(torch.FloatTensor(input_features,output_features))
        if func is None:
            # TODO: change activation to LeakyRelu(0.01)
            func = nn.LeakyReLU(0.1, inplace=True)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias',None)
        self.func = func
        self.reset_parameters()

    def reset_parameters(self):
        std = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std,std)
        if self.bias is not None:
            self.bias.data.uniform_(-std,std)

    def forward(self,x, indices=None):
        k=0
        if x.dim() == 4:  # 预测时 不是训练时
            k=1
            h, w = x.shape[1], x.shape[2]
            x = x.reshape(x.shape[0], h * w, x.shape[3])
            adj = get_odj_5(h, w)
        else:
            h = int(math.sqrt(x.shape[1]))
            adj = get_odj_5(h, h)


        adj = torch.Tensor(adj).to('cuda:0')
        adj = adj.unsqueeze(0).repeat(x.shape[0], 1, 1)
        if indices is None:
            output = torch.matmul(x, self.weights)
            output = torch.bmm(adj, output)
            #for i in range(x.shape[0]):
            #    output[i] = torch.mm(adj, output[i])
            if self.bias is not None:
                output = output + self.bias
        else:
            weight = self.weights[:, indices]
            output = torch.matmul(x, weight)
            output = torch.bmm(adj, output)
            #for i in range(x.shape[0]):
            #    output[i] = torch.mm(adj, output[i])
            if self.bias is not None:
                output = output + self.bias[indices]
        output = self.func(output)
        if k==1:
            output = output.reshape(x.shape[0], h, w, output.shape[2])
        return output

class ForwardSumModel(pl.LightningModule):
    def __init__(self, lr, n_inp, n_out, bias=False):
        super(ForwardSumModel,self).__init__()
        self.lr = lr
        self.input_size=n_inp
        self.hidden_size=512
        self.num_class = n_out

        self.net_lat = nn.Sequential(
                        FeedForward(n_inp, 512),
                        FeedForward(512, 512))

        self.net_out = Feed(self.hidden_size,n_out,func=ELU(alpha=0.01, beta=0.01), bias=bias)
        self.dropout = 0.5
        self.save_hyperparameters()

    def inp_to_lat(self, x):
        x = self.net_lat.forward(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def lat_to_out(self, x, indices=None):
        x = self.net_out.forward(x, indices)
        return x

    def forward(self, x, indices=None):
        x = self.inp_to_lat(x)
        x = self.lat_to_out(x, indices)
        return x

    def training_step(self, batch, batch_idx):
        x, y_mean = batch
        mask = get_disk_mask(55/16)
        mask = torch.BoolTensor(mask).to('cuda')
        y_pred = self.forward(x)
        y_pred = y_pred.reshape(y_pred.shape[0], mask.shape[0], mask.shape[1], y_pred.shape[2])
        y_pred = torch.masked_select(y_pred, mask.unsqueeze(0).unsqueeze(-1)).view(y_pred.shape[0], -1, y_pred.shape[-1])  # （batch_size, super_size_in_spot, gene_nums）

        y_mean_pred = y_pred.mean(-2)

        norm_y = torch.norm(y_mean).to('cuda')
        norm_y_pre = torch.norm(y_mean_pred).to('cuda')
        corr_loss = (((y_mean.t() @ y_mean)/norm_y - (y_mean_pred.t() @ y_mean_pred)/norm_y_pre)**2).mean()

        mse = ((y_mean_pred - y_mean)**2).mean()
        loss = mse + 0.1 * corr_loss
        self.log('loss', loss**0.5, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

class SpotDataset(Dataset):

    def __init__(self, x_all, y, locs, radius):
        super().__init__()
        mask = get_disk_mask(radius)
        x = get_patches_flat(x_all, locs, mask)  #（n, a, b, C）n是spot_batch. ab是spot圈住的超像素点  C是c1+c2+3
        #isin = np.isfinite(x).all((-1, -2))
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
    shape = np.array(mask.shape)  # 横纵
    mask = np.ones_like(mask, dtype=bool)
    # 换成2维  shape=[201,201]  r=[[-101, 101], [-101, 101]]
    center = shape // 2  #
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    for s in locs:
        patch = img[  # 这里是把圈着spot的大正方形相加，下面还有mask操作
                s[0]+r[0][0]:s[0]+r[0][1],  # s[0]是这个spot的基准线，然后正负一个spot的半径
                s[1]+r[1][0]:s[1]+r[1][1]]

        x = patch[mask]  # 一个spot对应的圈里的特征向量 （a,b,C） C=c1+c1+3
        x_list.append(x)
    x_list = np.stack(x_list)  # 在0轴上堆叠
    return x_list

def get_data(prefix):
    gene_names = read_lines(f'{prefix}gene-names.txt')
    cnts = get_gene_counts(prefix)
    cnts = cnts[gene_names]  # 只要高变基因
    embs = get_embeddings(prefix)
    # embs = embs[..., :192]  # use high-level features only
    # embs = reduce_embeddings(embs)
    locs = get_locs(prefix, target_shape=embs.shape[:2])
    # embs = add_coords(embs)
    return embs, cnts, locs

def get_model_kwargs(kwargs):
    return get_model(**kwargs)

def get_model(
        x, y, locs, radius, prefix, batch_size, epochs, lr,
        load_saved=False, device='cuda'):

    print('x:', x.shape, ', y:', y.shape)  # x是特征 y是空转

    x = x.copy()

    dataset = SpotDataset(x, y, locs, radius)  # torch.utils.data.Dataset 类的实例
    model = train_load_model(
            model_class=ForwardSumModel,
            model_kwargs=dict(
                n_inp=x.shape[-1],  # c1+c2+3
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
    # states: different initial values for training
    # batches: subsets of observations
    # groups: subsets outcomes

    batch_size_outcome = 100

    model_states = [mod.to(device) for mod in model_states]  # 5个模型

    # get features of second last layer
    z_states_batches_1 = [  # batch个元素  每一个有
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


    # predict and save y by batches in outcome dimension
    idx_list = np.arange(len(name_list))
    n_groups_outcome = len(idx_list) // batch_size_outcome + 1
    idx_groups = np.array_split(idx_list, n_groups_outcome)
    for idx_grp in idx_groups:
        name_grp = name_list[idx_grp]
        y_ran = y_range[idx_grp]  # y_range：形状是（2，n） 表示每个基因在pixel中的最大最小值
        y_grp = np.concatenate([
            np.median([
                predict_single_out(mod, z, idx_grp, name_grp, y_ran)
                for mod, z in zip(model_states, z_states)], 0)  # z: 135 7 7 512
            for z_states in z_states_batches_1])

        z_1 = np.zeros((h + (gra_size - h % gra_size), w + (gra_size - w % gra_size), y_grp.shape[-1]))
        k = 0
        for i in range(0, h, gra_size):
            for j in range(0, w, gra_size):
                z_1[i:i + gra_size, j:j + gra_size, :] = y_grp[k]
                k = k + 1
        z_1 = z_1[0:h, 0:w, :]
        for i, name in enumerate(name_grp):
            save_pickle(z_1[..., i], f'{prefix}cnts-super/{name}.pickle')


def impute(  # n_states=5（表示训练几遍，每一遍出一个结果）, load_saved=False, n_jobs=1
        embs, cnts, locs, radius, epochs, batch_size, prefix,
        n_states=1, load_saved=False, device='cuda', n_jobs=1):

    names = cnts.columns
    cnts = cnts.to_numpy()
    cnts = cnts.astype(np.float32)

    __, cnts, __, (cnts_min, cnts_max) = normalize(embs, cnts)  # 标准化

    # mask = np.isfinite(embs).all(-1)
    # embs[~mask] = 0.0

    kwargs_list = [  # 训练多次
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

    model_list = [out[0] for out in out_list]  # 5个model
    dataset_list = [out[1] for out in out_list]  # 5个数据集  是继承的torch.utils.data
    mask_size = dataset_list[0].mask.sum()  # 一个spot内包含几个pixel? .sum()是求true的数量

    # embs[~mask] = np.nan
    # cnts_min:一维数组，每个基因在所又spot中的最小值 =cnts.min(0)
    cnts_range = np.stack([cnts_min, cnts_max], -1)  # 每个基因在spot的最小最大值 每行是一个基因 有两列
    cnts_range /= mask_size  # 每个基因在pixel的最小最大值

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
    # 1、保存了所有基因的嵌入（256维，MLP的最后一层的输出）（h',w',1000）
    # 2、分别保存每个基因的预测值 1000个文件
    predict(h, w,
            model_states=model_list, x_batches=embs_batches,
            name_list=names, y_range=cnts_range,
            prefix=prefix, device=device, gra_size=gra_size)



def main(prefix, epoch=500, device='cuda', n_states=5, load_saved=False):
    embs, cnts, locs = get_data(prefix)

    factor = 16
    radius = int(read_string(f'{prefix}radius.txt'))  # spot半径
    radius = radius / factor  # 特征图缩小了16倍，直径也缩小（对应覆盖面）

    n_train = cnts.shape[0]  # spot数量 一个spot是一个样本
    batch_size = min(128, n_train//16)

    impute(
            embs=embs, cnts=cnts, locs=locs, radius=radius,
            epochs=epoch, batch_size=batch_size,
            n_states=n_states, prefix=prefix,
            load_saved=load_saved,
            device=device, n_jobs=1)
