import math
import torch
import numpy as np
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from utils import get_disk_mask
from scipy.spatial.distance import cdist

def get_odj(h, w):

    coordinates = np.array([[i, j] for i in range(h) for j in range(w)])

    distances = cdist(coordinates, coordinates)

    adj_matrix = np.zeros((h*w, h*w), dtype=int)

    for i in range(h*w):

        distances_from_node = distances[i]
        sorted_indices = np.argsort(distances_from_node)

        nearest_indices1 = sorted_indices[1:5]

        for idx in nearest_indices1:
            adj_matrix[i][idx] = 1
            adj_matrix[idx][i] = 1

    return adj_matrix

class GraphConvLayer(nn.Module):
    def __init__(self, num_features, num_hidden):
        super(GraphConvLayer,self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(num_features, num_hidden))
        self.func = nn.LeakyReLU(0.1, inplace=True)
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, x, indices=None):
        k=0
        if x.dim() == 4:
            k=1
            h, w = x.shape[1], x.shape[2]
            x = x.reshape(x.shape[0], h * w, x.shape[3])
            adj = get_odj(h, w)
        else:
            h = int(math.sqrt(x.shape[1]))
            adj = get_odj(h, h)


        adj = torch.Tensor(adj).to('cuda')
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
            if self.bias is not None:
                output = output + self.bias[indices]
        output = self.func(output)
        if k==1:
            output = output.reshape(x.shape[0], h, w, output.shape[2])
        return output

class Linear(nn.Module):
    def __init__(self, num_hidden, num_genes, alpha, beta, bias=False):
        super(Linear, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(num_hidden,num_genes))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(num_genes))
        else:
            self.register_parameter('bias', None)
        self.func = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta
        self.reset_parameters()

    def reset_parameters(self):
        std = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, x, indices=None):

        if indices is None:
            output = torch.matmul(x, self.weights)
            if self.bias is not None:
                output = output + self.bias
        else:
            weight = self.weights[:, indices]
            output = torch.matmul(x, weight)
            if self.bias is not None:
                output = output + self.bias[indices]
        output = self.func(output) + self.beta
        return output

class scstGCN(pl.LightningModule):
    def __init__(self, lr, num_features, num_genes, ori_radius, bias=False):
        super(scstGCN, self).__init__()

        self.lr = lr
        self.ori_radius = ori_radius

        self.GCN_module = nn.Sequential(
            GraphConvLayer(num_features, 512),
            GraphConvLayer(512, 512))

        self.output_module = Linear(512, num_genes, alpha=0.01, beta=0.01, bias=bias)

        self.save_hyperparameters()

    def get_hidden(self, x):
        x = self.GCN_module.forward(x)
        x = F.dropout(x, 0.5, training=self.training)
        return x

    def get_gene(self, x, indices=None):
        x = self.output_module.forward(x, indices)
        return x

    def forward(self, x, indices=None):
        x = self.get_hidden(x)
        x = self.get_gene(x, indices)
        return x

    def training_step(self, batch, batch_idx):
        x, y_mean = batch
        mask = get_disk_mask(self.ori_radius/16)
        mask = torch.BoolTensor(mask).to('cuda')
        y_pred = self.forward(x)
        y_pred = y_pred.reshape(y_pred.shape[0], mask.shape[0], mask.shape[1], y_pred.shape[2])
        y_pred = torch.masked_select(y_pred, mask.unsqueeze(0).unsqueeze(-1)).view(y_pred.shape[0], -1, y_pred.shape[-1])

        y_mean_pred = y_pred.mean(-2)

        mse = ((y_mean_pred - y_mean)**2).mean()
        loss = mse
        self.log('loss', loss**0.5, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
