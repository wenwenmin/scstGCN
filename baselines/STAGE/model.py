import os
import time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import pandas as pd
import numpy as np
import anndata as ad
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from .loss import *
from .dataset import *
from .utils import *


class Encoder(nn.Module):
    def __init__(self, gene_number, X_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(gene_number, 1000)
        self.fc1_bn = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc2_bn = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 50)
        self.fc3_bn = nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(50, 10)
        self.fc4_bn = nn.BatchNorm1d(10)
        self.fc5 = nn.Linear(10, X_dim)
    def forward(self, input, relu):
        h1 = F.relu(self.fc1_bn(self.fc1(input)))
        h2 = F.relu(self.fc2_bn(self.fc2(h1)))
        h3 = F.relu(self.fc3_bn(self.fc3(h2)))
        h4 = F.relu(self.fc4_bn(self.fc4(h3)))
        if relu:
            return F.relu(self.fc5(h4))
        else:
            return self.fc5(h4)


class Decoder(nn.Module):
    def __init__(self, gene_number, X_dim):
        super(Decoder, self).__init__()
        self.fc6 = nn.Linear(X_dim, 10)
        self.fc6_bn = nn.BatchNorm1d(10)
        self.fc7 = nn.Linear(10, 50)
        self.fc7_bn = nn.BatchNorm1d(50)
        self.fc8 = nn.Linear(50, 500)
        self.fc8_bn = nn.BatchNorm1d(500)
        self.fc9 = nn.Linear(500, 1000)
        self.fc9_bn = nn.BatchNorm1d(1000)
        self.fc10 = nn.Linear(1000, gene_number)
    def forward(self, z, relu):
        h6 = F.relu(self.fc6_bn(self.fc6(z)))
        h7 = F.relu(self.fc7_bn(self.fc7(h6)))
        h8 = F.relu(self.fc8_bn(self.fc8(h7)))
        h9 = F.relu(self.fc9_bn(self.fc9(h8)))
        if relu:
            return F.relu(self.fc10(h9))
        else:
            return self.fc10(h9)


def STAGE(
        adata,
        save_path='./STAGE_results',
        data_type='10x',
        experiment='generation',
        down_ratio=0.5,
        coord_sf=77,
        sec_name='section',
        select_section=[1, 3, 5, 6, 8],
        gap=0.05,
        train_epoch=2000,
        seed=1234,
        batch_size=512,
        learning_rate=1e-3,
        w_recon=0.1,
        w_w=0.1,
        w_l1=0.1,
        step_size=500,
        gamma=1,
        relu=True,
        device='cpu',
        path1='file_tmp',
):
    """ This functions outputs generated or recovered data.

        Args:
            adata: AnnData object storing preprocessed original data.
            save_path: File path saving results including net and AnnData object.
            data_type: Data type. Available options are: "ST_KTH", "10x", and "Slide-seq". Default is "10x". Among them,
                "ST_KTH" is "Spatial Transcriptomics" data developed by KTH Royal Institute of Technology, which refers
                to the earliest sequencing-based low-resolution ST data,
                "10x" is 10x Visium data, which is improved on the basis of "ST_KTH" and has been commercially available
                on a large scale, and
                "Slide-seq" is sequencing-based high-resolution (near single-cell level) ST data.
            experiment: Different tasks. Available options are: "generation" and "recovery" when data_type = "10x";
                "generation" when data_type = "ST_KTH"; "3d_model" when data_type = "Slide-seq". Default is "generation".
            down_ratio: Down-sampling ratio. Default is 0.5.
            coord_sf: Size factor to scale spatial location. Default is 77.
            sec_name: Item in adata.obs.columns used for choosing training sections.
            select_section: Index of training sections.
            gap: Distance between simulated and real sections. Half of distance between adjacent real sections.
                These parameters (sec_name, select_section, and gap) are available when experiment = "3d_model".
            train_epoch: Training epoch number. Default is 2000.
            batch_size: Batch size. Default is 512.
            learning_rate: Learning rate. Default is 1e-3.
            w_recon: Weight of reconstruction loss in total loss. Default is 0.1.
            w_w: Weight of W loss in latent loss. Default is 0.1.
            w_l1: Weight of L1 loss in reconstruction loss. Default is 0.1.
            step_size: Step size for learning rate dampling. Default is 500.
            gamma: Learning rate dampling ratio. Default is 1.
            relu: Whether the output layer of encoder and decoder activated by ReLU. Default is True.
        Return:
            adata_stage: Generated AnnData object when experiment = "generation";
                Recovered AnnData object when experiment = "recovery";
                Generated AnnData object in real sections when experiment = "3d_model".
            adata_simu: Generated AnnData object in simulated sections. Available when experiment = "3d_model".
            adata_sample: Down-sampled AnnData object. Available when experiment = "recovery".
    """

    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # Set the device for the computation
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # First check if data_type is valid
    valid_data_types = ['10x', 'ST_KTH', 'Slide-seq']
    if data_type not in valid_data_types:
        raise ValueError(f"Valid data type must be one of the following: {', '.join(valid_data_types)}")

    # Then check if experiment is valid based on data_type
    if data_type == '10x' and experiment not in ['generation', 'recovery']:
        raise ValueError("Experiments designed for 10x Visium data are only 'generation' and 'recovery'.")
    elif data_type == 'ST_KTH' and experiment != 'generation':
        raise ValueError("Experiment designed for Spatial Transcriptomics data is only 'generation'.")
    elif data_type == 'Slide-seq' and experiment != '3d_model':
        raise ValueError("Experiment designed for Slide-seq data is only '3d_model'.")

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isdir(path1):
        os.mkdir(path1)

    # Preparation
    if experiment=='generation' and data_type=='10x':
        coor_df, fill_coor_df = generation_coord_10x(adata)
        used_gene, normed_data = get_data(adata, experiment=experiment)
    elif experiment=='recovery' and data_type=='10x':
        coor_df, fill_coor_df, sample_index, sample_barcode = recovery_coord(adata, down_ratio=down_ratio, path1=path1)
        used_gene, normed_data, adata_sample = get_data(adata, experiment=experiment, sample_index=sample_index, sample_barcode=sample_barcode, path1=path1)
    elif experiment=='generation' and data_type=='ST_KTH':
        coor_df, fill_coor_df = generation_coord_ST(adata)
        used_gene, normed_data = get_data(adata, experiment=experiment)
    elif experiment=='3d_model' and data_type=='Slide-seq':
        used_gene, normed_data = get_data(adata, experiment=experiment, sec_name=sec_name, select_section=select_section)
        coor_df, fill_coor_df, new_coor_df, all_coor_df = Slide_seq_coord_3d(
            adata,sec_name=sec_name,select_section=select_section,gap=gap)

    if experiment == 'recovery':
        normed_coor_df = coor_df.copy()
        normed_coor_df = normed_coor_df / coord_sf

        normed_fill_coor_df = fill_coor_df.copy()
        normed_fill_coor_df = normed_fill_coor_df / coord_sf

        normed_coor_df.to_csv(path1+"/coord.txt", header=0, index=0)
        normed_fill_coor_df.to_csv(path1+"/fill_coord.txt", header=0, index=0)

    if experiment == 'generation' or experiment == 'recovery':
        normed_coor_df = coor_df / coord_sf
        X_dim = 2
    elif experiment == '3d_model' and data_type == 'Slide-seq':
        normed_coor_df = coor_df
        normed_coor_df.iloc[:, range(2)] = normed_coor_df.iloc[:, range(2)] / coord_sf
        X_dim = 3

    transformed_dataset = MyDataset(normed_data=normed_data, coor_df=normed_coor_df, transform=transforms.Compose([ToTensor()]))
    train_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    # Training process
    gene_number = normed_data.shape[0]
    encoder, decoder = Encoder(gene_number, X_dim), Decoder(gene_number, X_dim)

    encoder.train()
    decoder.train()

    encoder, decoder = encoder.to(device), decoder.to(device)

    enc_optim = optim.Adam(encoder.parameters(), lr=learning_rate)
    dec_optim = optim.Adam(decoder.parameters(), lr=learning_rate)

    enc_sche = optim.lr_scheduler.StepLR(enc_optim, step_size=step_size, gamma=gamma)
    dec_sche = optim.lr_scheduler.StepLR(dec_optim, step_size=step_size, gamma=gamma)


    with tqdm(range(train_epoch), total=train_epoch, desc='Epochs') as epoch:
        for j in epoch:

            train_loss = []
            train_lc_loss = []
            train_re_loss = []

            for xdata, xlabel in train_loader:
                xdata = xdata.to(torch.float32)
                xlabel = xlabel.to(torch.float32)

                enc_optim.zero_grad()
                dec_optim.zero_grad()

                xdata, xlabel, = Variable(xdata.to(device)), Variable(xlabel.to(device))

                latent = encoder(xdata, relu)
                latent = latent.view(-1, X_dim)
                xlabel = xlabel.float().to(device)
                latent_loss = loss1(latent, xlabel) + w_w * sliced_wasserstein_distance(latent, xlabel, 1000, device=device)
                xrecon = decoder(latent, relu)
                recon_loss = loss2(xrecon, xdata) + w_l1 * loss1(xrecon, xdata)

                total_loss = 0.1 * latent_loss + 0.1 * w_recon * recon_loss

                total_loss.backward()

                enc_optim.step()
                dec_optim.step()

                enc_sche.step()
                dec_sche.step()

                train_lc_loss.append(latent_loss.item())
                train_re_loss.append(recon_loss.item())
                train_loss.append(total_loss.item())

            epoch_info = 'latent_loss: %.5f, recon_loss: %.5f, total_loss: %.5f' % \
                         (torch.mean(torch.FloatTensor(train_lc_loss)),
                          torch.mean(torch.FloatTensor(train_re_loss)),
                          torch.mean(torch.FloatTensor(train_loss)))
            epoch.set_postfix_str(epoch_info)

    torch.save(encoder, save_path+'/encoder.pth')
    torch.save(decoder, save_path+'/decoder.pth')

    encoder.eval()
    decoder.eval()

    # Get generated or recovered data
    if experiment=='generation' or experiment=='recovery':
        normed_fill_coor_df = fill_coor_df.copy()
        normed_fill_coor_df = normed_fill_coor_df / coord_sf
        normed_fill_coor_df = torch.from_numpy(np.array(normed_fill_coor_df))
        normed_fill_coor_df = normed_fill_coor_df.to(torch.float32)
        normed_fill_coor_df = Variable(normed_fill_coor_df.to(device))
        generate_profile = decoder(normed_fill_coor_df, relu)
        generate_profile = generate_profile.cpu().detach().numpy()

        if not relu:
            generate_profile = np.clip(generate_profile, a_min=0, a_max=None)

        if experiment=='recovery':
            np.savetxt(save_path+"/fill_data.txt", generate_profile)

        adata_stage = sc.AnnData(generate_profile)
        adata_stage.obsm["coord"] = fill_coor_df.to_numpy()
        adata_stage.var.index = used_gene

        adata.write(save_path + '/original_data.h5ad')

        if experiment=='generation':
            adata_stage.write(save_path + '/generated_data.h5ad')
            return adata_stage
        elif experiment=='recovery' and data_type=='10x':
            adata_sample.write(save_path + '/sampled_data.h5ad')
            adata_stage.obs = adata.obs
            adata_stage.write(save_path + '/recovered_data.h5ad')
            return adata_sample, adata_stage

    elif experiment=='3d_model' and data_type=='Slide-seq':

        # recovered data
        normed_fill_coor_df = fill_coor_df.copy()
        normed_fill_coor_df.iloc[:, range(2)] = normed_fill_coor_df.iloc[:, range(2)] / coord_sf
        normed_fill_coor_df = torch.from_numpy(np.array(normed_fill_coor_df))
        normed_fill_coor_df = normed_fill_coor_df.to(torch.float32)
        normed_fill_coor_df = Variable(normed_fill_coor_df.to(device))
        generate_profile = decoder(normed_fill_coor_df, relu)
        generate_profile = generate_profile.cpu().detach().numpy()

        if not relu:
            generate_profile = np.clip(generate_profile, a_min=0, a_max=None)

        adata_stage = sc.AnnData(generate_profile)
        adata_stage.obsm["coord"] = fill_coor_df.to_numpy()
        adata_stage.var.index = used_gene
        adata_stage.obs = adata.obs

        # generated data
        normed_new_coor_df = new_coor_df.copy()
        normed_new_coor_df.iloc[:, range(2)] = normed_new_coor_df.iloc[:, range(2)] / coord_sf
        normed_new_coor_df = torch.from_numpy(np.array(normed_new_coor_df))
        normed_new_coor_df = normed_new_coor_df.to(torch.float32)
        normed_new_coor_df = Variable(normed_new_coor_df.to(device))
        generate_profile = decoder(normed_new_coor_df, relu)
        generate_profile = generate_profile.cpu().detach().numpy()

        if not relu:
            generate_profile = np.clip(generate_profile, a_min=0, a_max=None)

        adata_simu = sc.AnnData(generate_profile)
        adata_simu.obsm["coord"] = new_coor_df.to_numpy()
        adata_simu.var.index = used_gene
        new_coor_df.index = adata_simu.obs.index
        new_coor_df.columns = ["xcoord", "ycoord", "zcoord"]
        adata_simu.obs = new_coor_df

        # all data
        adata_all = ad.concat([adata_stage, adata_simu], merge="same")

        adata.write(save_path + '/original_data.h5ad')
        adata_stage.write(save_path + '/recovered_data.h5ad')
        adata_simu.write(save_path + '/simulated_data.h5ad')
        adata_all.write(save_path + "/all_data.h5ad")

        return adata_stage, adata_simu, adata_all


def VGE(
        adata,
        save_path='./VGE_results',
        #down_ratio=0.5,
        coord_sf=77,
        train_epoch=2000,
        seed=1234,
        batch_size=512,
        learning_rate=1e-3,
        w_recon=0.1,
        w_l1=0.1,
        step_size=500,
        gamma=1,
        relu=True,
        device='cpu',
        path1='file_tmp',
):
    """ This functions outputs recovered data.

        Args:
            adata: AnnData object storing original data. Raw data should to be normalized. Highly variable genes should be identified.
            save_path: File path saving results including net and AnnData object.
            down_ratio: Down-sampling ratio. Default is 0.5.
            coord_sf: Size factor to scale spatial location. Default is 77.
            train_epoch: Training epoch number. Default is 2000.
            batch_size: Batch size. Default is 512.
            learning_rate: Learning rate. Default is 1e-3.
            w_recon: Weight of reconstruction loss in total loss. Default is 0.1.
            w_l1: Weight of L1 loss in reconstruction loss. Default is 0.1.
            step_size: Step size for learning rate dampling. Default is 500.
            gamma: Learning rate dampling ratio. Default is 1.
            relu: Whether the output layer of encoder and decoder activated by ReLU. Default is True.

        Return:
            adata_vge: Recovered AnnData object.
            adata_sample: Down-sampled AnnData object.
    """

    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # Set the device for the computation
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Preparation
    #coor_df, fill_coor_df, sample_index, sample_barcode = recovery_coord(adata, down_ratio=down_ratio)
    normed_coor_df=pd.read_csv(path1 + "/coord.txt", header=None, index_col=None, sep=",")
    normed_fill_coor_df=pd.read_csv(path1 + "/fill_coord.txt", header=None, index_col=None, sep=",")
    coor_df = normed_coor_df.copy()
    coor_df = coor_df * coord_sf
    fill_coor_df = normed_fill_coor_df.copy()
    fill_coor_df = fill_coor_df * coord_sf
    s = pd.read_csv(path1 + '/sample_index.txt', header=None, index_col=None)
    sample_index = np.array(s[0])
    #sample_barcode = coor_df.index[sample_index]
    s = pd.read_csv(path1 + '/sample_barcode.txt', header=None, index_col=None)
    sample_barcode = np.array(s[0])
    used_gene, normed_data, adata_sample = get_data(adata, experiment='recovery', sample_index=sample_index, sample_barcode=sample_barcode)

    #normed_coor_df = coor_df / coord_sf
    X_dim = 2

    transformed_dataset = MyDataset(normed_data=normed_data, coor_df=normed_coor_df, transform=transforms.Compose([ToTensor()]))
    train_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    # Training process
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    gene_number = normed_data.shape[0]
    decoder = Decoder(gene_number, X_dim)

    decoder.train()

    decoder = decoder.to(device)

    dec_optim = optim.Adam(decoder.parameters(), lr=learning_rate)

    dec_sche = optim.lr_scheduler.StepLR(dec_optim, step_size=step_size, gamma=gamma)

    with tqdm(range(train_epoch), total=train_epoch, desc='Epochs') as epoch:
        for j in epoch:

            train_loss = []
            train_re_loss = []

            for xdata, xlabel in train_loader:
                xdata = xdata.to(torch.float32)
                xlabel = xlabel.to(torch.float32)

                dec_optim.zero_grad()

                xdata, xlabel, = Variable(xdata.to(device)), Variable(xlabel.to(device))

                xrecon = decoder(xlabel, relu)
                recon_loss = loss2(xrecon, xdata) + w_l1 * loss1(xrecon, xdata)

                total_loss = 0.1 * w_recon * recon_loss

                total_loss.backward()

                dec_optim.step()

                dec_sche.step()

                train_re_loss.append(recon_loss.item())
                train_loss.append(total_loss.item())

            epoch_info = 'recon_loss: %.5f, total_loss: %.5f' % \
                         (torch.mean(torch.FloatTensor(train_re_loss)),
                          torch.mean(torch.FloatTensor(train_loss)))
            epoch.set_postfix_str(epoch_info)

    torch.save(decoder, save_path+'/decoder.pth')

    decoder.eval()

    # Get recovered data
    #normed_fill_coor_df = fill_coor_df.copy()
    #normed_fill_coor_df = normed_fill_coor_df / coord_sf
    normed_fill_coor_df = torch.from_numpy(np.array(normed_fill_coor_df))
    normed_fill_coor_df = normed_fill_coor_df.to(torch.float32)
    normed_fill_coor_df = Variable(normed_fill_coor_df.to(device))
    generate_profile = decoder(normed_fill_coor_df, relu)
    generate_profile = generate_profile.cpu().detach().numpy()

    if not relu:
        generate_profile = np.clip(generate_profile, a_min=0, a_max=None)

    adata_vge = sc.AnnData(generate_profile)



    adata_vge.obsm["coord"] = fill_coor_df.to_numpy()
    adata_vge.var.index = used_gene

    adata.write(save_path + '/original_data.h5ad')

    adata_sample.write(save_path + '/sampled_data.h5ad')
    adata_vge.obs = adata.obs
    adata_vge.write(save_path + '/recovered_data.h5ad')
    return adata_sample, adata_vge

