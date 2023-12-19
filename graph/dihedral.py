import numpy as np
import pandas as pd
from ase.io import read
from ase import Atoms
import os
import xarray as xr
import linecache
import torch
from torch import nn
from torch.nn.functional import leaky_relu
from torch_geometric.nn import SGConv, global_add_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, to_networkx, scatter
from torch_geometric.transforms import ToDevice
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import coo_matrix
from uniplot import plot
import graphite
from graphite.nn.models.alignn import Encoder, Processor, ALIGNN
from graphutils import name2graph, angulargraphpairer
from datautils import get_filelist, get_conflist, get_mollist, shuffle_eights, get_path, get_energies, get_abs_MMFF94

class Decoder(nn.Module):
    def __init__(self, node_dim, out_dim):
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        
    def forward(self, data):
        # Globally pool all atom/bond/angle components into a single feature vector
        # This operation assumes all components have the same dimensionality, 
        # which should be the case for ALIGNN
        if data.num_graphs != 1: #if data is batched
            h_atm_pooled = scatter(data.h_atm, data.x_atm_batch, dim=0, reduce='sum')
            #h_bnd_pooled = scatter(data.h_bnd, data.x_bnd_batch, dim=0, reduce='sum')
            #h_ang_pooled = scatter(data.h_ang, data.x_ang_batch, dim=0, reduce='sum')
            h_pooled = h_atm_pooled
        else: # for single graphs, only need to sum along one axis
            h_pooled = data.h_atm.sum(dim = 0)
        return h_pooled

def set_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
        print("Using GPU")
    else:
        device = torch.device("cpu")   # Use CPU
        print("Using CPU")
    
    return device


def read_atoms(path, device):
    data_list = []
    conflist = get_conflist(path)
    print("Generating dihedral graphs....")
    for conf in tqdm(conflist):
        data = name2graph(conf)
        data = data.to(device)
        data_list.append(data)
    
    return data_list

# data_list = shuffle_eights(data_list)

def energy_comparison(conf1, conf2, energies, device):
    
    en1, en2 = energies[conf1], energies[conf2]
    label1 = torch.tensor([en1 - en2], dtype=torch.float32, device=device)
    label2 = torch.tensor([en2 - en1], dtype=torch.float32, device=device)
        
    return label1, label2


def get_paired_data(path, data_list, device):
    """pairs the conformers up, create tuple of conf1, conf2, energy, append this list to paired_data_list"""
    print("Graphs generated, pairing ....")
    CCSD_energies, MMFF94_energies = get_energies(path)
    MMFF94_abs = get_abs_MMFF94(path)
    filelist = get_filelist(path)
    paired_data_list = []
    nconfs, nmols = 8, 37
    for mol in tqdm(range(0, nconfs * nmols, nconfs)):
        for conf_i in range(nconfs):
            for conf_j in range(conf_i + 1, nconfs):
                
                molconf1 = filelist[mol + conf_i].replace("_MMFF94.xyz", "")
                molconf2 = filelist[mol + conf_j].replace("_MMFF94.xyz", "")
                
                label1, label2 = energy_comparison(molconf1, molconf2, CCSD_energies, device)
                MMFF_pair1 = torch.tensor([MMFF94_energies[molconf1], MMFF94_energies[molconf2]], device=device)
                MMFF_pair2 = torch.tensor([MMFF94_energies[molconf2], MMFF94_energies[molconf1]], device=device)
                
                pair1 = angulargraphpairer(data_list[mol + conf_i], data_list[mol + conf_j])
                pair1.forcepair = MMFF_pair1
                pair1.y = label1
                
                pair2 = angulargraphpairer(data_list[mol + conf_j], data_list[mol + conf_i])
                pair2.forcepair = MMFF_pair2
                pair2.y = label2
                
                paired_data_list.append(pair1)
                paired_data_list.append(pair2)

    return paired_data_list


def print_graph(graph):
    G = nx.to_networkx_graph(graph.edge_index_G.T)
    G_pos = nx.spring_layout(G)
    
    nx.draw(G, G_pos, node_size=50)
    plt.savefig('graph1_plot.png')

def split_data(paired_data_list, batch_size):
    nmols, nconfs = 37, 8
    nperms = nconfs * (nconfs - 1)

    frac = int(0.8 * nmols) / nmols
    test_idx = int(frac * nmols * nperms)
    print(test_idx)

    train_data = paired_data_list[:test_idx]
    test_data = paired_data_list[test_idx:]
    
    
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, follow_batch=['x_atm', 'x_bnd', 'x_ang'])
    test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True, follow_batch=['x_atm', 'x_bnd', 'x_ang'])
    
    return train_data, test_data

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        inner_size = 16
        self.alignnd = ALIGNN(
        encoder   = Encoder(num_species=10, cutoff=5.0, dim=inner_size, dihedral=True),
        processor = Processor(num_convs=3, dim=inner_size),
        decoder   = Decoder(node_dim=inner_size, out_dim=1),
        )
        self.l1 = nn.Linear(inner_size + 2, inner_size)
        self.l2 = nn.Linear(inner_size, 1)
    
    def forward(self, data):
        if data.num_graphs != 1:
            data.forcepair = data.forcepair.view(data.num_graphs, 2)

        x = self.alignnd(data)
        x = torch.cat((x, data.forcepair), dim=-1)

        x = leaky_relu(self.l1(x))
        x = self.l2(x)

        return x


def train_loop(train_data, model, loss_fn, optimizer, batch_size):
    model.train()
    
    for data in train_data:
        optimizer.zero_grad()
        out = model(data)
        if batch_size != 1:
            out = torch.squeeze(out)

        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
    return loss

def test_loop(test_data, model, loss_fn, batch_size, test_loss_list):
    model.eval()
    test_loss = 0
    
    for data in test_data:
        out = model(data)
        if batch_size != 1:
            out = torch.squeeze(out)
        test_loss += loss_fn(out, data.y).item()

    test_loss /= len(test_data)
    test_loss_list.append(test_loss)
    # print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss:{test_loss:>8f} \n")

def run_epochs(epochs, learning_rate, batch_size):
    print("running dihedral")
    device = set_gpu()
    path = get_path()
    data_list = read_atoms(path, device)
    
    paired_data_list = get_paired_data(path, data_list, device)
    train_data, test_data = split_data(paired_data_list, batch_size)

    
    model = Net().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.5e-2) # 0.00001 lr
    
    test_loss_list = []
    
    for t in range(epochs):
        # print(f"Epoch {t+1}\n------------------------")
        train_loop(train_data, model, loss_fn, optimizer, batch_size)
        test_loop(test_data, model, loss_fn, batch_size, test_loss_list)
        if (t % 2 == 0): plot(test_loss_list)
    
    torch.save(model.state_dict(), 'dihedral_trained_rel.pt')
    
    print("Done!")


