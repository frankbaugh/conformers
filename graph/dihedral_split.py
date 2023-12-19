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
from dihedral import Decoder, set_gpu, read_atoms, get_paired_data

class Decoder(nn.Module):
    def __init__(self, node_dim, out_dim):
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        
    def forward(self, data):
        # Globally pool all atom/bond/angle components into a single feature vector
         # for single graphs, only need to sum along one axis
        h_pooled = data.h_atm.sum(dim = 0)
        return h_pooled


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        inner_size = 16
        self.alignnd1 = ALIGNN(
        encoder   = Encoder(num_species=8, cutoff=5.0, dim=inner_size, dihedral=True),
        processor = Processor(num_convs=3, dim=inner_size),
        decoder   = Decoder(node_dim=inner_size, out_dim=1),
        )
        self.alignnd2 = ALIGNN(
        encoder   = Encoder(num_species=8, cutoff=5.0, dim=inner_size, dihedral=True),
        processor = Processor(num_convs=3, dim=inner_size),
        decoder   = Decoder(node_dim=inner_size, out_dim=1),
        )
            
        self.l1 = nn.Linear(2 * inner_size, inner_size)
        self.l2 = nn.Linear(inner_size, 1)
    
    def forward(self, data):
        
        x1 = data[0]
        x2 = data[1]
        
        x1 = self.alignnd1(x1)
        x2 = self.alignnd2(x2)
        
        x = torch.cat((x1, x2), dim=-1)
        x = leaky_relu(self.l1(x))
        x = self.l2(x)

        return x

def get_graph_pair(conf_i, conf_j):
    
    pair = Batch.from_data_list([conf_i, conf_j])
    return pair


def en_comparison(conf_i, conf_j, CCSD_energies, device):
    
    en1, en2 = CCSD_energies[conf_i], CCSD_energies[conf_j]
    endiff = torch.tensor([en1 - en2], dtype=torch.float32, device=device)
        
    return endiff

def get_paired_data_split(path, graphs, device):
    """pairs the conformers up, create tuple of conf1, conf2, energy, append this list to paired_data_list"""
    print("Graphs generated, pairing ....")
    CCSD_energies, MMFF94_energies = get_energies(path)
    conflist, mollist = get_conflist(path), get_mollist(path)
    paired_data_list = []
    
    for mol in mollist:
        confs = [conf for conf in conflist if mol in conf]
        for conf_i in confs:
            for conf_j in confs:
                if conf_i != conf_j:
                    
                    label = en_comparison(conf_i, conf_j, CCSD_energies, device)
                    graph_pair = [graphs[conf_i], graphs[conf_j], label]
                    paired_data_list.append(graph_pair)
    
    print(len(paired_data_list))
    return paired_data_list


def get_graphs(path):
    conflist = get_conflist(path)
    graphs = {f'{conf}': name2graph(conf) for conf in conflist}
    
    return graphs

def get_split_data(paired_data_list):
    nmols, nconfs = 37, 8
    nperms = nconfs * (nconfs - 1)

    frac = int(0.8 * nmols) / nmols
    test_idx = int(frac * nmols * nperms)
    print(test_idx)
    
    train_data = paired_data_list[:test_idx]
    test_data = paired_data_list[test_idx:]
    
    return train_data, test_data
    
def train_loop(train_data, model, loss_fn, optimizer):
    model.train()
    
    for data in train_data:
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data[2])
        loss.backward()
        optimizer.step()
    return loss

def test_loop(test_data, model, loss_fn, test_loss_list):
    model.eval()
    test_loss = 0
    
    for data in test_data:
        out = model(data)
        test_loss += loss_fn(out, data[2]).item()

    test_loss /= len(test_data)
    test_loss_list.append(test_loss)
    # print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss:{test_loss:>8f} \n")


def run_split_epochs(epochs, learning_rate):
    
    device = set_gpu()
    path = get_path()
    graphs = get_graphs(path)
    
    paired_data_list = get_paired_data_split(path, graphs, device)
    train_data, test_data = get_split_data(paired_data_list)
    
    model = Net().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.5e-2) # 0.00001 lr
    test_loss_list = []
    
    for t in range(epochs):
        # print(f"Epoch {t+1}\n------------------------")
        train_loop(train_data, model, loss_fn, optimizer)
        test_loop(test_data, model, loss_fn, test_loss_list)
        if (t % 2 == 0): plot(test_loss_list)
    
    torch.save(model.state_dict(), 'split_dihedral.pt')
    
    print("Done!")
    

