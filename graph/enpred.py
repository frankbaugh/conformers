import numpy as np
import pandas as pd
from ase.io import read
from ase import Atoms
import os
import xarray as xr
import linecache
import torch
from torch import nn
from torch.nn.functional import relu
from torch_geometric.nn import SSGConv, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import coo_matrix
from uniplot import plot


# Don't even need to create any graphs, models.SchNet takes 2 torch.tensor parameters:
# z; Atomic number of each atom with shape [num_atoms] 
# pos; Coordinates of each atom with shape [num_atoms, 3]
def set_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
        print("Using GPU")
    else:
        device = torch.device("cpu")   # Use CPU
        print("Using CPU")
    
    return device
device = set_gpu()

maxN = 65
nconfs = 8
nmols = 37
nperms = nconfs * (nconfs - 1)


# path = '/Users/frank/code/conf/37Conf8/MMFF94/'
path = '/home/fmvb21/MMFF94/'

filelist = []
mollist = []

for filename in os.listdir(path):
    if filename.endswith('.xyz'):

        filelist.append(filename)
        molecule = filename.replace("_MMFF94.xyz", "")
        molecule = molecule[:-2]
        
        mollist.append(molecule)

filelist = sorted(filelist)
mollist = list(set(mollist))

def get_X(atoms_list):
    N_max = len(atoms_list)
    all_atoms = ["H", "C", "N", "O", "F", "P", "S", "Cl"] # In order of atomic number
    X = np.zeros((N_max, len(all_atoms)))
    
    for idx, atom in enumerate(atoms_list):
        X[idx, all_atoms.index(atom)] = 10

    return torch.Tensor(X)

def get_adj_mat(positions):
    N_max = len(positions)
    r_cutoff = 5
    N_atoms = positions.shape[0]
    adj_mat = np.zeros((N_max, N_max))
    for i in range(N_atoms):
        for j in range(i + 1, N_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < r_cutoff:
                adj_mat[i][j] = adj_mat[j][i] = 10 / dist
    
    
    return adj_mat

def print_net(G):
    pos=nx.spring_layout(G) # pos = nx.nx_agraph.graphviz_layout(G)
    nx.draw_networkx(G,pos)
    plt.savefig('graph.png')

data_list = []

print("Generating graphs....")
for idx, filename in enumerate(tqdm(filelist)):
    atoms = read(path + filename)
    atoms_list = atoms.get_chemical_symbols()
    X = get_X(atoms_list)
    adj_mat = get_adj_mat(positions=atoms.get_positions())
    
    sparse_adj_mat = coo_matrix(adj_mat)
    edge_index = torch.Tensor(sparse_adj_mat.nonzero()).to(torch.int64)
    edge_attr = torch.Tensor(sparse_adj_mat.data)
    
    data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr)
    # G = nx.from_numpy_array(adj_mat)
    # if idx == 0: print_net(G)
    #data = from_networkx(G)
    
    enline1 = linecache.getline(path + filename, 2)
    en1 = torch.Tensor([float(enline1[5:].strip())])

    data_list.append([data, en1])

def shuffle_eights(data_list):
    
    data_list_grouped = []
    for i in range(0, len(data_list), nconfs):
        data_list_grouped.append(data_list[i:i+nconfs])
    print(len(data_list_grouped))
    
    np.random.seed(33)
    np.random.shuffle(data_list_grouped)
    
    data_list_shuffled = sum(data_list_grouped, [])
    
    return data_list_shuffled


data_list = shuffle_eights(data_list)

print("Graphs generated, pairing ....")

# Train / Test split:

def get_data():
    frac = int(0.8 * nmols) / nmols
    test_idx = int(frac * nmols * nconfs)
    for data in data_list:
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)
    
    train_data = data_list[:test_idx]
    test_data = data_list[test_idx:]
    
    return train_data, test_data

train_data, test_data = get_data()

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer_size = 8
        self.gcn1 = SSGConv(in_channels=-1, out_channels=self.hidden_layer_size, alpha=0.1)
        self.gcn2 = SSGConv(in_channels=self.hidden_layer_size, out_channels=self.hidden_layer_size, alpha=0.1)
        
        self.l1 = nn.Linear(self.hidden_layer_size, 4)
        self.l2 = nn.Linear(4, 1)
    
    def forward(self, graph1):
        
        x1, edge_index1, edge_attr1 = graph1.x, graph1.edge_index, graph1.edge_attr

        x1 = relu(self.gcn1(x1, edge_index1, edge_attr1))
        x1 = relu(self.gcn2(x1, edge_index1, edge_attr1))
        x1 = x1.sum(dim=0)
        

        x1 = relu(self.l1(x1))
        x1 = self.l2(x1)
        
        return x1

model = Net().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1)
epochs = 1000

def train_loop(train_data, model, loss_fn, optimizer):
    model.train()
    
    for data in train_data:
        optimizer.zero_grad()
        out = model(data[0])
        loss = loss_fn(out, data[1])
        train_loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    return loss

def test_loop(test_data, model, loss_fn):
    model.eval()
    size = len(test_data)
    test_loss = 0
    correct = 0
    
    for data in test_data:
        out = model(data[0])
        test_loss += loss_fn(out, data[1]).item()
    
    test_loss /= len(test_data)
    test_loss_list.append(test_loss)
    # print(f"Avg loss:{test_loss:>8f} \n")

train_loss_list, test_loss_list, epoch_list = [], [], []
for t in range(epochs):
    epoch_list.append(t)
    train_loop(train_data, model, loss_fn, optimizer)
    test_loop(test_data, model, loss_fn)
    if (t % 25) == 0: plot(test_loss_list)
    

print("Done!")