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
    r_cutoff = 5.0
    N_atoms = positions.shape[0]
    adj_mat = np.zeros((N_max, N_max))
    for i in range(N_atoms):
        for j in range(i + 1, N_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < r_cutoff:
                adj_mat[i][j] = adj_mat[j][i] = 10 / dist
    
    # See dihedrals ALIGN-d paper
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

    data.validate(raise_on_error=True)
    data_list.append(data)
    
    # do the X node features align with the network ordering? who knows....

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

def energy_comparison(molconf1, molconf2):
    
    enline1 = linecache.getline(path + molconf1, 2)
    enline2 = linecache.getline(path + molconf2, 2)
    en1 = float(enline1[5:].rstrip())
    en2 = float(enline2[5:].rstrip())
    one = torch.Tensor([1.])
    zero = torch.Tensor([0.])
    
    # return torch.Tensor([en2 - en1]), torch.Tensor([en1 - en2])

    if en1 < en2: return one, zero
    elif en2 < en1: return zero, one
    elif en1 == en2: return torch.Tensor([0.5]), torch.Tensor([0.5])
    else: return 'panic', 'panic'


# pair_datas
paired_data_list = []

for mol in tqdm(range(0, nconfs * nmols, nconfs)):
    perm_count = 0
    for conf_i in range(nconfs):
        for conf_j in range(conf_i +1, nconfs):
            
            molconf1 = filelist[mol + conf_i]
            molconf2 = filelist[mol + conf_j]
            
            label1, label2 = energy_comparison(molconf1, molconf2)
            
            pair1 = [data_list[mol + conf_i], data_list[mol + conf_j], label1]
            paired_data_list.append(pair1)
            
            
            pair2 = [data_list[mol + conf_j], data_list[mol + conf_i], label2]
            paired_data_list.append(pair2)


# Train / Test split:

def get_data():
    frac = int(0.9 * nmols) / nmols
    test_idx = int(frac * nmols * nperms)
    
    for data in paired_data_list:
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)
        data[2] = data[2].to(device)
    
    train_data = paired_data_list[:test_idx]
    test_data = paired_data_list[test_idx:]
    return train_data, test_data

train_data, test_data = get_data()

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer_size = 8
        self.gcn1 = SGConv(in_channels=-1, out_channels=self.hidden_layer_size)
        self.gcn2 = SGConv(in_channels=self.hidden_layer_size, out_channels=self.hidden_layer_size)
        
        self.l1 = nn.Linear(2 * self.hidden_layer_size, 4)
        self.l2 = nn.Linear(4, 1)
    
    def forward(self, graph1, graph2):
        
        x1, edge_index1, edge_attr1 = graph1.x, graph1.edge_index, graph1.edge_attr
        x2, edge_index2, edge_attr2 = graph2.x, graph2.edge_index, graph2.edge_attr

        x1 = leaky_relu(self.gcn1(x1, edge_index1, edge_attr1))
        x1 = leaky_relu(self.gcn2(x1, edge_index1, edge_attr1))
        x1 = x1.sum(dim=0)

        x2 = leaky_relu(self.gcn1(x2, edge_index2, edge_attr2))
        x2 = leaky_relu(self.gcn2(x2, edge_index2, edge_attr2))
        x2 = x2.sum(dim=0)
        
        x = torch.cat((x1, x2), dim=0)
        x = leaky_relu(self.l1(x))
        x = self.l2(x)
        
        return x

model = Net().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-1)
epochs = 100
test_loss_list = []

def train_loop(train_data, model, loss_fn, optimizer):
    model.train()
    
    for data in train_data:
        optimizer.zero_grad()
        out = model(data[0], data[1])
        loss = loss_fn(out, data[2])
        loss.backward()
        optimizer.step()
    return loss

def test_loop(test_data, model, loss_fn):
    model.eval()
    size = len(test_data)
    test_loss = 0
    correct = 0
    
    for data in test_data:
        out = model(data[0], data[1])
        test_loss += loss_fn(out, data[2]).item()
        correct += ((out > 0.5).squeeze().long() == data[2].squeeze().long()).sum().item()
    
    test_loss /= len(test_data)
    correct /= size
    test_loss_list.append(test_loss)
    # print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss:{test_loss:>8f} \n")

for t in range(epochs):
    # print(f"Epoch {t+1}\n------------------------")
    train_loop(train_data, model, loss_fn, optimizer)
    test_loop(test_data, model, loss_fn)
    if (t % 2 == 0): plot(test_loss_list)


print("Done!")