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
from torch_geometric.transforms import ToDevice
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import coo_matrix
from uniplot import plot
from ase.neighborlist import neighbor_list

path = '/Users/frank/code/conf/37Conf8/MMFF94/'
# path = '/home/fmvb21/MMFF94/'

def set_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
        print("Using GPU")
    else:
        device = torch.device("cpu")   # Use CPU
        print("Using CPU")
    
    return device
device = set_gpu()

def get_filelist():
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
    return filelist, mollist
filelist, mollist = get_filelist()

def get_graph(atoms):
    """ Returns, in COO format the graph with nodes = atoms, edges = bonds
    Can easily get this to return edge vectors if needed later; specify 'ijdD'
    Also maybe preferable (see ALIGNN-d) to just use global large cutoff, say G3
    neighbour_list sorts by i but not necessarily j, this is ok """
    ovito_cutoff = {
    ('H', 'C'): 1.74, ('H', 'N'): 1.65,  ('H', 'O'): 1.632,
    ('C', 'C'): 2.04, ('C', 'N'): 1.95,  ('C', 'O'): 1.932,
    ('N', 'N'): 1.86, ('N', 'O'): 1.842, ('O', 'O'): 1.824,
    }
    i, j, d = neighbor_list('ijd', atoms, ovito_cutoff)
    
    edge_index_G, bond_lengths = np.stack((i,j)), d.astype(np.float32)
    return edge_index_G, bond_lengths

def get_x_atoms(symbols):
    """Returns One-hot atom types"""
    N_max = len(symbols)
    all_atoms = ["H", "C", "N", "O", "F", "P", "S", "Cl"] # In order of atomic number
    X = np.zeros((N_max, len(all_atoms)))
    for idx, atom in enumerate(symbols):
        X[idx, all_atoms.index(atom)] = 1

    return X


def dihedral_graph(edge_index_G):
    """Return the "dihedral angle line graph" of the input graph.
    This is a devastating little 4 lines from the Alignn-d source code that I don't understand at all,
    hopefully it works -_-
    """
    src, dst = edge_index_G
    edge_index_A = [
        (u, v)
        for i, j in edge_index_G.T
        for u in np.flatnonzero((dst == i) & (src != j))
        for v in np.flatnonzero((dst == j) & (src != i))
    ]
    return np.array(edge_index_A).T
    

def atoms2pygdata(atoms, filename):
    
    #Define graph connectivity based on specific cutoffs for each bond type
    x_atoms = get_x_atoms(atoms.get_chemical_symbols())
    edge_index_G, bond_lengths = get_graph(atoms)
    first_line_graph = get_first_line_graph(edge_index_G)
    edge_index_A = dihedral_graph(edge_index_G)
    print(line_graph_G)



def read_atoms():
    data_list = []
    print("Generating dihedral graphs....")
    for idx, filename in enumerate(tqdm(filelist)):
        atoms = read(path + filename)
        data = atoms2pygdata(atoms, filename)
        data_list.append(data)
        
    return data_list

data_list = read_atoms()