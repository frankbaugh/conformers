import numpy as np
import ase
from ase.neighborlist import neighbor_list
import torch
import graphite
from graphite.nn.models.alignn import Encoder, Processor, ALIGNN
from graphite.utils.alignn import line_graph, dihedral_graph, get_bnd_angs, get_dih_angs
from graphite.data import AngularGraphPairData
from torch_geometric.data import Data

ovito_cutoff = {
    ('H', 'C'): 1.74, ('H', 'N'): 1.65,  ('H', 'O'): 1.632,
    ('C', 'C'): 2.04, ('C', 'N'): 1.95,  ('C', 'O'): 1.932,
    ('N', 'N'): 1.86, ('N', 'O'): 1.842, ('O', 'O'): 1.824,
}

def get_molecular_graph(atoms):
    """Returns edge indices of strong chemical bonds according to a pre-defined
    element-specific cutoff criteria.  
    """
    dummy_cell = np.diag([30, 30, 30])
    dummy_pbc  = np.array([False]*3)
    i, j, d = neighbor_list('ijd', atoms, cutoff=ovito_cutoff)
    return np.stack((i, j)), d

def get_x(symbols, all_atoms):
    N_max = len(symbols)
    
    X = np.zeros((N_max, len(all_atoms)))

    for idx, atom in enumerate(symbols):
        X[idx, all_atoms.index(atom)] = 1

    return X

def atoms2pygdata(atoms, all_atoms):
    """Converts ASE `atoms` into a PyG graph data holding the molecular graph (G) and the angular graph (A).
    The angular graph holds both bond and dihedral angles.
    """
    edge_index_G, x_bnd = get_molecular_graph(atoms)
    edge_index_bnd_ang = line_graph(edge_index_G)
    edge_index_dih_ang = dihedral_graph(edge_index_G)
    edge_index_A = np.hstack([edge_index_bnd_ang, edge_index_dih_ang])
    x_atm = get_x(atoms.get_chemical_symbols(), all_atoms)
    #x_atm = OneHotEncoder(sparse=False).fit_transform(atoms.numbers.reshape(-1,1))
    x_bnd_ang = get_bnd_angs(atoms, edge_index_G, edge_index_bnd_ang)
    x_dih_ang = get_dih_angs(atoms, edge_index_G, edge_index_dih_ang)
    x_ang = np.concatenate([x_bnd_ang, x_dih_ang])
    mask_dih_ang = [False]*len(x_bnd_ang) + [True]*len(x_dih_ang)

    data = AngularGraphPairData(
        edge_index_G = torch.tensor(edge_index_G, dtype=torch.long),
        edge_index_A = torch.tensor(edge_index_A, dtype=torch.long),
        x_atm        = torch.tensor(x_atm,        dtype=torch.float),
        x_bnd        = torch.tensor(x_bnd,        dtype=torch.float),
        x_ang        = torch.tensor(x_ang,        dtype=torch.float),
        mask_dih_ang = torch.tensor(mask_dih_ang, dtype=torch.bool),
    )
    data.x_bnd_ang = torch.tensor(x_bnd_ang, dtype=torch.float)
    data.x_dih_ang = torch.tensor(x_dih_ang, dtype=torch.float)
    
    return data

def name2graph(name):
    
    path = '/Users/frank/code/conf/37conf8/MMFF94/'
    all_atoms = ["H", "C", "N", "O", "F", "P", "S", "Cl"] # In order of atomic number
    atoms = ase.io.read(path + name + '_MMFF94.xyz')
    graph = atoms2pygdata(atoms)
    
    return graph
    
def angulargraphpairer(conf_i, conf_j):
    """ The two graphs are placed in a block-diagonal fashion into one big graph. The bond angles of both
    are put in the first half of x_ang and the dihedrals in the second
    """
    j_edge_index_G = conf_j.edge_index_G + len(conf_i.x_atm) # ensure nodes dont overlap by adding number of nodes in i to j
    j_edge_index_A = conf_j.edge_index_A + len(conf_i.x_bnd)
    
    # Rearrange mask_dih_ang to reflect which angles are dihedrals (only the ones at the end)
    mask_dih_ang = [False] * (len(conf_i.x_bnd_ang) + len(conf_j.x_bnd_ang)) + [True] * (len(conf_i.x_dih_ang) + len(conf_j.x_dih_ang))
    
    data = AngularGraphPairData(
        edge_index_G = torch.cat((conf_i.edge_index_G, j_edge_index_G), dim=-1),
        edge_index_A = torch.cat((conf_i.edge_index_A, j_edge_index_A), dim=-1),
        x_atm        = torch.cat((conf_i.x_atm, conf_j.x_atm), dim=0),
        x_bnd        = torch.cat((conf_i.x_bnd, conf_j.x_bnd), dim=-1),
        x_ang        = torch.cat((conf_i.x_bnd_ang, conf_j.x_bnd_ang, conf_i.x_dih_ang, conf_j.x_dih_ang), dim=-1),
        mask_dih_ang = torch.tensor(mask_dih_ang, dtype=torch.bool),
        )
    return data