import numpy as np
import pandas as pd
from ase.io import read
from ase import Atoms
import os
import xarray as xr
import linecache

# Don't even need to create any graphs, models.SchNet takes 2 torch.tensor parameters:
# z; Atomic number of each atom with shape [num_atoms] 
# pos; Coordinates of each atom with shape [num_atoms, 3]

maxN = 65
nconfs = 8
nmols = 37
nperms = nconfs * (nconfs - 1)


path = '/Users/frank/code/conf/37Conf8/MMFF94/'
# path = '/home/fmvb21/MMFF94/'

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

print(mollist)
print("==========================")


z_padded = np.zeros((nmols * nconfs, maxN))
positions_padded = np.zeros((nmols * nconfs, maxN, 3))

for idx, filename in enumerate(filelist):
    atoms = read(path + filename)
    atomic_nums = atoms.get_atomic_numbers()
    atomic_pos = atoms.get_positions()
    z_padded[idx, :len(atomic_nums)] = atomic_nums
    positions_padded[idx, :len(atomic_pos), :] = atoms.get_positions()


z_pairs = np.zeros((2, nmols * nperms, maxN))
position_pairs = np.zeros((2, nmols * nperms, maxN, 3))
labels = np.zeros(nmols * nperms)
permfilelist = []

def energy_comparison(molconf1, molconf2):
    
    enline1 = linecache.getline(path + molconf1, 2)
    enline2 = linecache.getline(path + molconf2, 2)
    en1 = float(enline1[5:].rstrip())
    en2 = float(enline2[5:].rstrip())
    
    if en1 < en2: return 1., 0.
    else: return 0., 1.


for mol in range(0, nconfs * nmols, nconfs):
    perm_count = 0
    for conf_i in range(nconfs):
        for conf_j in range(conf_i +1, nconfs):
            
            idx = int((mol / nconfs) * nperms)
            
            conf_i_z = z_padded[mol + conf_i, :]
            conf_j_z = z_padded[mol + conf_j, :]
            
            
            pair1_z = np.stack((conf_i_z, conf_j_z))
            pair2_z = np.stack((conf_j_z, conf_i_z))
            
            conf_i_pos = positions_padded[mol + conf_i, :, :]
            conf_j_pos = positions_padded[mol + conf_j, :, :]
            
            pair1_pos = np.stack((conf_i_pos, conf_j_pos))
            pair2_pos = np.stack((conf_j_pos, conf_i_pos))
            
            molconf1 = filelist[mol + conf_i]
            molconf2 = filelist[mol + conf_j]
            
            label1, label2 = energy_comparison(molconf1, molconf2)
            col_title1 = molconf1.replace("_MMFF94.xyz", "") + molconf2.replace("_MMFF94.xyz", "")[-1]
            permfilelist.append(col_title1)
            col_title2 = molconf2.replace("_MMFF94.xyz", "") + molconf1.replace("_MMFF94.xyz", "")[-1]
            permfilelist.append(col_title2)
            
            z_pairs[:, idx + perm_count, :] = pair1_z
            position_pairs[:, idx + perm_count, :, :] = pair1_pos
            labels[idx + perm_count] = label1
            
            perm_count += 1
            
            z_pairs[:, idx + perm_count, :] = pair2_z
            position_pairs[:, idx + perm_count, :, :] = pair2_pos
            labels[idx + perm_count] = label2
            
            perm_count += 1
            


# Data structure: position_pairs:
# first index = 0 or 1 for each conformer
# second index = which pair of conformers
# 3rd index = which atom
# 4th index = x, y and coordinates

# z_pairs:
# first index = 0 or 1 for each conformer
# second index = which pair of conformers
# 3rd index = which atomic number

np.save('z_pairs.npy', z_pairs)
np.save('position_pairs.npy', position_pairs)
np.save('graph_labels.npy', labels)
print(labels)