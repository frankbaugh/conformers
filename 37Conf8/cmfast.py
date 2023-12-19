import numpy as np
from ase.io import read
from dscribe.descriptors import MBTR
import os
import pandas as pd
import pickle
import linecache

def getmaxN():
    path = '/Users/frank/code/conf/37Conf8/MMFF94/'
    os.system('head -n 1 ' + path + '*.xyz | sort -n | tail -n 1')

## getmaxN()
## >> 65 (4225 elements in matrix)
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

## ideally this is more general but not sure for mbtr
descriptor_length = 3600

df = np.zeros((nmols * nconfs, descriptor_length))

labelsdf = np.zeros((nmols * nperms))

mbtr = MBTR(
    species=["H", "N", "C", "O", "F", "Cl", "P", "S"],
    geometry={"function": "inverse_distance"},
    grid={"min": 0, "max": 1, "sigma": 0.1, "n": 100},
    weighting = {"function": "exp", "r_cut": 10, "threshold": 1e-3},
    sparse=False,
    dtype="float32",
    normalization="l2",
)


for idx, filename in enumerate(filelist):
    atoms = read(path + filename)
    cm = mbtr.create(atoms)
    print(cm)
    cm = np.float32(cm)
    df[idx, :] = cm

def energy_comparison(molconf1, molconf2):
    
    enline1 = linecache.getline(path + molconf1, 2)
    enline2 = linecache.getline(path + molconf2, 2)
    en1 = float(enline1[5:].rstrip())
    en2 = float(enline2[5:].rstrip())
    
    if en1 < en2: return 1., 0.
    else: return 0., 1.


pairsdf = np.zeros((nmols * nperms, 2*descriptor_length))
permfilelist = []

## This takes about a minute, maybe rewrite with np arrays?
## Create 0 array to avoid resizing the array every time vstack is called
for mol in range(0, nconfs * nmols, nconfs):
    perm_count = 0
    for conf_i in range(nconfs):
        for conf_j in range(conf_i +1, nconfs):
            confi = df[mol + conf_i,:]
            confj = df[mol + conf_j,:]
            
            pair1 = np.concatenate([confi, confj])
            pair2 = np.concatenate([confj, confi])
        
            idx = int((mol / nconfs) * nperms)
            
            molconf1 = filelist[mol + conf_i]
            molconf2 = filelist[mol + conf_j]
            label1, label2 = energy_comparison(molconf1, molconf2)
            col_title1 = molconf1.replace("_MMFF94.xyz", "") + molconf2.replace("_MMFF94.xyz", "")[-1]
            permfilelist.append(col_title1)
            col_title2 = molconf2.replace("_MMFF94.xyz", "") + molconf1.replace("_MMFF94.xyz", "")[-1]
            permfilelist.append(col_title2)
            
            pairsdf[idx + perm_count,:] = pair1
            labelsdf[idx + perm_count] = label1
            perm_count += 1
            pairsdf[idx + perm_count, :] = pair2
            labelsdf[idx + perm_count] = label2
            perm_count += 1
            

print()
print("made arrays, loading into pandas .....")
# suprisingly xarray had the same speed as pandas, with the latter being more readable

# xarr_pairs = xr.DataArray(data=pairsdf, dims=['rows', 'cols'], coords=[labelsdf.columns, np.arange(pairsdf.shape[1])])

panda_pairs = pd.DataFrame(pairsdf, index=permfilelist)
panda_labels = pd.DataFrame(labelsdf, index=permfilelist)

panda_pairs.to_pickle('g2_pairs.pkl')
panda_labels.to_pickle('g2_labels.pkl')


print('==========================')
print(panda_pairs)
print(panda_labels)

