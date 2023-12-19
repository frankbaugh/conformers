import numpy as np
from ase.io import read
from dscribe.descriptors import CoulombMatrix
import os
import pandas as pd
import re
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

path = '/Users/frank/code/conf/37Conf8/MMFF94/'

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
df = pd.DataFrame()
labelsdf = pd.DataFrame()


for filename in filelist:
    atoms = read(path + filename)
    cm = CoulombMatrix(n_atoms_max=maxN)
    cm = cm.create(atoms)
    cm = np.float32(cm)
    col_title = filename.replace("_MMFF94.xyz", "")
    cm = pd.Series(cm, name=col_title)
    df = pd.concat([df, cm], axis=1)
    
def energy_comparison(molconf1, molconf2):
    ## 1 < 2 => +1
    ## 1 > 2 => 0
    enline1 = linecache.getline(path + molconf1 + "_MMFF94.xyz", 2)
    enline2 = linecache.getline(path + molconf2 + "_MMFF94.xyz", 2)
    en1 = float(enline1[5:].rstrip())
    en2 = float(enline2[5:].rstrip())
    
    if en1 < en2: return 1
    else: return 0

pairsdf = pd.DataFrame()

## This takes about a minute, maybe rewrite with np arrays?

#Iterate over the molecules by taking steps of nconfs
for mol in range(0, nconfs * nmols, nconfs):
    
    # i<j loop, pairs up conformers, concatenate downwards, then add as new column into pairsdf
    # Also creates the labels for each pair to keep same ordering
    for conf_i in range(nconfs):
        for conf_j in range(conf_i +1, nconfs):
            confi = df.iloc[:,mol + conf_i]
            confj = df.iloc[:,mol + conf_j]
            
            pair1 = pd.concat([confi, confj], axis=0)
            pair2 = pd.concat([confj, confi], axis=0)
        
            pairsdf = pd.concat([pairsdf, pair1], axis=1)
            pairsdf = pd.concat([pairsdf, pair2], axis=1)
            
            molconf1 = df.columns[mol + conf_i]
            molconf2 = df.columns[mol + conf_j]
            
            label1 = energy_comparison(molconf1, molconf2)
            label2 = energy_comparison(molconf2, molconf1)
            labelsdf = pd.concat([labelsdf, pd.Series([label1], dtype='float32')], axis=1)
            labelsdf = pd.concat([labelsdf, pd.Series([label2], dtype='float32')], axis=1)



pairsdf.to_pickle('pairsdf.pkl')
labelsdf.to_pickle('labelsdf.pkl')

print(labelsdf)
