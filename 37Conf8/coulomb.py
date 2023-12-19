import numpy as np
from ase.io import read
from dscribe.descriptors import CoulombMatrix
import os
import pandas as pd
import re


def getmaxN():
    path = '/Users/frank/code/conf/37Conf8/MMFF94/'
    os.system('head -n 1 ' + path + '*.xyz | sort -n | tail -n 1')

## getmaxN()
## >> 65 (4225 elements in matrix)
maxN = 4225


def get_name_list():
    
    name_list = []
    path = '/Users/frank/code/conf/37Conf8/MMFF94/'
    
    for filename in os.listdir(path):
        if filename.endswith('.xyz'): 
            
            filename = filename.replace('_MMFF94.xyz', '')
            name_list.append(filename)
    
    return name_list

name_list = get_name_list()


def write_cm(name_list, df):

    n_confs = 8
    path = '/Users/frank/code/conf/37Conf8/MMFF94/'
            
    for molecule in name_list:
        file_conf = path + molecule + '_MMFF94.xyz'
        print(file_conf)
        if os.path.exists(file_conf):
            atoms = read(file_conf)
            cm = CoulombMatrix(n_atoms_max=65)
            cm = cm.create(atoms, verbose=True)
            #Note: flattened by default
            df.at[molecule[:-2], 'conf' + molecule[-1]] = [cm]

    return df


n_confs = 8

name_set = list(set(name_list))

df = pd.DataFrame(name_set, columns=['Molecule'])


for i in range(1, n_confs + 1):
    col_title = 'conf' + str(i)
    df[col_title] = np.zeros(len(name_set))
    df[col_title] = df[col_title].astype(object)

print(df)
df = write_cm(name_list, df)
    
print(df)
