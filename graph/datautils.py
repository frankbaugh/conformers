import os, ase, torch
import numpy as np
import subprocess
import pandas as pd
from tqdm import tqdm
import rdkit
from rdkit import Chem
from ase import Atoms
from graphutils import atoms2pygdata
import torchani

def get_filelist(path):

    filelist = []

    for filename in os.listdir(path):
        if filename.endswith('.xyz'):
            filelist.append(filename)
   
    return sorted(filelist)
    
def get_conflist(path):
    filelist = get_filelist(path)
    return [file.replace("_MMFF94.xyz", "") for file in filelist]

def get_mollist(path):
    conflist = get_conflist(path)
    return sorted(list(set([conf[:-2] for conf in conflist])))

def shuffle_eights(data_list):
    nconfs = 8
    data_list_grouped = []
    for i in range(0, len(data_list), nconfs):
        data_list_grouped.append(data_list[i:i+nconfs])
    print(len(data_list_grouped))
    
    np.random.seed(33)
    np.random.shuffle(data_list_grouped)
    
    data_list_shuffled = sum(data_list_grouped, [])
    
    return data_list_shuffled

def get_path():
    pwdout = os.getcwd()
    
    if 'fmvb21' in pwdout:
        path = '/home/fmvb21/MMFF94/'
    elif 'frank' in pwdout:
        path = '/Users/frank/code/conf/37Conf8/MMFF94/'
    
    return path

def get_energies(path):

    filename = '37Conf8_data.xlsx'

    df = pd.read_excel(path + filename, sheet_name='Rel_Energy_OPT', header=2)
    df = df.drop(df.index[-3:], axis=0)

    CCSD_energies = {}
    MMFF94_energies = {}

    for index, row in df.iterrows():
        conf = str(row.iloc[0]).strip() + '_' + str((row.iloc[1]))[0]
        CCSD_energy , MMFF94_energy = row.loc['DLPNO-CCSD(T)/TZ'], row.loc['MMFF94']
        CCSD_energies[conf], MMFF94_energies[conf] = CCSD_energy, MMFF94_energy
    
    return CCSD_energies, MMFF94_energies

def get_abs_MMFF94(path):
    filepath = os.path.dirname(path) + '/37Conf8_data.xlsx'

    df = pd.read_excel(filepath, sheet_name='Abs_Energy_OPT', header=2)
    MMFF94_abs = {}
    for index, row in df.iterrows():
        conf = str(row.iloc[0]).strip() + '_' + str((row.iloc[1]))[0]
        MMFF94_energy =row.loc['MMFF94, kcal/mol']
        MMFF94_abs[conf] = MMFF94_energy
    
    return MMFF94_abs

def mol2atoms(mol):
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atoms = Atoms(symbols=symbols, positions=positions)
    return atoms

def get_alldict(path):
    
    inf = open(path + 'lim_data/opt_openff-1.1.0.sdf','rb')
    all_atoms = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"] # Also test

    alldict = {}
    confcount = 1
    with Chem.ForwardSDMolSupplier(inf, removeHs=False) as fsuppl:
        for mol in tqdm(fsuppl):
            if mol is None: continue
            confdict = {}
            try:
                ase_atoms = mol2atoms(mol)
                confdict['graph'] = atoms2pygdata(ase_atoms, all_atoms)
            except Exception as e:
                print(f"Error forming graph of {name}")  
                continue #Exclude the ~ 6 molecules which for some reason dont work
            
            name = mol.GetProp('_Name')
            if name not in alldict: # New molecule, reset moldict and confcount
                moldict = {}
                confcount = 1
            
            confname = name + '_' + str(confcount)
            
            confdict['CCSD_en'] = mol.GetProp('Energy QCArchive')
            confdict['FFXML_en'] = mol.GetProp('Energy FFXML')
    
            moldict[confname] = confdict
            moldict['nconfs'] = confcount
            
            confcount += 1
            alldict[name] = moldict

                
    
    return alldict

def get_anidict(path):
    
    inf = open(path + 'lim_data/opt_openff-1.1.0.sdf','rb')
    all_atoms = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"] # Also test
    calculator = torchani.models.ANI2x().ase()
    

    alldict = {}
    confcount = 1
    with Chem.ForwardSDMolSupplier(inf, removeHs=False) as fsuppl:
        for mol in tqdm(fsuppl):
            if mol is None: continue
            
            confdict = {}
            try:
                ase_atoms = mol2atoms(mol)
                ase_atoms.set_calculator(calculator)
                graph = atoms2pygdata(ase_atoms, all_atoms)
                
                confdict['atoms'] = ase_atoms
                
            except Exception as e:
                print(f"Error forming graph of {name}")  
                continue #Exclude the ~ 6 molecules which for some reason dont work
            
            name = mol.GetProp('_Name')
            if name not in alldict: # New molecule, reset moldict and confcount
                moldict = {}
                confcount = 1
            
            confname = name + '_' + str(confcount)
            
            if all(symbol != 'Br' and symbol != 'I' and symbol != 'P' for symbol in ase_atoms.get_chemical_symbols()):
                confdict['ani_en'] = ase_atoms.get_potential_energy()
            

            moldict[confname] = confdict
            
            confcount += 1
            alldict[name] = moldict


    
    return alldict


    