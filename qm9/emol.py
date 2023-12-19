import os, pickle
import pandas as pd
import ase
from ase.io import sdf
from rdkit import Chem
from tqdm import tqdm
import sys
sys.path.append('../graph')
import datautils
import graphutils
import torchani
sys.path.append('./obcalc')
import openbabel
from obcalc import OBForceField

def get_data_below_line(lines, search_string):
    
    for i, line in enumerate(lines):
        if search_string in line:
            line_idx = i + 1
    
    return lines[line_idx].strip()


def get_alldict(sampled_filelist):
    
    all_atoms = ["H", "C", "N", "O", "F"]
    alldict = {}
    
    print('Generating Graphs .......')
    for file in tqdm(sampled_filelist):
        
        confdict = {}
        
        with open(file, 'r') as afile:
            lines = afile.readlines()
        
        mol_id = get_data_below_line(lines, 'molecule_id')
        conf_id = get_data_below_line(lines, 'conformer_id')
        confdict['qm_en'] = float(get_data_below_line(lines, 'energy_abs'))
        
        try:
            ase_atoms = sdf.read_sdf(file)
            confdict['graph'] = graphutils.atoms2pygdata(ase_atoms, all_atoms)
        except Exception as e:
            print(f"Error forming graph of ID:  {mol_id}")  
            continue #Exclude the ~ 6 molecules which for some reason dont work
        
        if mol_id not in alldict: # New molecule, reset moldict and confcount
            alldict[mol_id] = {}
        

        alldict[mol_id][conf_id] = confdict


    return alldict

def get_emolanidict(sampled_filelist):
    
    all_atoms = ["H", "C", "N", "O", "F"]
    emol_anidict = {}
    calculator = OBForceField(force_field='MMFF94', bonds=None)
    
    print('Calculating ANI energies .......')
    for file in tqdm(sampled_filelist):
        
        confdict = {}
        
        with open(file, 'r') as afile:
            lines = afile.readlines()
        
        mol_id = get_data_below_line(lines, 'molecule_id')
        conf_id = get_data_below_line(lines, 'conformer_id')
        
        if mol_id == '4413': continue

        
        try:
            ase_atoms = sdf.read_sdf(file)
            ase_atoms.set_calculator(calculator)
            #confdict['graph'] = graphutils.atoms2pygdata(ase_atoms, all_atoms)
            confdict['ff_en'] = ase_atoms.get_potential_energy()
            
        except Exception as e:
            print(f"Error calculating ANI energy ID:  {mol_id}")
            continue 
        
        if mol_id not in emol_anidict: # New molecule, reset moldict and confcount
            emol_anidict[mol_id] = {}
        

        emol_anidict[mol_id][conf_id] = confdict
    
    return emol_anidict
    
def get_emolforcedict(sampled_fileslist):
    
    all_atoms = ["H", "C", "N", "O", "F"]
    emol_anidict = {}
    calculator = torchani.models.ANI2x().ase()
    
    print('Calculating ANI energies .......')
    for file in tqdm(sampled_filelist):
        
        confdict = {}
        
        with open(file, 'r') as afile:
            lines = afile.readlines()
        
        mol_id = get_data_below_line(lines, 'molecule_id')
        conf_id = get_data_below_line(lines, 'conformer_id')
        
        if mol_id == '4413': continue

        
        try:
            ase_atoms = sdf.read_sdf(file)
            ase_atoms.set_calculator(calculator)
            #confdict['graph'] = graphutils.atoms2pygdata(ase_atoms, all_atoms)
            confdict['ani_en'] = ase_atoms.get_potential_energy()
            
        except Exception as e:
            print(f"Error calculating ANI energy ID:  {mol_id}")
            continue 
        
        if mol_id not in emol_anidict: # New molecule, reset moldict and confcount
            emol_anidict[mol_id] = {}
        

        emol_anidict[mol_id][conf_id] = confdict
    
    return emol_anidict
    



def get_filelist(path):

    filelist = []

    for filename in os.listdir(path):
        if filename.endswith('.sdf'): #Only get the FF optimized geomerties
            filelist.append(path + filename)
   
    return sorted(filelist)

path = '/Users/frank/code/conf/datasets/eMol9_dataset/eMol9_data/'
filelist = get_filelist(path)
frac = 1
sampled_filelist = [file for file in filelist[:int(frac * len(filelist))] if file.endswith('mmff.sdf')]

#alldict = get_alldict(sampled_filelist)

emol_ffdict = get_emolanidict(sampled_filelist)

pickle.dump(emol_ffdict, open('emol_FFdict.pkl', 'wb'))


