""" 
This script will contain the functions which are relevent for bond features 
This is an addaptation of one of Ferdies scripts
"""

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np 

def bondType(bond):
    if bond !=None:
        bondType=str(bond.GetBondType())
        if bondType=='SINGLE':
            bondType=1
        elif bondType=='DOUBLE':
            bondType=2
        elif bondType=='TRIPLE':
            bondType=3
        elif bondType=='AROMATIC':
            bondType=4
        elif bondType=='IONIC':
            bondType=5
    else:
        bondType=0
    return bondType

def bond_Feature(bond):
    bondFeature=[]
    bondFeature.append(bondType(bond))
    return bondFeature