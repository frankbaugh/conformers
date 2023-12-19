""" 
This not book contains the relevent functions for atom features 
This is an addaptation of one of Ferdies scripts
"""
# In[]
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

from morfeus import Sterimol, SASA, BuriedVolume, Dispersion, XTB, read_xyz, read_geometry


def atomTypeOneHot(atom):
    AtomList=['H','C','N','O','S','Si','P','Cl','Br', 'F']
    oneHot=np.zeros(len(AtomList))
    if atom!=None:
        try:
            oneHot[AtomList.index(atom.GetSymbol())]=1
        except:
            print(atom.GetSymol())
    return oneHot

def atomicNumber(atom):
    if atom != None:
        return atom.GetAtomicNum()
    else:
        return 0
def atomMass(atom):
    if atom != None:
        return atom.GetMass()
    else: 
        return 0
def atomicConectivity(atom):
    ''' This gets the total conectivity of the atom including H's '''
    if atom != None:
        return atom.GetTotalDegree()
    else:
        return 0
def atomValenceImplicit(atom):
    if atom != None:
        return atom.GetImplicitValence()
    else:
        return 0
def atomValenceExplicit(atom):
    if atom != None:
        return atom.GetExplicitValence()
    else: 
        return 0
def atomHybridization(atom):
    Hybridization=['S','SP','SP2','SP3','SP3D','SP3D2']
    oneHot=np.zeros(len(Hybridization))
    if atom != None:
        atomHybrid=str(atom.GetHybridization())
        if atomHybrid=='UNSPECIFIED':
            atomHybrid='S'
        oneHot[Hybridization.index(atomHybrid)]=1
    return oneHot
def atomFormalCharge(atom):
    if atom != None:
        return atom.GetFormalCharge()
    else: 
        return 0
def atomPartialCharge(atom):
    """ the line `Chem.AllChem.ComputeGasteigerCharges(mol)` must have been run for this to work"""
    if atom != None:
        gasterCarge=atom.GetProp('_GasteigerCharge')
        if gasterCarge in ('nan','-nan','-inf','inf'):
            gasterCarge=0
        return float(gasterCarge)
    else:
        return 0
def atomRadialElectronNum(atom):
    if atom != None:
        return atom.GetNumRadicalElectrons()
    else:
        return 0
def atomIsAromatic(atom):
    if atom != None:
        return float(atom.GetIsAromatic())
    else: 
        return 0
def atomIsInRing(atom):
    if atom != None:
        return float(atom.IsInRing())
    else:
        return 0
def atomIsChiral(atom):
    if atom!=None:
        return atom.HasProp('_ChiralityPossible')
    else:
        return 0



def Atom_features(atom):
    atomicFeatures=[]
    atomicFeatures.extend(atomTypeOneHot(atom))
    atomicFeatures.extend([atomMass(atom)])
    atomicFeatures.extend([atomicNumber(atom)])
    atomicFeatures.extend([atomFormalCharge(atom)])
    atomicFeatures.append(atomicConectivity(atom))
    atomicFeatures.append(atomIsAromatic(atom))
    atomicFeatures.append(atomValenceExplicit(atom))
    atomicFeatures.append(atomValenceImplicit(atom))
    atomicFeatures.extend(atomHybridization(atom))
    atomicFeatures.append(atomFormalCharge(atom))
    atomicFeatures.append(atomPartialCharge(atom))
    atomicFeatures.append(atomRadialElectronNum(atom))
    atomicFeatures.append(atomIsAromatic(atom))
    atomicFeatures.append(atomIsInRing(atom))
    atomicFeatures.append(atomIsChiral(atom))

    return atomicFeatures