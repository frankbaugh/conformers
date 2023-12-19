import rdkit
from rdkit import Chem
import pubchempy
from ase import Atoms
from graphutils import atoms2pygdata
from tqdm import tqdm

inf = open('/Users/frank/Code/conf/37Conf8/lim_data/opt_openff-1.2.0.sdf','rb')

alldict = {}
confcount = 1

def mol2atoms(mol):
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atoms = Atoms(symbols=symbols, positions=positions)
    
    return atoms


with Chem.ForwardSDMolSupplier(inf, removeHs = False) as fsuppl:
    for mol in tqdm(fsuppl):
        if mol is None: continue
        
        name = mol.GetProp('_Name')
        if name not in alldict: # New molecule, reset moldict and confcount
            moldict = {}
            confcount = 1
        
        confname = name + '_' + str(confcount)
        
        confdict = {}
        confdict['graph'] = mol2atoms(mol)
        confdict['CCSD_en'] = mol.GetProp('Energy QCArchive')
        confdict['FFXML_en'] = mol.GetProp('Energy FFXML')
   
        moldict[confname] = confdict
        moldict['nconfs'] = confcount
        
        confcount += 1
        alldict[name] = moldict
        
all_atoms = ["H", "C", "N", "O", "F", "P", "S", "Cl"]
atomlist = []
            
for name, moldict in alldict.items():
    
    for confname, confdict in moldict.items():
        if confname == 'nconfs':
            continue
    
        atoms = confdict['graph']
        #print(atoms)
        atomlist.append(atoms.get_chemical_symbols())

Icount, Brcount, Ccount = 0, 0, 0
for symbollist in atomlist:
    if "I" in symbollist:
        Icount += 1
    if "Br" in symbollist:
        Brcount += 1
    if "C" in symbollist:
        Ccount += 1

print(len(atomlist))
print(f"I count: {Icount}")
print(f"Brcount: {Brcount}")
print(f"Ccount: {Ccount}")

