import numpy as np
import os
import ase
import torch
import graphite
from graphite.nn.models.alignn import Encoder, Processor, ALIGNN
from graphite.utils.alignn import line_graph, dihedral_graph, get_bnd_angs, get_dih_angs
from graphite.data import AngularGraphPairData
from datautils import get_filelist, get_conflist, get_mollist, get_energies, get_abs_MMFF94, get_path
from graphutils import name2graph, angulargraphpairer
from scipy.stats import kendalltau
from dihedral import Net
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import Data


def bubblesort(confs, model):
    
    conformers = list(confs)
    n = len(conformers)
    for i in range(n - 1):
        swapped = False
        for j in range(0, n - i - 1):
            
            if model(conformers[j], conformers[j + 1]) > 0:
                swapped = True
                conformers[j], conformers[j + 1] = conformers[j + 1], conformers[j]
            
        if not swapped:
            break
    return conformers

# also get the order given my the MMFF94
# Adjust model to handle batch size = 1
            
def CCSD_order(conf_i, conf_j):
    # test model: correct values
    en_i, en_j = CCSD_energies[conf_i], CCSD_energies[conf_j]
    if en_i < en_j:
        return -1
    else: 
        return 1





def model_call(conf_i, conf_j):
    graph_i, graph_j = graphs[conf_i], graphs[conf_j]
    graph_pair = angulargraphpairer(graph_i, graph_j)
    graph_pair.num_graphs = 1
    MMFF_pair = torch.tensor([MMFF94_energies[conf_i], MMFF94_energies[conf_j]])
    graph_pair.forcepair = MMFF_pair
    
    pred = model(graph_pair)
    return pred

def forcefield_order(conf_i, conf_j):
    
    en_i, en_j = MMFF94_energies[conf_i], MMFF94_energies[conf_j]
    if en_i < en_j: return -1
    else: return 1



def run_benchmark(saved_model):
    
    path = get_path()
    filelist, conflist, mollist = get_filelist(path), get_conflist(path), get_mollist(path)


    CCSD_energies, MMFF94_energies = get_energies(path)
    MMFF94_abs = get_abs_MMFF94(path)
    
    model = Net()
    model.load_state_dict(torch.load(saved_model, map_location=torch.device('cpu')))
    model.eval()
    
    
    sorted_true = {}
    sorted_model = {}
    sorted_MMFF94 = {}
    tau_model = {}
    tau_baseline = {}
    graphs = {}

        
    # for each molecule, create two rankings: the true ranking and the model ranking, then
    # measure kendall tau coefficient. Make more robust for unequal number of conformers....
    print('Calculating Kendall Tau Correlations ................')
    for mol in tqdm(mollist):
        confs = [f"{mol}_{i}" for i in range(1, 9)] #creates the conformer names: 18crown6_1, 18crown6_2 etc...
        graphs = {f'{conf}': name2graph(conf) for conf in confs}
        sorted_model[mol] = bubblesort(confs, model_call)
        sorted_MMFF94[mol] = bubblesort(confs, forcefield_order)
        sorted_true[mol] = bubblesort(confs, CCSD_order)
        tau_model[mol] = kendalltau(sorted_true[mol], sorted_model[mol], nan_policy='raise').statistic
        tau_baseline[mol] = kendalltau(sorted_true[mol], sorted_MMFF94[mol], nan_policy='raise').statistic
    
    
    df = pd.DataFrame({'Dihedral Model': tau_model, 'MMFF94 Baseline': tau_baseline}, index = list(tau_model.keys()))

    print(df)

# run_benchmark('diheral_trained_rel.pt')