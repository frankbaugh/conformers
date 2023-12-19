from benchmarking import bubblesort
import pickle, random, torch
from graphutils import angulargraphpairer
from limdihedral import Net
from scipy.stats import kendalltau
import pandas as pd
from tqdm import tqdm

def model_call(conf_i, conf_j):
    mol = conf_i[:-2]
    if mol[-1] == "_": mol = mol[:-1]
    
    graph_i, graph_j = alldict[mol][conf_i]['graph'], alldict[mol][conf_j]['graph']
    pair = angulargraphpairer(graph_i, graph_j)
    pair.num_graphs = 1
    en1, en2 = float(alldict[mol][conf_i]['FFXML_en']), float(alldict[mol][conf_j]['FFXML_en'])
    pair.forcepair = torch.tensor([en1 - en2])
    
    
    return model(pair)
    
def forcefield_order(conf_i, conf_j):
    mol = conf_i[:-2]
    if mol[-1] == "_": mol = mol[:-1]
    en1, en2 = float(alldict[mol][conf_i]['FFXML_en']), float(alldict[mol][conf_j]['FFXML_en'])
    if en1 < en2: return -1
    else: return 1

def ccsd_order(conf_i, conf_j):
    mol = conf_i[:-2]
    if mol[-1] == "_": mol = mol[:-1]
    en1, en2 = float(alldict[mol][conf_i]['CCSD_en']), float(alldict[mol][conf_j]['CCSD_en'])
    if en1 < en2: return -1
    else: return 1




with open('limdict.pickle', 'rb') as handle:
    alldict = pickle.load(handle)

random.seed(42)
sampled_fraction = 0.6 # All graphs take 4 minutes to pair, random sample
keys_list = list(alldict.keys())
sampled_keys = keys_list[:int(len(keys_list) * sampled_fraction)]
sampled_dict = {key: alldict[key] for key in sampled_keys}



sorted_ccsd = {}
sorted_model = {}
sorted_FFXML = {}
tau_model = {}
tau_baseline = {}

model = Net()
model.load_state_dict(torch.load('lim_dihedral.x.0.6.16.2.pt', map_location=torch.device('cpu')))
model.eval()

test_keys = sampled_keys[int(0.8 * len(sampled_keys)):]
test_dict = {key: sampled_dict[key] for key in test_keys}

for name, moldict in tqdm(test_dict.items()):
    confnames = [confname for confname in moldict.keys() if confname != 'nconfs']
    sorted_model[name] = bubblesort(confnames, model_call)
    sorted_FFXML[name] = bubblesort(confnames, forcefield_order)
    sorted_ccsd[name] = bubblesort(confnames, ccsd_order)
    tau_model[name] = kendalltau(sorted_ccsd[name], sorted_model[name], nan_policy='raise').statistic
    tau_baseline[name] = kendalltau(sorted_ccsd[name], sorted_FFXML[name], nan_policy='raise').statistic

df = pd.DataFrame({'Dihedral Model': tau_model, 'FFXML Baseline': tau_baseline}, index = list(tau_model.keys()))


with open('lim_dihedral_x0.6.16.2.pkl', 'wb') as f:
    pickle.dump(df, f)