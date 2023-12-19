import pickle, random, torch, random
from scipy.stats import kendalltau
import pandas as pd
from tqdm import tqdm
import torch_geometric
from torch_geometric.loader import DataLoader
import sys
sys.path.append('../graph')
from emol_forcesplit import Net

def partition(name, conflist, low, high, model):
    pivot = conflist[high]
    i = low - 1
    for j in range(low, high):
        if model(conflist[j], pivot, name) >= 0:
            i += 1
            conflist[i], conflist[j] = conflist[j], conflist[i]
    conflist[i + 1], conflist[high] = conflist[high], conflist[i + 1]
    return i + 1

# Quicksort function
def quicksort(name, confs, low, high, model):
    conflist = confs.copy()
    if low < high:
        pivot_index = partition(name, conflist, low, high, model)
        quicksort(name, conflist, low, pivot_index - 1, model)
        quicksort(name, conflist, pivot_index + 1, high, model)
    return conflist

def model_call(conf_i, conf_j, name):
    graph_i, graph_j = alldict[name][conf_i]['graph'], alldict[name][conf_j]['graph']
    force_i, force_j = float(anidict[name][conf_i]['ani_en']), float(anidict[name][conf_j]['ani_en'])
    force = torch.tensor([force_i - force_j])
    graph_i.num_graphs = 1
    graph_j.num_graphs = 1
    
    return model(graph_i, graph_j, force)
    


def forcefield_order(conf_i, conf_j, name):

    en1, en2 = float(alldict[name][conf_i]['FFXML_en']), float(alldict[name][conf_j]['FFXML_en'])
    if en1 < en2: return -1
    else: return 1

def ccsd_order(conf_i, conf_j, name):

    en1, en2 = float(alldict[name][conf_i]['qm_en']), float(alldict[name][conf_j]['qm_en'])
    if en1 < en2: return -1
    else: return 1

def ani_order(conf_i, conf_j, name): # ANI only has energies for molecules containing H, C, N, O, F, Cl, and S
    en1, en2 = float(anidict[name][conf_i]['ani_en']), float(anidict[name][conf_j]['ani_en'])
    if en1 < en2: return -1
    else: return 1

with open('emol_1.0.pkl', 'rb') as handle:
    alldict = pickle.load(handle)

with open('emol_anidict.pkl', 'rb') as handle:
    anidict = pickle.load(handle)

model = Net()
model.load_state_dict(torch.load('./trained_models/emol_forcesplit.pt', map_location=torch.device('cpu')))
model.eval()


keys_list = list(alldict.keys())
test_keys = keys_list[int(0.8 * len(keys_list)):]
test_keys = test_keys[:int(0.5 * len(test_keys))] # Reduce size of test data set

test_dict = {key: alldict[key] for key in test_keys}
test_ani = {key: anidict[key] for key in test_keys}

sorted_ccsd = {}
sorted_model = {}
sorted_FFXML = {}
sorted_ani = {}
tau_model = {}
tau_baseline = {}
tau_ani = {}

for name, moldict in tqdm(test_dict.items()):
    
    confnames = [confname for confname in moldict.keys() if confname != 'nconfs']
    if len(confnames) == 1:
        continue
    high = len(confnames) - 1

    sorted_model[name] = quicksort(name, confnames, 0, high, model_call)
    #sorted_FFXML[name] = quicksort(name, confnames, 0, high, forcefield_order)
    sorted_ccsd[name] = quicksort(name, confnames, 0, high, ccsd_order)
    sorted_ani[name] = quicksort(name, confnames, 0, high, ani_order)
    tau_model[name] = kendalltau(sorted_ccsd[name], sorted_model[name], nan_policy='raise').statistic
    #tau_baseline[name] = kendalltau(sorted_ccsd[name], sorted_FFXML[name], nan_policy='raise').statistic
    tau_ani[name] = kendalltau(sorted_ccsd[name], sorted_ani[name], nan_policy='raise').statistic

df = pd.DataFrame({'Model': tau_model, 'ANI': tau_ani}, index = list(tau_model.keys()))


with open('./benched_models/emol_forcesplit.pkl', 'wb') as f:
    pickle.dump(df, f)

diff_col = df['Model'] - df['ANI']
df['Difference'] = diff_col

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)
    
score = (df['Difference'] >= 0).mean()
modelavg = df['Model'].mean()
aniavg = df['ANI'].mean()

print('================================================================')
print(f"Model Average: {modelavg}")
print(f"ANI Average: {aniavg}")
print(f"Score [chance of improvement] {score}")