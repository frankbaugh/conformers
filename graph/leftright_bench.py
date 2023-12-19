
import pickle, random, torch, random
from limforcesplit import Net
from scipy.stats import kendalltau
import pandas as pd
from tqdm import tqdm
import torch_geometric
from torch_geometric.loader import DataLoader


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
    force_i, force_j = float(alldict[name][conf_i]['FFXML_en']), float(alldict[name][conf_j]['FFXML_en'])
    force = torch.tensor([force_i - force_j])
    graph_i.num_graphs = 1
    graph_j.num_graphs = 1
    
    return model(graph_i, graph_j, force)
    
def forcefield_order(conf_i, conf_j, name):

    en1, en2 = float(alldict[name][conf_i]['FFXML_en']), float(alldict[name][conf_j]['FFXML_en'])
    if en1 < en2: return -1
    else: return 1

def ccsd_order(conf_i, conf_j, name):

    en1, en2 = float(alldict[name][conf_i]['CCSD_en']), float(alldict[name][conf_j]['CCSD_en'])
    if en1 < en2: return -1
    else: return 1

def ani_order(conf_i, conf_j, name): # ANI only has energies for molecules containing H, C, N, O, F, Cl, and S
    en1, en2 = float(anidict[name][conf_i]['ani_en']), float(anidict[name][conf_j]['ani_en'])
    if en1 < en2: return -1
    else: return 1



def score_order(confs, name):
    conflict_score = 0
    search_depth = 3
    if search_depth >= len(confs):
        search_depth = len(confs) - 1
    
    for idx, conf in enumerate(confs):
        upper = idx + search_depth
        if upper > (len(confs) - 1): upper = len(confs) - 1
        for conf2 in confs[idx:upper]:
            if model_call(conf, conf2, name) < 0: # IF they are in the wrong oder, increment score
                conflict_score += 1
        
    print(conflict_score)
    return conflict_score

def best_order(name, confnames, model):
    sorted = []
    confs = confnames.copy()
    sorted.append(quicksort(name, confs, 0, len(confnames) -1, model))
    random.seed(12)
    confsshuff = confnames.copy()
    random.shuffle(confsshuff)
    sorted.append(quicksort(name, confsshuff , 0, len(confnames) -1, model))
    random.seed(3)
    confsshuff2 = confnames.copy()
    random.shuffle(confsshuff2)
    sorted.append(quicksort(name, confsshuff2, 0, len(confnames) -1, model))
    
    scores = score_order(sorted[0], name), score_order(sorted[1], name), score_order(sorted[2], name)
    min_idx = scores.index(min(scores))
    
    return sorted[min_idx]
    
    




with open('limdict.pickle', 'rb') as handle:
    alldict = pickle.load(handle)


with open('anidict.pickle', 'rb') as handle:
    anidict = pickle.load(handle)

sampled_fraction = 1 # All graphs take 4 minutes to pair, random sample
keys_list = list(alldict.keys())
sampled_keys = keys_list[:int(len(keys_list) * sampled_fraction)]
sampled_dict = {key: alldict[key] for key in sampled_keys}
sampled_ani = {key: anidict[key] for key in sampled_keys}




sorted_ccsd = {}
sorted_model = {}
sorted_FFXML = {}
sorted_ani = {}
tau_model = {}
tau_baseline = {}
tau_ani = {}

model = Net()
model.load_state_dict(torch.load('./trained_models/limforcesplit_all.pt', map_location=torch.device('cpu')))
model.eval()

test_keys = sampled_keys[int(0.8 * len(sampled_keys)):]
test_dict = {key: sampled_dict[key] for key in test_keys}
test_ani = {key: sampled_ani[key] for key in test_keys}
# Get only those molecules that ANI can handle: recall all_atoms = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
roguecount = 0


for name, moldict in tqdm(test_dict.items()):
    
    confnames = [confname for confname in moldict.keys() if confname != 'nconfs']
    high = len(confnames) - 1
    if 'ani_en' not in test_ani[name][confnames[0]]:
        roguecount += 1
        continue
    
    if moldict['nconfs'] == 1:
        continue


    sorted_model[name] = quicksort(name, confnames, 0, high, model_call)
    sorted_FFXML[name] = quicksort(name, confnames, 0, high, forcefield_order)
    sorted_ccsd[name] = quicksort(name, confnames, 0, high, ccsd_order)
    sorted_ani[name] = quicksort(name, confnames, 0, high, ani_order)
    tau_model[name] = kendalltau(sorted_ccsd[name], sorted_model[name], nan_policy='raise').statistic
    tau_baseline[name] = kendalltau(sorted_ccsd[name], sorted_FFXML[name], nan_policy='raise').statistic
    tau_ani[name] = kendalltau(sorted_ccsd[name], sorted_ani[name], nan_policy='raise').statistic

df = pd.DataFrame({'Dihedral Model': tau_model, 'FFXML Baseline': tau_baseline, 'ANI': tau_ani}, index = list(tau_model.keys()))

print(f"Count of rogue atoms: {roguecount}")

with open('./benched_models/limforcesplit_ani', 'wb') as f:
    pickle.dump(df, f)

diff_col = df['Dihedral Model'] - df['ANI']
df['Difference'] = diff_col

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)
    
score = (df['Difference'] >= 0).mean()

print('================================================================')
print(f"Score vs ANI: {score}")