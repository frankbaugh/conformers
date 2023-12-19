import datautils
import graphutils
from dihedral import split_data, set_gpu
import torch
from uniplot import plot
from torch_geometric.loader import DataLoader
from torch import nn
import pickle
from tqdm import tqdm
import random
from torch_geometric.utils import scatter
import graphite
from graphite.nn.models.alignn import Encoder, Processor, ALIGNN
from torch.nn.functional import leaky_relu

class Decoder(nn.Module):
    def __init__(self, node_dim, out_dim):
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        
    def forward(self, data):
        # Globally pool all atom/bond/angle components into a single feature vector
        # This operation assumes all components have the same dimensionality, 
        # which should be the case for ALIGNN
        if data.num_graphs != 1: #if data is batched
            h_atm_pooled = scatter(data.h_atm, data.x_atm_batch, dim=0, reduce='sum')
            #h_bnd_pooled = scatter(data.h_bnd, data.x_bnd_batch, dim=0, reduce='sum')
            #h_ang_pooled = scatter(data.h_ang, data.x_ang_batch, dim=0, reduce='sum')
            h_pooled = h_atm_pooled
        else: # for single graphs, only need to sum along one axis
            h_pooled = data.h_atm.sum(dim = 0)
        return h_pooled


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        inner_size = 16
        self.alignnd = ALIGNN(
        encoder   = Encoder(num_species=10, cutoff=3.0, dim=inner_size, dihedral=True),
        processor = Processor(num_convs=2, dim=inner_size),
        decoder   = Decoder(node_dim=inner_size, out_dim=1),
        )
        self.l1 = nn.Linear(inner_size + 1, inner_size)
        self.l2 = nn.Linear(inner_size, 1)
    
    def forward(self, data):

        #force = data.forcepair.unsqueeze(-1)
        force = data.forcepair

        x = self.alignnd(data)
        x = torch.cat((x, force), dim=-1)

        x = leaky_relu(self.l1(x))
        x = self.l2(x)

        return x



def get_paired_data(path, alldict, device):
    
    paired_data_list = []
    random.seed(42)
    sampled_fraction = 0.6 # All graphs take 4 minutes to pair, random sample. 0.6 is maximim for loading onto GPU
    keys_list = list(alldict.keys())
    sampled_keys = keys_list[:int(len(keys_list) * sampled_fraction)]
    sampled_dict = {key: alldict[key] for key in sampled_keys}
    
    print('Pairing graphs .....') 
    for name, moldict in tqdm(sampled_dict.items()):
        for confname_i, confdict_i in moldict.items():
            for confname_j, confdict_j in moldict.items():
                if confname_i == 'nconfs' or confname_j == 'nconfs':
                    continue
                if confname_i == confname_j:
                    continue
                
                pair1 = graphutils.angulargraphpairer(confdict_i['graph'], confdict_j['graph'])
                pair2 = graphutils.angulargraphpairer(confdict_j['graph'], confdict_i['graph'])
                
                en_i, en_j = float(confdict_i['FFXML_en']), float(confdict_j['FFXML_en'])
                pair1.forcepair = torch.tensor([en_i - en_j])
                pair2.forcepair = torch.tensor([en_j - en_i])
                
                ccsd_i, ccsd_j = float(confdict_i['CCSD_en']), float(confdict_j['CCSD_en'])
                pair1.y = torch.tensor([ccsd_i - ccsd_j])
                pair2.y = torch.tensor([ccsd_j - ccsd_i])
                
                paired_data_list.append(pair1)
                paired_data_list.append(pair2)
    
    return paired_data_list
                
def split_data(paired_data_list, batch_size):
    
    test_idx = int(0.8 * len(paired_data_list))
    
    train_data = paired_data_list[:test_idx]
    test_data = paired_data_list[test_idx:]
    
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, follow_batch=['x_atm', 'x_bnd', 'x_ang'])
    test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True, follow_batch=['x_atm', 'x_bnd', 'x_ang'])
    
    return train_data, test_data
          
def train_loop(train_data, model, loss_fn, optimizer, batch_size, device):
    model.train()
    
    for data in train_data:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        if batch_size != 1:
            out = torch.squeeze(out)

        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
    return loss

def test_loop(test_data, model, loss_fn, batch_size, test_loss_list, device):
    model.eval()
    test_loss = 0
    
    for data in test_data:
        data = data.to(device)
        out = model(data)
        if batch_size != 1:
            out = torch.squeeze(out)
        test_loss += loss_fn(out, data.y).item()

    test_loss /= len(test_data)
    test_loss_list.append(test_loss)



def run_epochs(epochs, learning_rate, batch_size):
    print("running dihedral")
    
    
    device = set_gpu()
    path = '/Users/frank/code/conf/37Conf8/'
    path = '/home/fmvb21/'

    with open(path + 'limdict.pickle', 'rb') as handle:
        alldict = pickle.load(handle)
    
    paired_data_list = get_paired_data(path, alldict, device)
    train_data, test_data = split_data(paired_data_list, batch_size)
 

    
    model = Net().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.5e-2) # 0.00001 lr
    
    test_loss_list = []
    
    for t in range(epochs):
        # print(f"Epoch {t+1}\n------------------------")
        train_loop(train_data, model, loss_fn, optimizer, batch_size, device)
        test_loop(test_data, model, loss_fn, batch_size, test_loss_list, device)
        if (t % 1 == 0): plot(test_loss_list)
        
    
    torch.save(model.state_dict(), 'lim_dihedral.justang.pt')
    
    print(f'Min loss: {min(test_loss_list)}')
    print("Done!")