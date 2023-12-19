import datautils
import graphutils
from dihedral import set_gpu
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
        inner_size = 32
        self.alignnd_left = ALIGNN(
        encoder   = Encoder(num_species=5, cutoff=4.0, dim=inner_size, dihedral=True),
        processor = Processor(num_convs=3, dim=inner_size),
        decoder   = Decoder(node_dim=inner_size, out_dim=1),
        )
        self.alignnd_right = ALIGNN(
        encoder   = Encoder(num_species=5, cutoff=4.0, dim=inner_size, dihedral=True),
        processor = Processor(num_convs=3, dim=inner_size),
        decoder   = Decoder(node_dim=inner_size, out_dim=1),
        )
        
        
        self.l1 = nn.Linear(inner_size * 2, inner_size)
        self.l2 = nn.Linear(inner_size, 1)
    
    def forward(self, left, right):

        x_left = self.alignnd_left(left)
        x_right = self.alignnd_right(right)
        
        x = torch.cat((x_left, x_right), dim=-1)

        x = leaky_relu(self.l1(x))
        x = self.l2(x)

        return x



def get_paired_data(alldict, frac):
    
    left_list, right_list, y_list = [], [], []
    sampled_fraction = frac # Sample the dict if too slow to pair
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
                
                left_list.append(confdict_i['graph'])
                right_list.append(confdict_j['graph'])
                
                ccsd_i, ccsd_j = float(confdict_i['qm_en']), float(confdict_j['qm_en'])
                y_list.append(torch.tensor([ccsd_i - ccsd_j]))   
    
    return left_list, right_list, y_list
                
def batcher(left_list, right_list, y_list, batch_size):
    test_idx = int(0.8 * len(y_list))
    train_left, train_right, train_y = left_list[:test_idx], right_list[:test_idx], y_list[:test_idx]
    test_left, test_right, test_y = left_list[test_idx:], right_list[test_idx:], y_list[test_idx:]
    
    left_train = DataLoader(train_left, batch_size=batch_size, shuffle=False, follow_batch=['x_atm', 'x_bnd', 'x_ang'])
    right_train = DataLoader(train_right, batch_size=batch_size, shuffle=False, follow_batch=['x_atm', 'x_bnd', 'x_ang'])
    y_train = DataLoader(train_y, batch_size=batch_size, shuffle=False)
    
    
    left_test = DataLoader(test_left, batch_size=batch_size, shuffle=False, follow_batch=['x_atm', 'x_bnd', 'x_ang'])
    right_test = DataLoader(test_right, batch_size=batch_size, shuffle=False, follow_batch=['x_atm', 'x_bnd', 'x_ang'])
    y_test = DataLoader(test_y, batch_size=batch_size, shuffle=False)

    
    
    return left_train, right_train, y_train, left_test, right_test, y_test
    
    
          
def train_loop(left_batched, right_batched, y_batched, model, loss_fn, optimizer, batch_size, device):
    model.train()
    
    for left, right, y in zip(left_batched, right_batched, y_batched):
        left, right, y = left.to(device), right.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(left, right)

        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
    return loss

def test_loop(left_batched, right_batched, y_batched, model, loss_fn, batch_size, test_loss_list, device):
    model.eval()
    test_loss = 0
    
    for left, right, y in zip(left_batched, right_batched, y_batched):
        left, right, y = left.to(device), right.to(device), y.to(device)
        out = model(left, right)
        test_loss += loss_fn(out, y).item()

    test_loss /= len(y_batched)
    test_loss_list.append(test_loss)


def run_epochs(epochs, learning_rate, batch_size, name, frac):
    print("running split dihedral on eMol Dataset: ")
    
    
    device = set_gpu()
    path = '/home/fmvb21/'

    with open(path + 'emol_1.0.pkl', 'rb') as handle:
        alldict = pickle.load(handle)
    
    left_list, right_list, y_list = get_paired_data(alldict, frac)
    left_train, right_train, y_train, left_test, right_test, y_test = batcher(left_list, right_list, y_list, batch_size)
 

    
    model = Net().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.5e-2) # 0.00001 lr
    
    test_loss_list = []
    
    for t in range(epochs):
        # print(f"Epoch {t+1}\n------------------------")
        train_loop(left_train, right_train, y_train, model, loss_fn, optimizer, batch_size, device)
        test_loop(left_test, right_test, y_test, model, loss_fn, batch_size, test_loss_list, device)
        if (t % 1 == 0): plot(test_loss_list)
        
    
    torch.save(model.state_dict(), name)
    
    print(f'Min loss: {min(test_loss_list)}')
    print("Done!")