import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import SchNet
from tqdm import tqdm

def set_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
        print("Using GPU")
    else:
        device = torch.device("cpu")   # Use CPU
        print("Using CPU")
    
    return device
device = set_gpu()

## batches are handled differently, by adding in atoms but not connecting them with edges
## might need to think about this if training too slowly. involving reshape? the batch adds
## extra dimension e.g z -> (batch, conf, (idx = specified), atom)
## so reshape then create the list?

class pairs_Dataset(Dataset):
    def __init__(self, z, pos, labels):
        #Convert to torch tensor then load onto GPU
        self.z = torch.Tensor(z).type(torch.LongTensor).to(device)
        self.pos = torch.Tensor(pos).to(device)
        self.labels = torch.Tensor(labels).to(device)
    
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        
        return self.z[:, idx, :], self.pos[:, idx, :, :], self.labels[idx]


class CombinedNetwork(nn.Module):
    def __init__(self, schnet1, schnet2):
        super().__init__()
        self.layersize = 2
        self.schnet1 = schnet1
        self.schnet2 = schnet2
        self.l1 = nn.Linear(2, self.layersize)
        self.l2 = nn.Linear(self.layersize, 1)
        
    def forward(self, z, pos):
        z, pos = torch.squeeze(z), torch.squeeze(pos)
        y1 = self.schnet1(z[:, 0, ...], pos[:, 0, ...])
        y2 = self.schnet2(z[:, 1, ...], pos[:, 1, ...])
        y = torch.cat((y1, y2), dim=1)
        y = F.relu(self.l1(y))
        y = self.l2(y)
        return y


def get_data():
    
    path = '/Users/frank/code/conf/graph/'
    # path = '/home/fmvb21/graph/'
    z_pairs = np.load(path + 'z_pairs.npy')
    position_pairs = np.load(path + 'position_pairs.npy')
    labels = np.load(path + 'graph_labels.npy')
    
    z_pairs = np.int32(z_pairs)
    nmols = 37
    nconfs = 8
    nperms = nconfs * (nconfs - 1)
    
    
    # Ensure that it doesnt train on all the molecules
    #Get approx 80 % for training, but ensure divisibility so that 
    # No contamination of some test molecule conformers
    
    frac = int(0.8 * nmols) / nmols
    test_idx = int(frac * nmols * nperms)
    train_z = z_pairs[:, :test_idx, :]
    test_z = z_pairs[:, test_idx:, :]
    train_pos = position_pairs[:, :test_idx, :, :]
    test_pos = position_pairs[:, test_idx:, :, :]
    train_labels = labels[:test_idx]
    test_labels = labels[test_idx:]
    
    train_dataset = pairs_Dataset(train_z, train_pos, train_labels)
    test_dataset = pairs_Dataset(test_z, test_pos, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    return train_loader, test_loader


train_loaded, test_loaded = get_data()


def train_loop(dataloader, model, loss_fn, optimiser, train_loss_list):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    
    
    for batch, (z, pos, y) in enumerate(dataloader):

        pred = model(z, pos)
        pred = pred.reshape(1)
        loss = loss_fn(pred, y)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        
        total_loss += loss.item()
        
        if batch % 32 == 0:
            loss = loss.item()
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss_list.append(total_loss/(batch+1))

def test_loop(dataloader, model, loss_fn, test_loss_list):
    
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for z, pos, y in dataloader:
            pred = model(z, pos)
            pred = pred.reshape(1)
            test_loss += loss_fn(pred, y).item()
            
            correct += ((pred > 0.5).squeeze().long() == y.squeeze().long()).sum().item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()


    test_loss /= num_batches
    test_loss_list.append(test_loss)
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss:{test_loss:>8f} \n")


torch.manual_seed(50)
schnet1 = SchNet()
schnet2 = SchNet()


model = CombinedNetwork(schnet1, schnet2)

model = model.to(device)
learning_rate = 1e-3
epochs = 500

loss_fn = nn.BCEWithLogitsLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01)

train_loss_list = []
test_loss_list = []
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------")
    train_loop(train_loaded, model, loss_fn, optimiser, train_loss_list)
    test_loop(test_loaded, model, loss_fn, test_loss_list)

print("Done!")

np.save('test_loss.npy', np.asarray(test_loss_list))
np.save('train_loss.npy', np.asarray(train_loss_list))