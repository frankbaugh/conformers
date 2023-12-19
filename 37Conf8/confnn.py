import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd

def set_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
        print("Using GPU")
    else:
        device = torch.device("cpu")   # Use CPU
        print("Using CPU")
    
    return device

class cm_pairs_Dataset(Dataset):
    def __init__(self, features, labels):
        #Convert to torch tensor then load onto GPU
        self.features = torch.Tensor(features).to(device)
        self.labels = torch.Tensor(labels).to(device)
    
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

## c = conformer, L = layer
class NeuralNetwork(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.layersize = 256
        self.n_input = n_input
        self.c1L1 = nn.Linear(self.n_input, self.layersize)
        self.c2L1 = nn.Linear(self.n_input, self.layersize)
        self.c1L2 = nn.Linear(self.layersize, self.layersize)
        self.c2L2 = nn.Linear(self.layersize, self.layersize)
        self.L3 = nn.Linear(2 * self.layersize, 2 * self.layersize)
        self.L4 = nn.Linear(2 * self.layersize, self.layersize)
        self.L5 = nn.Linear(self.layersize, 1)

    def forward(self, X):
        y1, y2 = X[:,:self.n_input], X[:,self.n_input:]
        y1 = F.relu(self.c1L1(y1))
        y2 = F.relu(self.c2L1(y2))
        y1 = F.relu(self.c1L2(y1))
        y2 = F.relu(self.c2L2(y2))
        y = torch.cat((y1, y2), dim=1)
        y = F.relu(self.L3(y))
        y = F.relu(self.L4(y))
        y = self.L5(y)
        return y


def get_data():
    path = '/Users/frank/code/conf/37Conf8/'
    # path = '/home/fmvb21/'
    
    pairsdf = pd.read_pickle(path + 'panda_pairs.pkl')
    labelsdf = pd.read_pickle(path + 'panda_labels.pkl')

    ##Split test/train, also convert pandas series to numpy arrays
    
    # Cross validation?
    nmols = 37
    nconfs = 8
    
    # Ensure that it doesnt train on all the molecules
    #Get approx 80 % for training, but ensure divisibility so that 
    # No contamination of some test molecule conformers
    
    frac = int(0.8 * nmols) / nmols
    test_idx = int(frac* len(pairsdf))
    train_features = pairsdf.iloc[:test_idx, :].values
    test_features = pairsdf.iloc[test_idx:, :].values
    train_labels = labelsdf.iloc[:test_idx, :].values
    test_labels = labelsdf.iloc[test_idx:, :].values

    train_data = cm_pairs_Dataset(train_features, train_labels)
    test_data = cm_pairs_Dataset(test_features, test_labels)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    descriptor_length = train_data.features.shape[1]
    
    return train_dataloader, test_dataloader, descriptor_length

device = set_gpu()
train_dataloader, test_dataloader, descriptor_length = get_data()


def train_loop(dataloader, model, loss_fn, optimiser, train_loss_list):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    
    for batch, (X, y) in enumerate(dataloader):

        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        
        total_loss += loss.item()
        
        if batch % 32 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss_list.append(total_loss/(batch+1))

def test_loop(dataloader, model, loss_fn, test_loss_list):
    
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
            correct += ((pred > 0.5).squeeze().long() == y.squeeze().long()).sum().item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()


    test_loss /= num_batches
    test_loss_list.append(test_loss)
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss:{test_loss:>8f} \n")


torch.manual_seed(50)
model = NeuralNetwork(int(descriptor_length / 2))
model = model.to(device)
learning_rate = 1e-3
epochs = 500

loss_fn = nn.BCEWithLogitsLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01)

train_loss_list = []
test_loss_list = []
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------")
    train_loop(train_dataloader, model, loss_fn, optimiser, train_loss_list)
    test_loop(test_dataloader, model, loss_fn, test_loss_list)

print("Done!")

np.save('test_loss.npy', np.asarray(test_loss_list))
np.save('train_loss.npy', np.asarray(train_loss_list))