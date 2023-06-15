#!/usr/bin/env python
# coding: utf-8

# # Train a neural network to predict MHC ligands
import torch
import esm
from torch.autograd import Variable
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pytorchtools import EarlyStopping

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from argparse import ArgumentParser


parser = ArgumentParser(description="FFNN_model python program")

parser.add_argument("-t", action="store", dest="training_file", type=str, help="File with training data")
parser.add_argument("-e", action="store", dest="evaluation_file", type=str, help="File with evaluation data")
parser.add_argument("-epi", action="store", dest="epsilon", type=float, default=0.01, help="Epsilon (default 0.01)")
parser.add_argument("-s", action="store", dest="seed", type=int, default=1, help="Seed for random numbers (default 1)")
parser.add_argument("-i", action="store", dest="epochs", type=int, default=3000, help="Number of epochs to train (default 3000)")
parser.add_argument("-ef", action="store", dest="encoder_flag", type=str, help="Type of encoder used for the model (blosum, sparse, ESM)")
parser.add_argument("-a", action="store", dest="allele", type=str, help="Allele ID (e.g A0201,...)")
parser.add_argument("-stop", action="store_true", dest="early_stopping", help="Use Early stopping")
parser.add_argument("-nh", action="store", dest="hidden_layer_dim", type=int, default=32, help="Number of hidden neurons (default 32)")
parser.add_argument("--numbers", type=int, nargs='*', help="Supply the cycle numbers from bash to help name files")

args, unknown = parser.parse_known_args()
encoder_flag = args.encoder_flag
training_file = args.training_file
evaluation_file = args.evaluation_file
epsilon = args.epsilon
epochs = args.epochs
seed = args.seed
early_stopping = args.early_stopping
hidden_layer_dim = args.hidden_layer_dim
allele = args.allele
cycle_numbers = args.numbers


SEED= seed
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_blosum(filename):
    """
    Read in BLOSUM values into matrix.
    """
    aa = ['A', 'R', 'N' ,'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    df = pd.read_csv(filename, sep='\s+', comment='#', index_col=0)
    df.loc[aa, aa]
    return df.loc[aa, aa]


def create_soft_sparse():
    """
    Create soft sparse matrix.
    """
    aa = ['A', 'R', 'N' ,'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    size = len(aa)
    data = [[0.9 if i == j else 0.05 for j in range(size)] for i in range(size)]
    df = pd.DataFrame(data, index=aa, columns=aa)
    return df

def load_ESM_1(filename):
    """
    Read in ESM-1 values into matrix.
    """
    # XXX
    return

def load_peptide_target(filename, encoder_flag):
    """
    Read amino acid sequence of peptides and
    corresponding log transformed IC50 binding values from text file.
    """
    if encoder_flag != "ESM":
        df = pd.read_csv(filename, 
                         sep = "\s+", 
                         usecols = [0,2], 
                         names = ["peptide", "target"])
    else:
        df = pd.read_csv(filename, 
                         sep = "\s+", 
                         usecols = [0, 1, 2, 3, 4], 
                         names = ["peptide", "protein", "target", "start", "stop"])
        
    return df.sort_values(by='target', ascending=False).reset_index(drop=True)



def encode_peptides(Xin, encoder_flag):
    """
    Encode AA seq of peptides using BLOSUM50.
    Returns a tensor of encoded peptides of shape (batch_size, MAX_PEP_SEQ_LEN, n_features)
    """
    batch_size = len(Xin)
    
    if encoder_flag == "blosum":
        blosum = load_blosum(blosum_file_62)
        n_features = len(blosum)
    
        Xout = np.zeros((batch_size, MAX_PEP_SEQ_LEN, n_features), dtype=np.int8)
    
        for peptide_index, row in Xin.iterrows():
            for aa_index in range(len(row.peptide)):
                Xout[peptide_index, aa_index] = blosum[ row.peptide[aa_index] ].values
                
    elif encoder_flag == "sparse":
        sparse = create_soft_sparse()
        n_features = len(sparse)
    
        Xout = np.zeros((batch_size, MAX_PEP_SEQ_LEN, n_features), dtype=np.float32)
    
        for peptide_index, row in Xin.iterrows():
            for aa_index in range(len(row.peptide)):
                Xout[peptide_index, aa_index] = sparse[ row.peptide[aa_index] ].values
    
    elif encoder_flag == "ESM":
        
        # Load ESM-1b model
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        batch_converter = alphabet.get_batch_converter()
        
        n_features = model.args.embed_dim  # ESM model's embedding dimension
        Xout = np.zeros((batch_size, 9, n_features), dtype=np.float32)  # Assuming a fixed length of 9 for the peptide

        for peptide_index, row in Xin.iterrows():
            protein_sequence = row.protein
            start, stop = row.start, row.stop

            # Calculate how much of the sequence before and after the peptide we can include
            before = min(506, start)  # number of amino acids before the peptide
            after = min(506, len(protein_sequence) - stop)  # number of amino acids after the peptide

            # If we can't include 506 amino acids on both sides, take more from the other side
            if before < 506:
                after = min(len(protein_sequence) - stop, 1014 - before)
            elif after < 506:
                before = min(start, 1014 - after)

            # Extract the part of the protein sequence we're interested in
            extract_start = start - before
            extract_stop = stop + after
            extract_sequence = protein_sequence[extract_start:extract_stop]
            print(len(extract_sequence))

            data = [["protein", extract_sequence]]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            print(batch_tokens)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]

            # The peptide is now at a new position in the sequence
            new_start = before
            new_stop = new_start + 9
            peptide_representation = token_representations[0, new_start:new_stop]
            Xout[peptide_index] = peptide_representation.cpu().numpy()

    return Xout, Xin.target.values



def invoke(early_stopping, loss, model, implement=False):
    if implement == False:
        return False
    else:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            return True



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ## Arguments

MAX_PEP_SEQ_LEN = 9 
BINDER_THRESHOLD = 0.426


# # Main

# ## Load

### Windows path corrector
#windows_path = os.getcwd()
#linux_path = windows_path.replace('\\', '/')
#working_dir = linux_path

# Blosum62 need to be in working directory
blosum_file_62 = "../data/matrices/BLOSUM62"

# Files for debugging
#train_data = working_dir + "/../data/%s/train_BA" % ALLELE
#valid_data = working_dir + "/../data/%s/valid_BA" % ALLELE

# Files for training
train_data = training_file
valid_data = evaluation_file


train_raw = load_peptide_target(train_data, encoder_flag)
valid_raw = load_peptide_target(valid_data, encoder_flag)
print(train_raw.loc[0:2, :])


# ### Encode data
# For debugging
# encoder_flag = 'blosum'
x_train_, y_train_ = encode_peptides(train_raw.loc[0:2, :], encoder_flag)
x_valid_, y_valid_ = encode_peptides(valid_raw.loc[0:2, :], encoder_flag)
quit()

# Check the data dimensions for the train set and validation set (batch_size, MAX_PEP_SEQ_LEN, n_features)

#print(x_train_.shape)
#print(x_valid_.shape)


# ### Flatten tensors
x_train_ = x_train_.reshape(x_train_.shape[0], -1)
x_valid_ = x_valid_.reshape(x_valid_.shape[0], -1)



batch_size = x_train_.shape[0]
n_features = x_train_.shape[1]


# ### Make data iterable
x_train = Variable(torch.from_numpy(x_train_.astype('float32')))
y_train = Variable(torch.from_numpy(y_train_.astype('float32'))).view(-1, 1)

x_valid = Variable(torch.from_numpy(x_valid_.astype('float32')))
y_valid = Variable(torch.from_numpy(y_valid_.astype('float32'))).view(-1, 1)


# ## Build Model
class Net(nn.Module):

    def __init__(self, n_features, n_l1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, n_l1)
        self.fc2 = nn.Linear(n_l1, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ## Select Hyper-parameters

def init_weights(m):
    """
    https://pytorch.org/docs/master/nn.init.html
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0) # alternative command: m.bias.data.fill_(0.01)

EPOCHS = epochs
MINI_BATCH_SIZE = 100
N_HIDDEN_NEURONS = hidden_layer_dim
LEARNING_RATE = epsilon
PATIENCE = EPOCHS // 10


# ## Compile Model

net = Net(n_features, N_HIDDEN_NEURONS)
#net.apply(init_weights)

#count_parameters(net)

optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()


# ## Train Model

# No mini-batch loading
# mini-batch loading
def train():
    train_loss, valid_loss = [], []

    early_stopping = EarlyStopping(patience=PATIENCE)

    for epoch in range(EPOCHS):
        net.train()
        pred = net(x_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data)

        if epoch % (EPOCHS//10) == 0:
            print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.data))

        net.eval()
        pred = net(x_valid)
        loss = criterion(pred, y_valid)  
        valid_loss.append(loss.data)

        if invoke(early_stopping, valid_loss[-1], net, implement=True):
            net.load_state_dict(torch.load('checkpoint.pt'))
            break
            
    return net, train_loss, valid_loss



# Train with mini_batches
train_loader = DataLoader(dataset=TensorDataset(x_train, y_train), batch_size=MINI_BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=TensorDataset(x_valid, y_valid), batch_size=MINI_BATCH_SIZE, shuffle=True)

def train_with_minibatches():
    
    train_loss, valid_loss = [], []

    early_stopping = EarlyStopping(patience=PATIENCE)
    for epoch in range(EPOCHS):
        batch_loss = 0
        net.train()
        for x_train, y_train in train_loader:
            pred = net(x_train)
            loss = criterion(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.data
        train_loss.append(batch_loss / len(train_loader))

        batch_loss = 0
        net.eval()
        for x_valid, y_valid in valid_loader:
            pred = net(x_valid)
            loss = criterion(pred, y_valid)
            batch_loss += loss.data
        valid_loss.append(batch_loss / len(valid_loader))
        
        if epoch % (EPOCHS//10) == 0:
            print('Train Epoch: {}\tLoss: {:.6f}\tVal Loss: {:.6f}'.format(epoch, train_loss[-1], valid_loss[-1]))

        if invoke(early_stopping, valid_loss[-1], net, implement=True):
            net.load_state_dict(torch.load('checkpoint.pt'))
            break
            
    return net, train_loss, valid_loss


# ### Train model

#net, train_loss, valid_loss = train()                   # no mini-batch loading
net, train_loss, valid_loss = train_with_minibatches()   # mini-batch loading


# ### Save model
model_dir = "../models/%s/%s/%s" % (allele, encoder_flag, cycle_numbers[0])
os.makedirs(model_dir, exist_ok=True)
model_filename = "%s_%s_%s_%s_net.pt" % (allele, encoder_flag, cycle_numbers[0], cycle_numbers[1])
torch.save(net.state_dict(), model_dir + model_filename)

