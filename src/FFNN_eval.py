#!/usr/bin/env python
# coding: utf-8


import torch
from torch.autograd import Variable
import torch.nn as nn
#import esm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

from argparse import ArgumentParser


parser = ArgumentParser(description="FFNN_evaluation python program")

parser.add_argument("-t", action="store", dest="test_file", type=str, help="File with test data")
parser.add_argument("-s", action="store", dest="seed", type=int, default=1, help="Seed for random numbers (default 1)")
parser.add_argument("-ef", action="store", dest="encoder_flag", type=str, help="Type of encoder used for the model (blosum, sparse, ESM)")
parser.add_argument("-a", action="store", dest="allele", type=str, help="Allele ID (e.g A0201,...)")
parser.add_argument("-nh", action="store", dest="hidden_layer_dim", type=int, default=32, help="Number of hidden neurons (default 32)")
parser.add_argument("--numbers", type=int, nargs='*', help="Supply the cycle numbers from bash to help name files")


args, unknown = parser.parse_known_args()
encoder_flag = args.encoder_flag
test_file = args.test_file
seed = args.seed
hidden_layer_dim = args.hidden_layer_dim
allele = args.allele
cycle_numbers = args.numbers


SEED = seed
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_blosum(filename):
    """
    Read in BLOSUM values into matrix.
    """
    aa = ['A', 'R', 'N' ,'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    df = pd.read_csv(filename, sep='\s+', comment='#', index_col=0)
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

def load_peptide_target(filename, encoder_flag):
    """
    Read amino acid sequence of peptides and
    corresponding log transformed IC50 binding values from text file.
    """
    if encoder_flag != "ESM":
        df = pd.read_csv(filename, 
                         sep = "\s+", 
                         usecols = [0,1], 
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

# ## Arguments


MAX_PEP_SEQ_LEN = 9
BINDER_THRESHOLD = 0.426


# ## Load


#ALLELE = 'A0201' #'A0301'

# Blosum62 need to be in working directory
blosum_file_62 = "../data/matrices/BLOSUM62"

# Files for testing
test_data = test_file


test_raw = load_peptide_target(test_data, encoder_flag)


# ### Encode data


x_test_, y_test_ = encode_peptides(test_raw, encoder_flag)


# ### Flatten tensors


x_test_ = x_test_.reshape(x_test_.shape[0], -1)
x_test_.shape[1]



n_features = x_test_.shape[1]


# ### Make data iterable


x_test = Variable(torch.from_numpy(x_test_.astype('float32')))
y_test = Variable(torch.from_numpy(y_test_.astype('float32'))).view(-1, 1)


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

N_HIDDEN_NEURONS = hidden_layer_dim
criterion = nn.MSELoss()


# ## Evaluation

# ### Load model
model_dir = "../models/%s/%s/%s/" % (allele, encoder_flag, cycle_numbers[0])
model_filename = "%s_%s_%s_%s_net.pt" % (allele, encoder_flag, cycle_numbers[0], cycle_numbers[1])
test_perf_filename = "%s_%s_%s_%s_test_perf.txt" % (allele, encoder_flag, cycle_numbers[0], cycle_numbers[1])
model_PATH = model_dir + model_filename
test_perf_PATH = model_dir + test_perf_filename


net = Net(n_features, N_HIDDEN_NEURONS)
net.load_state_dict(torch.load(model_PATH))


# ### Predict on test set
net.eval()
pred = net(x_test)
loss = criterion(pred, y_test)


# ### Transform targets to class
y_test_class = np.where(y_test.flatten() >= BINDER_THRESHOLD, 1, 0)
y_pred_class = np.where(pred.flatten() >= BINDER_THRESHOLD, 1, 0)


# ### Receiver Operating Caracteristic (ROC) curve
def plot_roc_curve(peptide_length=[9]):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label = 'AUC = %0.2f (%smer)' %(roc_auc, '-'.join([str(i) for i in peptide_length])))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], c='black', linestyle='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')



# Combining targets and prediction values with peptide length in a dataframe
pred_per_len = pd.DataFrame([test_raw.peptide.str.len().to_list(),
                             y_test_class,
                             pred.flatten().detach().numpy()],
                            index=['peptide_length','target','prediction']).T

#plt.figure(figsize=(7,7))
# For each peptide length compute AUC and plot ROC
for length, grp in pred_per_len.groupby('peptide_length'):
    fpr, tpr, threshold = roc_curve(grp.target, grp.prediction)
    roc_auc = auc(fpr, tpr)
    
    #plot_roc_curve(peptide_length=[int(length)])
#print('AUC: ', roc_auc)


# ### Matthew's Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_test_class, y_pred_class)
#print('MCC: ', mcc)

"""
def plot_mcc():
    plt.title('Matthews Correlation Coefficient')
    plt.scatter(y_test.flatten().detach().numpy(), pred.flatten().detach().numpy(), label = 'MCC = %0.2f' % mcc)
    plt.legend(loc = 'lower right')
    plt.ylabel('Predicted')
    plt.xlabel('Validation targets')
    plt.show()

plot_mcc()

"""
text = 'AUC: ' + str(roc_auc) + '\n' + 'MCC: ' + str(mcc)
with open(test_perf_PATH, 'w') as f:
    f.write(text)


