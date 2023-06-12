import numpy as np
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
from sklearn.model_selection import StratifiedKFold

# Hobohm 1 algorithm for sequence clustering
def hobohm1(sequences, threshold):
    clusters = []
    for sequence in sequences:
        for cluster in clusters:
            alignment = pairwise2.align.globaldx(sequence, cluster[0], matlist.blosum62)
            score = alignment[0][2]
            if score >= threshold:
                cluster.append(sequence)
                break
        else:
            clusters.append([sequence])
    return clusters

# Stratified splitting of datasets
def create_folds(clusters, n_folds):
    skf = StratifiedKFold(n_splits=n_folds)
    folds = []
    for train_index, test_index in skf.split(clusters, [len(cluster) for cluster in clusters]):
        train_clusters = [cluster for i, cluster in enumerate(clusters) if i in train_index]
        test_clusters = [cluster for i, cluster in enumerate(clusters) if i in test_index]
        folds.append((train_clusters, test_clusters))
    return folds

# Sorting sequences
def sort_sequences(sequences):
    return sorted(sequences, key=lambda x: x[0])

# Load data
data = np.loadtxt('your_data_file.txt', dtype=str)
sequences = data[:, 0]
affinities = data[:, 1]
alleles = data[:, 2]

# Apply Hobohm 1
clusters = hobohm1(sequences, threshold=0.62)

# Create stratified folds
folds = create_folds(clusters, n_folds=5)

# Write output to new files
for i, (train_clusters, test_clusters) in enumerate(folds):
    train_sequences = sort_sequences([seq for cluster in train_clusters for seq in cluster])
    test_sequences = sort_sequences([seq for cluster in test_clusters for seq in cluster])
    
    np.savetxt(f'f{i:03d}.txt', train_sequences, fmt='%s')
    np.savetxt(f'c{i:03d}.txt', test_sequences, fmt='%s')
