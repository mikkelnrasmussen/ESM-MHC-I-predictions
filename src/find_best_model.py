#!/usr/bin/env/python3

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_directory", required=True, help="Set path to directory containing information files on the trained models")
args = parser.parse_args()

data_dir = args.data_directory

list_of_data_files = []
list_of_networks = []

for path, subdirectory, files in os.walk(data_dir):
    for name in files:
        if name.endswith(".txt"):
            list_of_data_files.append(os.path.join(path, name))
        else:
            list_of_networks.append(name)

lowest_val_loss = 10000

for file in list_of_data_files:
    with open(file, "r") as infile:
        # The file should only contain one line
        for line in file:
            train_epoch, loss, val_loss = line.split("\t")
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                file_found_within = file[-7:-4]

# Now to only move the best network and remove the rest
for network in list_of_networks:
    if file_found_within in network:
        os.replace(data_dir + "/" + network, "../" + network)
