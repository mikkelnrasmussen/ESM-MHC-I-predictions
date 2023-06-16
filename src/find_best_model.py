#!/usr/bin/env/python3

import argparse
import os
import math

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_directory", required=True, help="Set path to directory containing information files on the trained models")
parser.add_argument("-o", "--output_dir", required=True, help="Set path to where output models should be moved")
args = parser.parse_args()

data_dir = args.data_directory
out_dir = args.output_dir

list_of_val_error = []
list_of_networks = []
list_of_graphs = []
list_of_metrics = []

for path, subdirectory, files in os.walk(data_dir):
    for name in files:
        if name.endswith(".txt"):
            list_of_val_error.append(os.path.join(path, name))
        elif name.endswith(".png"):
            list_of_graphs.append(name)
        elif name.endswith(".out"):
            list_of_metrics.append(os.path.join(path ,name))
        else:
            list_of_networks.append(name)
            
highest_combined_metric = -10000
for val_file, metric_file in zip(list_of_val_error, list_of_metrics):
    with open(val_file, "r") as val_infile:
        with open(metric_file, "r") as metric_infile:
            # The file should only contain one line
            for line in val_infile:
                if line != "":
                    val_loss = float(line.strip())
            for line in metric_infile:
                if line.startswith("AUC:"):
                    auc = float(line[4:].strip())
                    
            # Select model with best performance
            combined_metric = auc - val_loss
            if combined_metric > highest_combined_metric:
                highest_combined_metric = combined_metric
                file_found_within = val_file[-12:-9]

os.makedirs(out_dir + "graphs", exist_ok=True)
os.makedirs(out_dir + "best_models", exist_ok=True)

# Now to only move the best network and remove the rest
for network in list_of_networks:
    if file_found_within in network:
        os.replace(data_dir + "/" + network, out_dir + "best_models/" + network)

for graph in list_of_graphs:
    if file_found_within in graph:
        os.replace(data_dir + "/" + graph, out_dir + "graphs/" + graph)
        
