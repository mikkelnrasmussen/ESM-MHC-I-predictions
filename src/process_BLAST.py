#!/usr/bin/env/python3

import os
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", required=True, help="Give the path to your data directory where the alleles are kept")
parser.add_argument("-b", "--blast_results", required=True, help="Path to the file in which results from the BLAST run is kept")
parser.add_argument("-o", "--output_dir", required=True, help="Set output directory in which subdirectories will be located")
args = parser.parse_args()

root = args.path
blast_results_file = args.blast_results
output_dir = args.output_dir

# Ensuring that the output dir ends with a / (necesarry in future)
if not output_dir.endswith("/"):
    output_dir = output_dir + "/"

# Creating a list of all the full paths to the individual CV files
list_of_alleles = []
dict_of_OG_files = {}

# First getting all the alleles
for path, subdirs, files in os.walk(root):
    for subdir in subdirs:
        list_of_alleles.append(subdir)

# Next setting up a dictionary of the unique files for each allele. Getting their full path
for allele in list_of_alleles:
    dict_of_OG_files[allele] = []
    for path, subdirs, files in os.walk(root + allele):
        for name in files:
            if not name.endswith(".dat"):
                dict_of_OG_files[allele].append(os.path.join(path, name))

# Creating the blast_dict and filling it with 9-mers, sequences and start and end point for the 9-mer
blast_dict = {}

with open(blast_results_file, "r") as blast_file:
    for line in blast_file:
        seq_id, nine_mer, prot_id, pos_start, pos_end, sequence = line.strip().split("\t")
        if sequence != "":
            blast_dict[nine_mer] = (sequence, pos_start, pos_end)

# Creating output directory
os.makedirs(output_dir, exist_ok=True)

# Creating a new directory for each allele and filling it with files that are pre-partioned into CV files from earlier
# Just now with the sequences and start / stop points added in.
for allele in list_of_alleles:
    counter = 0
    os.makedirs(output_dir + allele, exist_ok=True)
    for file in dict_of_OG_files[allele]:
        with open(output_dir + allele + "/" + file[-4:], "w") as outfile:
            with open(file, "r") as infile:
                for line in infile:
                    nine_mer, binding_affinity, file_allele = line.strip().split()
                    try:
                        outfile.write(nine_mer + "\t" + blast_dict[nine_mer][0] + "\t" + binding_affinity + "\t" + blast_dict[nine_mer][1] + "\t" + blast_dict[nine_mer][2] + "\t" + file_allele + "\n")
                    except KeyError:
                        counter += 1
                        continue
    print("For {}, there was a total of {} lost 9-mers".format(allele, counter/5), file=sys.stderr, flush=True)
