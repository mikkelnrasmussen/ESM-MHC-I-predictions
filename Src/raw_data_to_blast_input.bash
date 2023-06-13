#!/bin/bash

# Initialize or clear the blast_file.fasta
> ../data/blast/blast_file.fasta

# Loop over each folder in data/raw_data
for folder in ../data/raw_data/*; do
    if [ -d "$folder" ]; then
        # If the folder contains a name.dat file
        if [ -f "$folder/*.dat" ]; then
            # Extract the first column, get unique lines, and append them to blast_file.fasta
            awk '{print $1}' "$folder/*dat" | sort -u >> ../data/blast/blast_file.fasta
        fi
    fi
done

# Sort the blast_file.fasta uniquely
sort -u ../data/blast/blast_file.fasta -o ../data/blast/blast_file.fasta
