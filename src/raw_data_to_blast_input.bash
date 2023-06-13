#!/bin/bash

# Initialize or clear the blast_file.fasta
blastfile="../data/blast/blast_seqs"
rm blastfile
touch blastfile

# Loop over each folder in data/raw_data
for folder in ../data/raw_data/*; do
    if [ -d "$folder" ]; then
        # If the folder contains a name.dat file
        for file in "$folder"/*.dat; do
            # Extract the first column, get unique lines, and append them to blast_file.fasta
            awk '{print $1}' "$file" | sort -u >> "$blastfile"
        done
    fi
done

# Sort the blast_file.fasta uniquely
sort -u "$blastfile" -o "$blastfile"
