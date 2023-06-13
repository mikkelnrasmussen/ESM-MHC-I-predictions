#!/bin/bash

# Define the encoding schemes
declare -a encodings=("sparse" "blosum" "ESM")

# Loop over each folder in data/raw_data
for folder in ../data/raw_data/*; do
    if [ -d "$folder" ]; then
        # Extract the folder name
        folder_name=$(basename "$folder")

        # Create a corresponding folder in models
        mkdir -p "../models/$folder_name"

        # Loop over each encoding scheme
        for encoding in "${encodings[@]}"; do
            # Run the train_network script with the encoding scheme and folder name as arguments
            python train_network.py "-$encoding" "-$folder_name" > "../models/$folder_name/output_$encoding.txt"
        done
    fi
done
