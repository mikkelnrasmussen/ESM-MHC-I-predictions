#!/bin/bash

# File to modify
file="../data/blast/blast_seqs"

# Temporary file
temp_file="../data/blast/temp_blast_seqs"

# Initialize counter
counter=0

# Read file line by line
while IFS= read -r line
do
  # Write sequence header and line to temporary file
  echo ">seq${counter}" >> "$temp_file"
  echo "$line" >> "$temp_file"
  
  # Increment counter
  ((counter++))
done < "$file"

# Replace original file with modified temporary file
mv "$temp_file" "../data/blast/blast_input.fasta"
