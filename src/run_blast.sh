#!/bin/bash

# Set the input file
input_file="$1"

# Run the blastp command
blastp -query "$input_file" -db /home/databases/blast/nr -out results.out -word_size 2 -evalue 200000 -max_target_seqs 100 -gapopen 9 -gapextend 1 -matrix PAM30 -soft_masking false -lcase_masking -threshold 11 -comp_based_stats 0 -window_size 40 -task blastp-short -taxidlist vira_bac_fungi.txids -outfmt "6 qseqid qseq sseqid staxid scomname pident length qlen slen qstart qend sstart send evalue bitscore qcovs" -num_threads 40

# Filter the output for 100% sequence identity and coverage
awk '($6 == 100 && $15 == 100) {print $0}' results.out > filtered_results.out
