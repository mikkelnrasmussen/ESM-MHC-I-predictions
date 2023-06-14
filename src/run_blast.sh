#!/bin/bash

# Function to rename file if it exists
rename_if_exists() {
    local file="$1"
    if [[ -e "$file" ]]; then
        mv "$file" "${file%.*}_old.${file##*.}"
    fi
}

# Set the input file
input_file="$1"
data_dir="../data/blast"
db_path="/home/databases/blast/nr"

# Run the blastp command
echo "BLASTp: Running..."
blastp -query "$input_file" -db $db_path -out "$data_dir/blast_results.out" -word_size 2 -evalue 200000 -max_target_seqs 100 -gapopen 9 -gapextend 1 -matrix PAM30 -soft_masking false -lcase_masking -threshold 11 -comp_based_stats 0 -window_size 40 -taxidlist "$data_dir/vira_bac_fungi.taxids" -outfmt "6 qseqid qseq sseqid staxid pident length qlen slen qstart qend sstart send evalue bitscore qcovs" -num_threads 50
echo "BLASTp: Done!"

# Filter the output for 100% sequence identity and coverage, sort by e-value, and then by query 
# sequence ID, keeping only the top hit for each query sequence
echo "Extracting top hits: Running..."
awk '($5 == 100 && $15 == 100 && $6 == 9) {print $0}' "$data_dir/blast_results.out" > "$data_dir/top_hits.out"
echo "Extracting top hits: Done!"

# Create final_output.out with the query sequence, start and stop position in the database sequence, and the complete subject sequence
# Skip identical query sequences
echo "Creating final output: Running..."
declare -A queryseqs
rename_if_exists "$data_dir/final_output.out"
while IFS=$'\t' read -r qseqid qseq sseqid staxid pident length qlen slen qstart qend sstart send evalue bitscore qcovs
do
    if [[ -z "${queryseqs[$qseq]}" ]]; then
        protein_seq=$(blastdbcmd -db /home/databases/blast/nr -entry "$sseqid" -outfmt %s)
        if [[ ! -z "$protein_seq" ]]; then
            sseqid_clean=$(echo "$sseqid" | cut -d '|' -f 2)
            echo -e "$qseqid\t$qseq\t$sstart\t$send\t$sseqid_clean\t$protein_seq" >> "$data_dir/final_output.out"
            queryseqs[$qseq]=1
        fi
    fi
done < "$data_dir/top_hits.out"
echo "Creating final output: Done!"