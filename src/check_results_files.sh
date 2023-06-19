#!/bin/bash

# Specify the directory
DIR_PATH="/zhome/62/8/145783/s193518/ESM-MHC-I-predictions/results"
#!/bin/bash

# Define the outer loops
ESMs="blosum"
ALLELES="A0101 A0201 A0202 A0203 A0206 A3001 A1101 A2402 A2403 A2601 A2902 A3001 A3002 A3101 A3301 A6801 A6802 A6901 B0702 B0801 B1501 B1801 B2705 B3501 B4001 B4002 B4402 B4403 B4501 B5101 B5301 B5401 B5701 B5801"

# Start the outer loops
for e in $ESMs
do
    for a in $ALLELES
    do
        # Initialize the counter
        COUNTER=0

        # Formulate the specific directory path
        SPECIFIC_DIR_PATH="$DIR_PATH/$a/$e/best_models"

        # Check if the specific directory exists
        if [ -d "$SPECIFIC_DIR_PATH" ]
        then
            # Use a for loop to iterate over each file in the directory
            for file in "$SPECIFIC_DIR_PATH"/*; do
                # Check if the file ends with "performance_results.out"
                if [[ $file == *"performance_results.out" ]]; then
                    # If it does, increment the counter
                    ((COUNTER++))
                fi
            done

            # Print the count
            echo "In folder $a $e. Found 'performance_results.out' files: $COUNTER times"
        else
            echo "The directory $SPECIFIC_DIR_PATH does not exist"
        fi
    done
done

