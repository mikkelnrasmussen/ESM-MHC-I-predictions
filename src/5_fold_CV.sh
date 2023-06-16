#!/bin/bash
## Define path to your code directory
RDIR=$(pwd)
ResDIR="/home/people/s193518/esm-mhc/ESM-MHC-I-predictions/results"

## Define path you where you have placed the HLA data sets
## IMPORTANT NOTE: Remember to set the data directory before running the code
DDIR="/home/people/s193518/esm-mhc/ESM-MHC-I-predictions/data/processed_data"

for e in sparse blosum #ESM
do  
    # Here you can type your allele names
    for a in A0101 A0201 A0202 A0203 A0206 A3001 A1101 A2402 A2403 A2601 A2902 A3001 A3002 A3101 A3301 A6801 A6802 A6901 B0702 B0801 B1501 B1801 B2705 B3501 B4001 B4002 B4402 B4403 B4501 B5101 B5301 B5401 B5701 B5801
    do

        mkdir -p $RDIR/$a/$e
        mkdir -p $ResDIR/$a/$e

        for n in 0 1 2 3 4
        do
            cat $DDIR/$a/c00$n > $RDIR/$a/$e/test
            
            for m in `echo $n | gawk '{for ( i=0; i<5; i++ ) { if ( i != $1) { print i}}}'`
            do
                cat $DDIR/$a/c00$m > $RDIR/$a/$e/eval

                touch $RDIR/$a/$e/train
                rm -f $RDIR/$a/$e/train

                for l in `echo $n $m | gawk '{ for ( i=0; i<5; i++ ) { if ( i != $1 && i != $2) { print i}}}'`
                do
                    cat $DDIR/$a/c00$l >> $RDIR/$a/$e/train
                done
                #run train-code
                $HOME/python3.11/bin/python3 $RDIR/FFNN_model.py -ef $e -m -p $RDIR/$a/$e -t train -e eval  -a $a --numbers $n $m
                $HOME/python3.11/bin/python3 $RDIR/FFNN_eval.py -ef $e -p $RDIR/$a/$e -o $RDIR/$a/$e -t eval -m $RDIR/$a/$e/models/${a}_${e}_${n}_${m}_net.pt -a $a --numbers $n $m
            done
            # Finding the best model from the m-loop and discarding all the others
            # Currently this only saves the model - not any of the other data. If that should be changed then let me know
            $HOME/python3.11/bin/python3 $RDIR/find_best_model.py -d $RDIR/$a/$e/models/ -o $ResDIR/$a/$e/
            model=$(find $ResDIR/$a/$e/best_models/ -maxdepth 1 -name "${a}_${e}_${n}_*")
            
            # Removing the models directory since keeping that around is not necesarry once the new model has been moved (?)
            rm -r $RDIR/$a/$e/models
            
            # Run evalaution evaluation
            $HOME/python3.11/bin/python3 $RDIR/FFNN_eval.py -ef $e -p $RDIR/$a/$e -o $ResDIR/$a/$e -t test -a $a --numbers $n -m $model
        
        done
        rm -rf $RDIR/$a
    done
done
