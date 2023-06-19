#!/bin/bash
## Define path to your code directory
RDIR=$(pwd)
ResDIR="/zhome/62/8/145783/s193518/ESM-MHC-I-predictions/results"

## Define path you where you have placed the HLA data sets
## IMPORTANT NOTE: Remember to set the data directory before running the code
DDIR="/zhome/62/8/145783/s193518/ESM-MHC-I-predictions/data/processed_data"

for e in ESM #blosum sparse
do  
    # Here you can type your allele names
    for a in B5301 B5401 B5701 B5801
    do

        mkdir -p $RDIR/$a/$e"_with_sigmoid"
        mkdir -p $ResDIR/$a/$e"_with_sigmoid"

        for n in 0 1 2 3 4
        do
            cat $DDIR/$a/c00$n > $RDIR/$a/$e"_with_sigmoid"/test
            
            for m in `echo $n | gawk '{for ( i=0; i<5; i++ ) { if ( i != $1) { print i}}}'`
            do
                cat $DDIR/$a/c00$m > $RDIR/$a/$e"_with_sigmoid"/eval

                touch $RDIR/$a/$e"_with_sigmoid"/train
                rm -f $RDIR/$a/$e"_with_sigmoid"/train

                for l in `echo $n $m | gawk '{ for ( i=0; i<5; i++ ) { if ( i != $1 && i != $2) { print i}}}'`
                do
                    cat $DDIR/$a/c00$l >> $RDIR/$a/$e"_with_sigmoid"/train
                done
                #run train-code
                python3 $RDIR/FFNN_model.py -ef $e -m -p $RDIR/$a/$e"_with_sigmoid" -t train -e eval  -a $a --numbers $n $m
                python3 $RDIR/FFNN_eval.py -ef $e -p $RDIR/$a/$e"_with_sigmoid" -o $RDIR/$a/$e"_with_sigmoid" -t eval -m $RDIR/$a/$e"_with_sigmoid"/models/${a}_${e}_${n}_${m}_net.pt -a $a --numbers $n $m
            done
            # Finding the best model from the m-loop and discarding all the others
            # Currently this only saves the model - not any of the other data. If that should be changed then let me know
            python3 $RDIR/find_best_model.py -d $RDIR/$a/$e"_with_sigmoid"/models/ -o $ResDIR/$a/$e"_with_sigmoid"/
            model=$(find $ResDIR/$a/$e"_with_sigmoid"/best_models/ -maxdepth 1 -name "${a}_${e}_${n}_*")
            
            # Removing the models directory since keeping that around is not necesarry once the new model has been moved (?)
            rm -r $RDIR/$a/$e"_with_sigmoid"/models
            
            # Run evalaution evaluation
            python3 $RDIR/FFNN_eval.py -ef $e -p $RDIR/$a/$e"_with_sigmoid" -o $ResDIR/$a/$e"_with_sigmoid" -t test -a $a --numbers $n -m $model
        
        done
        #rm -rf $RDIR/$a
    done
done
