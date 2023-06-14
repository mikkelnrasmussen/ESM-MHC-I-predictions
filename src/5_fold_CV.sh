#!/bin/bash
for e in blosum sparse esm
do
    ## Define path to your code directory
    RDIR=$(pwd)

    ## Define path you where you have placed the HLA data sets
    DDIR="../data/raw_data"
    
    # If we are using the ESM encoder then we need to use the alternate datasets
    if [$e = esm]
    then
        DDIR="../data/esm/"
    fi

    # Here you can type your allele names
    for a in A0101 A0201 A0202 A0203 A0206 A3001 A1101 A2402 A2403 A2601 A2902 A3001 A3002 A3101 A3301 A6801 A6802 A6901 B0702 B0801 B1501 B1801 B2705 B3501 B4001 B4002 B4402 B4403 B4501 B5101 B5301 B5401 B5701 B5801
    do

        mkdir -p $a.res

        cd $a.res

        for n in 0 1 2 3 4
        do
            cat c00$n > test

            for m in `echo $n | gawk '{for ( i=0; i<5; i++ ) { if ( i != $1) { print i}}}'`
            do
                cat c00$m > eval

                touch train
                rm -f train

                for l in `echo $n $m | gawk '{ for ( i=0; i<5; i++ ) { if ( i != $1 && i != $2) { print i}}}'`
                do
                    cat f00$l >> train
                    #cat c00$l >> train
                done

                #run train-code
                python $RDIR/FFNN.py --encoder $e -t train -e eval
                #run test evaluation

            done

            # Pick model with best test perf

            # Run evalaution evaluation
        done
    done
done