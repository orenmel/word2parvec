#!/bin/bash
# This script is used to apply subvec clustering to multiple subvec files in a directory

echo source home $1 
echo process num $2
echo vocab $3
echo cluster_num $4
echo min avg cluster size $5
echo cluster prunning $6
echo input dir $7
echo output dir $8
echo number inits $9
echo max iterations ${10}

cd $1
FILECOUNT="$(ls $7 | wc -l)"
echo filecount $FILECOUNT
let FPP=FILECOUNT/$2+1
echo files per process $FPP
COUNTER=0
while [  $COUNTER -lt $FILECOUNT ]; do
    let from=COUNTER
    let to=COUNTER+FPP
    echo "Running: /usr/bin/python parvecs/setup/cluster_subvecs.py $3 $4 $5 $6 $7 $8 $from $to $9 ${10} &"
    /usr/bin/python parvecs/setup/cluster_subvecs.py $3 $4 $5 $6 $7 $8 $from $to $9 ${10} &
    let COUNTER=COUNTER+FPP 
done
