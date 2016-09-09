#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for i in `seq 0 3`;
do
    echo 'Running script with option ' $i
    #nohup python $DIR/script.py $i &
done
