#!/bin/bash

start=0
end=19

for i in $(seq $start $end)
do
    python train.py class$i
    sleep 1
done