#!/bin/bash

start=0
end=9

for i in $(seq $start $end)
do
    power=$((2**i))
    CUDA_VISIBLE_DEVICES=5 python train.py class$power
    sleep 1
done