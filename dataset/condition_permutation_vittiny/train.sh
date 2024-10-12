#!/bin/bash

start=0
end=19

for i in $(seq $start $end)
do
    CUDA_VISIBLE_DEVICES=5 python train.py class$i
    sleep 1
done