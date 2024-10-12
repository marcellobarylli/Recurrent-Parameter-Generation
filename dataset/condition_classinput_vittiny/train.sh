#!/bin/bash

start=1
end=1022

for i in $(seq $start $end)
do
    CUDA_VISIBLE_DEVICES=5 python train.py class$i
    sleep 1
done