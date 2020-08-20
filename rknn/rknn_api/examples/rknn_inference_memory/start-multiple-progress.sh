#!/bin/bash


for i in {1..3}
do
    echo $i;
    nohup ./rknn_inference_memory mobilenet_v2.rknn dog_224x224.jpg  224 224 &
done
