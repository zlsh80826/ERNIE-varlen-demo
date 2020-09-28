#!/bin/bash

export LD_LIBRARY_PATH=/home/rewang/tools/cuda-11.0/TensorRT-7.2.1.3/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/rewang/tools/cuda-11.0/cudnn-8.0/lib64:$LD_LIBRARY_PATH

set -xe

python scripts/cbenchmark.py -d QNLI -m base -i trt-fp16 -b 32
