#!/bin/bash
set -xe

GPU_MODEL=`nvidia-smi -L | cut -d' ' -f4`

# benchmark dense
python3 scripts/cbenchmark.py -d QNLI -m large -i trt-fp16 -b {1,2,4,8,16,32,64,128,256} --stats_csv logs/${GPU_MODEL}-large-dense.csv

# benchmark sparse weight
python3 scripts/cbenchmark.py -d QNLI -m large -i trt-fp16 -b {1,2,4,8,16,32,64,128,256} --sparse 1 --stats_csv logs/${GPU_MODEL}-large-sparse.csv
