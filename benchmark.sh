#!/bin/bash
set -xe

GPU_MODEL=`nvidia-smi -L | cut -d: -f2 | cut -d'(' -f1 | xargs | sed -e 's/ /-/g'`

if nvidia-smi -L | grep MIG > /dev/null; then
    GPU_MODEL=${GPU_MODEL}-MIG
fi

# clean serialized file
rm models/*/*.engine models/*/_opt_cache -rf

# benchmark dense
python3 scripts/cbenchmark.py -d QNLI -m large -i trt-fp16 -b {1,2,4,8,16,32,64,128,256} --stats_csv logs/${GPU_MODEL}-large-dense.csv

# benchmark sparse weight
python3 scripts/cbenchmark.py -d QNLI -m large -i trt-fp16 -b {1,2,4,8,16,32,64,128,256} --sparse 1 --stats_csv logs/${GPU_MODEL}-large-sparse.csv
