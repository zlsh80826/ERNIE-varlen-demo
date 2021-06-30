#!/bin/bash
set -xe

GPU_MODEL=`nvidia-smi -L | cut -d: -f2 | cut -d'(' -f1 | xargs | sed -e 's/ /-/g'`

if nvidia-smi -L | grep MIG > /dev/null; then
    GPU_MODEL=${GPU_MODEL}-MIG
fi


MIG_LIST=`nvidia-smi -L | grep MIG | cut -d' ' -f 14 | cut -d')' -f1`

for BS in 1 2 4 8 16 32 64 128 256
do
  counter=0
  for MIG in $MIG_LIST
  do
      CUDA_VISIBLE_DEVICES=$MIG python3 scripts/cbenchmark.py -d QNLI -m large -i trt-fp16 -b $BS &
      pids[$counter]=$!
      counter=$((counter + 1))
  done

  for pid in ${pids[*]}
  do
    wait ${pid}
  done
done

for BS in 1 2 4 8 16 32 64 128 256
do
  counter=0
  for MIG in $MIG_LIST
  do
      CUDA_VISIBLE_DEVICES=$MIG python3 scripts/cbenchmark.py -d QNLI -m large -i trt-fp16 -b $BS --sparse 1 &
      pids[$counter]=$!
      counter=$((counter + 1))
  done

  for pid in ${pids[*]}
  do
    wait ${pid}
  done
done
