#!/bin/bash
set -xe

export LD_LIBRARY_PATH=<path to tensorrt lib>:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=<path to cudnn lib64>:$LD_LIBRARY_PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

for LEN in 256 384;
do
DATA=baidu/ch.$LEN.dev.inference.dynamic
MODEL=baidu/infer_model
MODE=trt-fp16

mkdir A10

for bs in 20 32;
do
# seq_lens = 0 -> var-len
./build/inference --logtostderr     \
	          --model ${MODEL}  \
                  --data  ${DATA}   \
                  --mode  ${MODE}   \
                  --batch_size $bs  \
                  --num_labels 2    \
                  --seq_lens   0    \
                  --out_predict 0   \
                  --ignore_copy 0   \
                  --min_graph   3   2> A10/bs.$bs.ch.$LEN.var.log

sleep 300
# seq_lens = $LEN, pad and truncate seq_len to $LEN
./build/inference --logtostderr     \
	          --model ${MODEL}  \
                  --data  ${DATA}   \
                  --mode  ${MODE}   \
                  --batch_size $bs  \
                  --num_labels 2    \
                  --seq_lens   $LEN \
                  --out_predict 0   \
                  --ignore_copy 0   \
                  --min_graph   3   2> A10/bs.$bs.ch.$LEN.fix.log

sleep 300

done
done
