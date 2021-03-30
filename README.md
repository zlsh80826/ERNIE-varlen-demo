# PaddlePaddle/ERNIE var-len demo
TensorRT 7.2 provides several plugins to support var-len BERT, this repo demos the ability and the performance of this feature

## Introduction
In the past, if we want to inference sequences with different sequence lengths, we have to pad the shorter sequences

For example, if we have following four sequences in an inference batch
```
1. AAAA
2. BB
3. CCC
4. D
```

Then we have to do the padding
```
1. AAAA
2. BBXX
3. CCCX
4. DXXX
```

However, this leads to the redundant computation on the padding values.

So, TensorRT 7.2 provides several plugins to support the var-len situation. 

Now, we feed the concatenated sequence as input instead of padding the shorter sequences.
```
1. AAAABBCCCD
```

See [Other details of this benchmark](#other-details-of-this-benchmark) for the input description

## Requirements
* python >= 3.6
* [cuda](https://developer.nvidia.com/cuda-downloads) >= 11.0
* [cudnn](https://developer.nvidia.com/cudnn) >= 8.0
* [TensorRT](https://developer.nvidia.com/tensorrt) >= 7.2.0.14

## Build PaddlePaddle with varlen features

We are still in progress on merging the varlen features into PaddlePaddle develop branch. We show the demo by [this branch](https://github.com/zlsh80826/Paddle/tree/nvinfer_plugin_var_len_cuda11). 

Follow the below instructions to build the paddle inference api.

``` bash
$ git clone 'https://github.com/PaddlePaddle/Paddle.git' && cd Paddle
$ mkdir build-env && cd build-env
$ cmake .. -DWITH_PYTHON=OFF \
           -DWITH_GPU=ON \
           -DWITH_TESTING=OFF \
           -DWITH_INFERENCE_API_TEST=OFF \
           -DCMAKE_BUILD_TYPE=Release \
           -DCUDA_ARCH_NAME=Turing \
           -DON_INFER=ON \
           -DWITH_MKL=OFF \
           -DWITH_AVX=OFF \
           -DWITH_NCCL=OFF \
           -DTENSORRT_ROOT=<path to TensorRT 7.2.0.14 root> \
           -DCUDNN_ROOT=<path to cudnn 8.0 root> \
           -DWITH_NVINFER_PLUGIN=ON
$ ulimit -n 2048
$ make -j `nproc`
```

## Build ERNIE-varlen demo
```bash
$ git clone 'https://github.com/zlsh80826/ERNIE-varlen-demo.git'
$ cd ERNIE-varlen-demo
$ mkdir build && cd build
$ cmake ../src -DFLUID_INFER_LIB=<path of paddle root>/build-env/paddle_inference_install_dir
$ make
```

## Download the pretrained model and the preprocessed dataset
* Pretrained model: [Google drive download link](https://drive.google.com/file/d/1eZEsxWQInqHEx8GpLH_gJGPB5bY4r6oe/view?usp=sharing)
* preprocessed dataset: [Google driver download link](https://drive.google.com/file/d/1iWNrse6N2U3o5nwfQ7IVDIBMit6TYtqf/view?usp=sharing)

After downloading the above compressed file, uncompress them under the ernie-varlen directory
```bash
$ tar -xf models.tar.xz
$ tar -xf data.tar.xz
```
After compressing the downloaded data, the ernie-varlen directory should be like
```bash
.
├── data
│   └── QNLI
├── logs
├── models
│   └── QNLI-base-2.0
├── scripts
└── src

```

## Run the benchmark

Edit [run.sh](https://gitlab-master.nvidia.com/rewang/ernie-varlen/-/blob/master/run.sh), and append the TensorRT, cudnn library path into the `LD_LIBRARY_PATH`. See run.sh for details

```
bash run.sh
```

## Other details of this benchmark

### Benchmark layout
* `src/inference.cc` is the core of the benchmark, it does following things: read the input/create the predictor/run the benchmark/output the prediction.
* `script/cbenchmark.py` is an python interface to use the `src/inference.cc`, and `script/cbenchmark.py` also validates the prediction accuracy.
* `run.sh` is used to set different environment variables and run the benchmark

### Input changes
Following is the old input for the PaddlePredictor. (B = batch size, S = max sequence length of the batch)
1. input embedding idx, shape=[B, S], data type = int64_t
2. sent type, shape = [B, S], data type = int64_t
3. position, shape = [B, S], data type = int64_t
4. mask, shape = [B, S], data type = float32

The new input is changed.
1. input embedding idx, shape=[**sum of (S)**], data type = **int32_t**. **sum of (S)** means the total sequence length of the batch
2. sent type, shape=[**sum of (S)**], data type = **int32_t**
3. cumulative sequence length, shape = [**B + 1**], data type = **in32_t**, 
   i.e. four sequences: AAAA, BB, CCC, D. We should feed: [0, 4, 6, 9, 10]
4. dummy input, shape = [**max of (S)**], data type = **int32_t**, the shape is the max sequence length of the batch, the value can be any integer
   i.e. four sequences: AAAA, BB, CCC, D. We should feed: [0, 0, 0, 0]

## Performance on T4
Here are some numbers for the reference
* Dataset: QNLI (5463 sequences)
* GPU: NVIDIA T4 (default clock)
* Batch size: 32
* Inference mode: FCFS

| Implemenation           | Sents/s |
|-------------------------|---------|
| Past (w/ padding)       | ~800    |
| This repo (w/o padding) | ~2100   |
