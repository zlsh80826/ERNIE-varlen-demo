# PaddlePaddle/ERNIE var-len demo

## Features

* TensorRT 7.2 provides new plugins for var-len BERT.
* TensorRT 8.0 supports [NVIDIA sparse tensor core](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt).

This repo demos the ability of above powerful features.

## Quick Start

### Clone the repo
``` bash
$ git clone https://github.com/zlsh80826/ERNIE-varlen-demo.git
$ git checkout sparsity
$ cd ernie-varlen-benchmark
```

### Download the models and data

Download the [models](https://drive.google.com/file/d/1RJeWVfbsXRt6a8gMb86zuhCty0GJ5biK/view?usp=sharing) and [data](https://drive.google.com/file/d/1Q_SOngP1qMGt7j5nJvmaRxEQDufrwugm/view?usp=sharing) through the links.

After downloading, extract them under this repo directory.

```bash
tar xf models.tar.xz
tar xf data.tar.xz
```

Then your directory will be like
```bash
$ tree .
.
├── benchmark-mig.sh
├── benchmark.sh
├── data
├── Dockerfile
├── models
├── README.md
├── scripts
│   ├── build.sh
│   ├── cbenchmark.py
│   ├── launch.sh
│   └── utils.py
└── src
    ├── CMakeLists.txt
    └── inference.cu
```

### Build the image
The image builds [PaddlePaddle](https://github.com/zlsh80826/Paddle/tree/tensorrt8-sparsity) and [ERNIE-varlen-demo](https://github.com/zlsh80826/ERNIE-varlen-demo/tree/sparsity). The building step may take 30-90 minutes depend on the CPU model. 
If your system memory is lower than 32GB, please modify the [Dockerfile](Dockerfile) for using less threads to compile Paddle (default use ``nproc``).
If you are going to benchmark with MIG, please configure the MIG before executing [launch.sh](scripts/launch.sh)
```
$ bash scripts/build.sh
$ bash scripts/launch.sh
$ # enter the container
```

### Benchmark
After entering the container, you can start the benchmark by
```bash
$ bash benchmark.sh
```

### Benchmark with MIG
If you are going to benchmark with MIG-ALL (run benchmark simultaneously on all mig), please enter the container before enabling and configuring the MIG.
The MIG-ALL benchmark has two parts. The first part executes the normal benchmark and generate the serialized trt engine file for second part. 
The second part then read the generated trt engine file to run benchmark on each mig to simulate the performance after enabling the mig.
```bash
$ bash benchmark.sh
$ bash benchmark-mig.sh
```

## Benchmark Results

### Notes

1. The following results were obtained on Intel(R) Xeon(R) Silver 4210R CPU with performance mode. The GPU frequency was set on default clock.
``` bash
sudo cpupower frequency-set -g performance
sudo nvidia-smi -rac
sudo nvidia-smi -rgc
```

2. The MIG setting on A100 was to split the GPU to 7 instances. The results on the following tables were obtained on one of the instances.
``` bash
sudo nvidia-smi -mig 1
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -C
```

3. The MIG setting on A30 was to split the GPU to 4 instances. The results on the following tables were obtained on one of the instances.
``` bash
sudo nvidia-smi -mig 1
sudo nvidia-smi mig -cgi 14,14,14,14 -C
```

### Sentences/second on dense weight 

| Batch Size  | 1    | 2    | 4    | 8    | 16    | 32    | 64    | 128    | 256    |
|:------------|:-----|------|------|------|-------|-------|-------|--------|-------:|
| A10         | 184  | 333  | 522  | 833  | 1014  | 1399  | 1505  | 1618   | 1660   |
| A30         | 228  | 392  | 675  | 932  | 1735  | 2061  | 2304  | 2454   | 2549   |
| A30-MIG     | 105  | 139  | 260  | 367  | 496   | 622   | 687   | 733    | 761    |
| A30-MIG-ALL | 420  | 556  | 1024 | 1415 | 1839  | 2213  | 2394  | 2475   | 2549   |
| A100        | 297  | 532  | 909  | 1311 | 2621  | 3346  | 3969  | 4245   | 4485   |
| A100-MIG    | 93   | 136  | 284  | 363  | 491   | 602   | 662   | 701    | 726    |

### Sentences/second on sparse weight 

| Batch Size  | 1    | 2    | 4    | 8    | 16    | 32    | 64    | 128    | 256    |
|:------------|:-----|------|------|------|-------|-------|-------|--------|-------:|
| A10         | 243  | 468  | 812  | 1267 | 1703  | 1885  | 2060  | 2136   | 2192   |
| A30         | 266  | 476  | 883  | 1407 | 2174  | 2740  | 3061  | 3182   | 3379   |
| A30-MIG     | 134  | 233  | 410  | 578  | 756   | 863   | 932   | 993    | 1026   |
| A30-MIG-ALL | 533  | 927  | 1621 | 2235 | 2834  | 3136  | 3362  | 3501   | 3603   |
| A100        | 307  | 595  | 1061 | 1869 | 3161  | 4152  | 5383  | 5786   | 5973   |
| A100-MIG    | 119  | 215  | 384  | 572  | 745   | 844   | 933   | 987    | 1019   |

### Sparse weight speedup

| Batch Size  | 1    | 2    | 4    | 8    | 16    | 32    | 64    | 128    | 256    |
|:------------|:-----|------|------|------|-------|-------|-------|--------|-------:|
| A10         | 1.32 | 1.41 | 1.56 | 1.52 | 1.68  | 1.35  | 1.37  | 1.32   | 1.32   |
| A30         | 1.17 | 1.21 | 1.54 | 1.51 | 1.25  | 1.33  | 1.33  | 1.30   | 1.33   |
| A30-MIG     | 1.58 | 1.68 | 1.58 | 1.58 | 1.52  | 1.39  | 1.36  | 1.35   | 1.35   |
| A30-MIG-ALL | 1.27 | 1.67 | 1.58 | 1.58 | 1.54  | 1.41  | 1.40  | 1.41   | 1.41   |
| A100        | 1.03 | 1.12 | 1.17 | 1.43 | 1.21  | 1.24  | 1.36  | 1.36   | 1.33   |
| A100-MIG    | 1.28 | 1.58 | 1.35 | 1.58 | 1.51  | 1.40  | 1.41  | 1.41   | 1.40   |
