docker run --gpus=all -it --rm --net=host \
           -e NVIDIA_VISIBLE_DEVICES=$NV_GPU \
           --ipc=host --cap-add=SYS_PTRACE --cap-add SYS_ADMIN \
           --cap-add DAC_READ_SEARCH --security-opt seccomp=unconfined \
           -v `pwd`/data:/workspace/ERNIE-varlen-demo/data \
           -v `pwd`/models:/workspace/ERNIE-varlen-demo/models \
           $USER/ernie-benchmark /bin/bash
