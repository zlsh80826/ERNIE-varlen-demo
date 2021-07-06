FROM nvcr.io/nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04

ARG uid
ARG gid
ARG username
ARG groupname
ARG TRT_VERSION=8.0.1.6

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        git \
        cmake \
        patchelf \
        python3-dev \
        unzip \
        gcc-8 \
        g++-8 \
        libgl1 \
        libssl-dev \
        aria2 \
        vim \
        python3-venv \
        sudo \
        python3-pip

RUN addgroup --gid ${uid} ${username}
RUN groupmod --gid 3000 dip
# RUN addgroup --gid ${gid} ${groupname}
RUN adduser --disabled-password --gecos GECOS -u ${uid} -gid ${gid} ${username}
RUN adduser ${username} sudo
RUN usermod -a -G ${groupname} ${username}
RUN usermod -a -G ${groupname} root
RUN usermod -a -G root ${username}
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir /workspace
RUN chown ${username}:${groupname} /workspace
USER ${username}

# Install TensorRT
RUN mkdir /workspace/tools
WORKDIR /workspace/tools
# RUN aria2c -x 4 'http://cuda-repo/release-candidates/Libraries/TensorRT/v8.0/8.0.1.4-9e96bdec/11.3-r465/Ubuntu18_04-x64/tar/TensorRT-8.0.1.4.Ubuntu-18.04.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz'
COPY TensorRT-8.*-11.3.cudnn8.2.tar.gz .
RUN tar xf TensorRT-8.*-11.3.cudnn8.2.tar.gz

# Install Paddle
WORKDIR /workspace
RUN git clone https://github.com/zlsh80826/Paddle.git --branch tensorrt8-sparsity
RUN mkdir -p Paddle/build-env
WORKDIR /workspace/Paddle/build-env
RUN cmake .. -DWITH_PYTHON=OFF \
         -DWITH_GPU=ON \
         -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_ARCH_NAME=All \
         -DWITH_MKL=OFF \
         -DON_INFER=ON \
         -DWITH_TESTING=OFF \
         -DWITH_INFERENCE_API_TEST=OFF \
         -DWITH_TENSORRT=ON \
         -DTENSORRT_ROOT=/workspace/tools/TensorRT-${TRT_VERSION} \
         -DCMAKE_C_COMPILER=`which gcc-8` -DCMAKE_CXX_COMPILER=`which g++-8`
RUN make -j`nproc`

WORKDIR /workspace

# Install ERNIE-varlen-demo
RUN git clone 'https://github.com/zlsh80826/ERNIE-varlen-demo.git' --branch sparsity
WORKDIR /workspace/ERNIE-varlen-demo
RUN mkdir build && cd build && \
    cmake ../src -DFLUID_INFER_LIB=/workspace/Paddle/build-env/paddle_inference_install_dir && \
    make -j
RUN mkdir logs
ENV LD_LIBRARY_PATH="/workspace/tools/TensorRT-${TRT_VERSION}/lib:${LD_LIBRARY_PATH}"
RUN pip3 install pandas jinja2 --user
