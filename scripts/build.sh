set -xe
docker build --tag=$USER/ernie-benchmark -f Dockerfile \
    --build-arg uid=$(id -u) --build-arg username=$(id -un) \
    --build-arg gid=$(id -g) --build-arg groupname=$(id -gn) .
