#!/bin/bash

MOUNT_OPS="-v $(pwd):/workspace"
OPTS=""

IMAGE="complex_sar"
REMOTE_IMAGE="agilemelon/complex_sar:torch113-cuda118"


if [ $# -eq 0 ]; then
    docker run -it --rm --gpus all $MOUNT_OPS $OPTS $REMOTE_IMAGE
elif [ "$1" = "local" ]; then
    docker run -it --rm --gpus all $MOUNT_OPS $OPTS $IMAGE
elif [ "$1" = "cpu" ]; then
    docker run -it --rm $MOUNT_OPS $OPTS $IMAGE
elif [ "$1" = "build" ]; then
    docker build -t $IMAGE -f dockers/Dockerfile dockers
else
    echo "Unknown argument: $1"
fi