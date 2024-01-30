#!/bin/bash

EXEC_PATH=`pwd`
COMMAND=bash

docker run -it --gpus all \
               -w ${EXEC_PATH} \
               --shm-size=2gb \
               -h bn_docker \
               -v /etc/group:/etc/group:ro \
               -v /etc/passwd:/etc/passwd:ro \
               -v /home/$USER/:/home/$USER \
               -v /mnt:/mnt \
               -u $(id -u $USER):$(id -g $USER) bottlenome/bracket_net:latest ${COMMAND}
