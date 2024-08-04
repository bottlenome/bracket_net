#!/bin/bash -eu

COMMAND=${1:-bash}
GITHUB_ACTIONS=${GITHUB_ACTIONS:-false}

# Determine the execution environment
if [ "$GITHUB_ACTIONS" = "true" ]; then
  EXEC_PATH=${GITHUB_WORKSPACE}
  USER_DIR=${GITHUB_WORKSPACE}
  DOCKER_USER=""
  DOCKER_OPTS=""
else
  EXEC_PATH=`pwd`
  USER_DIR="/home/$USER"
  DOCKER_USER="-u $(id -u $USER):$(id -g $USER)"
  DOCKER_OPTS="-it --gpus all "
fi

docker run ${DOCKER_OPTS} \
           -w ${EXEC_PATH} \
           --shm-size=2gb \
           -h bn_docker \
           -v /etc/group:/etc/group:ro \
           -v /etc/passwd:/etc/passwd:ro \
           -v /data:/data \
           -v ${USER_DIR}:${USER_DIR} \
           -v /mnt:/mnt \
           ${DOCKER_USER} \
           bottlenome/bracket_net:latest ${COMMAND}
