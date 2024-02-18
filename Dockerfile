FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
ENV LANG C.UTF-8
ENV LANGUAGE en_US:
ENV SHELL bash

RUN apt-get update && apt-get install -y \
    git

# neural a-astar dependency
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get -y install --no-install-recommends software-properties-common libgl1-mesa-dev wget libssl-dev

RUN pip3 install -U pip distlib setuptools wheel