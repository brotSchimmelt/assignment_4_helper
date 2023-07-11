#!/bin/bash
sudo mkdir -p /usr/local/cuda/lib64
sudo ln -s /usr/lib/x86_64-linux-gnu/libcudart.so /usr/local/cuda/lib64/libcudart.so

sudo pip3 install --upgrade pip
sudo pip3 install gdown pandas numpy notebook
sudo pip3 install --upgrade numexpr

jupyter notebook --no-browser --port=8080
