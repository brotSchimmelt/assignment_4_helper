#!/bin/bash
sudo mkdir -p /usr/local/cuda/lib64
sudo ln -s /usr/lib/x86_64-linux-gnu/libcudart.so /usr/local/cuda/lib64/libcudart.so

pip install --upgrade pip
pip install gdown pandas numpy notebook
pip install --upgrade numexpr

jupyter notebook --no-browser --port=8080
