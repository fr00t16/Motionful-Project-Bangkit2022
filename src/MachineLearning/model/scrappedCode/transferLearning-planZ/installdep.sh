#!/bin/bash
apt update
apt upgrade -y
apt install -y python3-pip python3 fish git python3-dev python3-venv
pip3 install scipy numpy scikit-image pillow pyyaml matplotlib cython tensorflow easydict munkres tf_slim tk