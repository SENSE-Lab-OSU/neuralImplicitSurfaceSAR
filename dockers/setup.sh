#!/bin/bash

# pytorch3d
# pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# FRNN
cd /user-app/FRNN/external/prefix_sum && pip install .
cd /user-app/FRNN/ && pip install -e .
