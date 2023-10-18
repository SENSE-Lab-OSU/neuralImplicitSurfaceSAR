#!/bin/bash

# install FRNN
cd /user-app/FRNN/external/prefix_sum && pip install .
cd /user-app/FRNN && pip install -e .

# # install torch-batch-svd
# cd /workspace/torch-batch-svd && python setup.py install