FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive

RUN apt update && apt-get install -y git ffmpeg libsm6 libxext6 

# install torch-cluster
RUN pip install https://data.pyg.org/whl/torch-1.13.0%2Bcu116/torch_cluster-1.6.0%2Bpt113cu116-cp39-cp39-linux_x86_64.whl

# install pytorch3d
RUN pip install fvcore iopath && pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/download.html

# install all other dependencies
RUN pip install trimesh pyyaml pandas easydict plotly pymeshlab==0.1.8 jupyter scikit-image
RUN pip install matplotlib pytorch-lightning plyfile

# install torch-batch-svd
RUN mkdir /user-app
COPY torch-batch-svd /user-app/torch-batch-svd
WORKDIR /user-app/torch-batch-svd
RUN python setup.py install

COPY FRNN /user-app/FRNN
COPY setup.sh /user-app/setup.sh
# WORKDIR /user-app/
# RUN bash setup.sh

WORKDIR /workspace

# # cleanup
# RUN rm -rf /user-app

# # install FRNN
# WORKDIR /user-app
# RUN git clone --recursive https://github.com/lxxue/FRNN.git
# WORKDIR /user-app/FRNN/external/prefix_sum
# RUN pip install .
# WORKDIR /user-app/FRNN
# RUN pip install -e .

# docker build -t complex_sar .