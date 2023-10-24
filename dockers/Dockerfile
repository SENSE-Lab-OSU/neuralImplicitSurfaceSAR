FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive

# famous nvidia key rotation
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

# install utils
RUN apt update && apt-get install -y git ffmpeg libsm6 libxext6 curl

# install torch-cluster
RUN pip install https://data.pyg.org/whl/torch-1.6.0%2Bcu101/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl

RUN mkdir /user-app
WORKDIR /user-app

# install torch-batch-svd
COPY torch-batch-svd /user-app/torch-batch-svd
WORKDIR /user-app/torch-batch-svd
RUN python setup.py install

# prep and install pytorch3d
RUN conda update -n base conda
RUN conda install -n base conda-libmamba-solver
RUN conda config --set solver libmamba
RUN conda install -n base -c fvcore -c iopath -c conda-forge fvcore iopath
RUN conda install -n base pytorch3d -c pytorch3d
# ignored: only needed if build from source
# RUN curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
# RUN tar xzf 1.10.0.tar.gz
# ENV CUB_HOME=$PWD/cub-1.10.0

# prep for FRNN
COPY FRNN /user-app/FRNN
COPY setup.sh /user-app/setup.sh

# install all other dependencies
RUN pip install trimesh pyyaml pandas easydict plotly pymeshlab==0.1.8 jupyter scikit-image matplotlib pytorch-lightning==1.2 plyfile h5py tensorboard

WORKDIR /workspace

