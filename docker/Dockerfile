FROM pytorchlightning/pytorch_lightning:2.0.9.post0-py3.9-torch1.13-cuda11.8.0

RUN mkdir /workspace
WORKDIR /workspace


# install pytorch3d
RUN pip install fvcore iopath && pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt1131/download.html

# install torch_cluster
RUN pip install https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_cluster-1.6.0%2Bpt113cu117-cp39-cp39-linux_x86_64.whl

# other dependencies
RUN pip install torchvision==0.14.1 trimesh pyyaml pandas easydict plotly pymeshlab==0.1.8 jupyter
