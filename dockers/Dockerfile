FROM nvcr.io/nvidia/pytorch:22.11-py3 

ENV DEBIAN_FRONTEND noninteractive

# ARG USERNAME=app
# ARG GRPNAME=app
# ARG USER_UID=1000
# ARG USER_GID=1000

# # create user
# RUN groupadd --gid ${USER_GID} ${GRPNAME} \
#     && useradd --uid ${USER_UID} -s /bin/bash --gid ${USER_GID} -m ${USERNAME} \
#     && usermod -aG root ${USERNAME}

# # # install pytorch3d
# # to get the correct whl, run
# # import sys
# # import torch
# # pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
# # version_str="".join([
# #     f"py3{sys.version_info.minor}_cu",
# #     torch.version.cuda.replace(".",""),
# #     f"_pyt{pyt_version_str}"
# # ])
# # print(version_str)
# RUN pip install fvcore iopath && pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt1131/download.html

# # install torch_cluster
# RUN pip install https://data.pyg.org/whl/torch-1.4.0%2Bcu101/torch_cluster-1.5.2-cp36-cp36m-linux_x86_64.whl

# # other dependencies
# RUN pip install trimesh pyyaml pandas easydict plotly pymeshlab==0.1.8 jupyter scikit-image

##
# make directory for locally compiled dependencies
RUN mkdir /user-app

# install FRNN
WORKDIR /user-app
RUN git clone --recursive https://github.com/lxxue/FRNN.git
WORKDIR /user-app/FRNN/external/prefix_sum
RUN pip install .
WORKDIR /user-app/FRNN
RUN pip install -e .

# docker run --rm -it --ipc=host --gpus all -v $(pwd):/workspace nvcr.io/nvidia/pytorch:22.11-py3