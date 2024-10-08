# Neural Implicit Surface SAR 

## Background information 
We utilize the Signed distance function (SDF) to represent the surface of objects. The SDF network is based on

DeepSDF - Park, Jeong Joon, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. "Deepsdf: Learning continuous signed distance functions for shape representation." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 165-174. 2019.
Multi-view surface reconstruction Yariv, Lior, Yoni Kasten, Dror Moran, Meirav Galun, Matan Atzmon, Basri Ronen, and Yaron Lipman. "Multiview neural surface reconstruction by disentangling geometry and appearance." Advances in Neural Information Processing Systems 33 (2020): 2492-2502.

The surface is iteratively refined and resampled to produce iso-points using resampling methods on the surface representation to get robust surface representation. The iso-points method extracts uniformly sampled points on the surface to get smooth surface representation using the method described in 

Yifan, Wang, Shihao Wu, Cengiz Oztireli, and Olga Sorkine-Hornung. "Iso-points: Optimizing neural implicit surfaces with hybrid representations." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 374-383. 2021.


## Getting started

### From docker

Clone this repo

```bash
git clone --recursive https://github.com/SENSE-Lab-OSU/neuralImplicitSurfaceSAR.git
```

Start a docker container with environment setup

```bash
cd neuralImplicitSurfaceSAR
./run_docker.sh
```


### Manual install
<!-- # Installation instructions -->

````bash 
git clone --recursive https://github.com/SENSE-Lab-OSU/neuralImplicitSurfaceSAR.git

cd neuralImplicitSurfaceSAR

# conda environment and dependencies
# update conda
conda update -n base -c defaults conda
# install requirements
conda env create --name DSS -f environment.yml
conda activate DSS

# setting up the compiler 
conda install https://anaconda.org/brown-data-science/gcc/5.4.0/download/linux-64/gcc-5.4.0-0.tar.bz2
cd ~/miniconda3/envs/DSS/lib
# Find the latest c++ lib
ls ~/miniconda3/envs/DSS/lib | grep libstdc++.so
# backup the libstc++
mv libstdc++.so libstdc++.so.bak
# link the latest file
ln -s libstdc++.so.6.0.32 libstdc++.so.6
# build additional dependencies of DSS
# FRNN - fixed radius nearest neighbors
cd FRNN
git submodule update --init --recursive
cd external/prefix_sum
python setup.py install
cd ../../
python setup.py install

# build batch-svd
cd ../torch-batch-svd
python setup.py install

# build DSS itself
cd ../..
python setup.py develop


# Running the script
python test_SAR_points_script.py data/Camry.ply --lambda_eikonal 15 --lambda_surface_sdf 5000 --lambda_surface_normal 500 --warm_up 500 --lambda_iso_normal 60 --denoise_normal --use_off_normal_loss -d sdf --lambda_iso_sdf 100 --lambda_inter_sdf 100 --numberFrequencyFeatures 10 --totalIteration 30000 --resample_every 2000  -o experiments/camry
