# %% import packages 
from DSS.core.cloud import PointClouds3D
import os
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import argparse
from collections import defaultdict
from DSS.utils.checkpointIO import CheckpointIO
from DSS.models.common import Siren, SDF,SDF_feature,RenderingNetwork,CombinedModel
from DSS.models.levelset_sampling import UniformProjection, EdgeAwareProjection
from pytorch3d.ops import knn_points, knn_gather
from DSS.utils import tolerating_collate, get_surface_high_res_mesh, scaler_to_color, valid_value_mask
from DSS.utils.point_processing import resample_uniformly, denoise_normals
from DSS.utils.io import save_ply, read_ply
from DSS.utils.mathHelper import eps_sqrt, to_homogen, estimate_pointcloud_normals, pinverse
from DSS.misc.visualize import plot_cuts
from DSS.training.losses import NormalLengthLoss
from DSS import set_deterministic_
import plotly.graph_objs as go
import frnn
from torch.utils.tensorboard import SummaryWriter
import h5py

set_deterministic_()


# %% Support classes and functions
class FooAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest+'_nondefault', True)

def load_data_cvdomes(filepath):
    with h5py.File(filepath) as f:
    # read the complex-valued amplitudes used in the image
        data_of_interest = np.array(f['amps'][:])
        amps = np.concatenate((data_of_interest['real'],data_of_interest['imag']),0)
    # read the complex-valued amplitudes used in the image
        data_of_interest = np.array(f['normalsSave'][:])
        normals = data_of_interest
     # read the 3D point coordinatew used in the image
        data_of_interest = np.array(f['points'][:],dtype=np.float32)
        points = data_of_interest  
    # read the view vector for radar 
        data_of_interest = np.array(f['viewVector'][:],dtype=np.float32)
        viewVector = data_of_interest   
    return amps,points,normals,viewVector


class Shape(object):
    """
    docstring
    """
    pass

    def __init__(self, points, n_points, normals=None):
        super().__init__()
        B, P, _ = points.shape
        assert(B==1)
        self.projection = UniformProjection(
            proj_max_iters=10, proj_tolerance=1e-5, total_iters=10, sample_iters=5, knn_k=16)
        self.ear_projection = EdgeAwareProjection(proj_max_iters=10, knn_k=16,
            proj_tolerance=1e-5, total_iters=10, resampling_clip=0.02, sample_iters=2, repulsion_mu=0.4,
            sharpness_angle=20, edge_sensitivity=1.0,upsample_ratio=3.5)
        rnd_idx = torch.randperm(P)[:n_points]
        points = points.view(-1, 3)[rnd_idx].view(1, -1, 3)
        if normals is not None:
            normals = normals.view(-1, 3)[rnd_idx].view(1, -1, 3)
        self.points = resample_uniformly(PointClouds3D(points, normals=normals), shrink_ratio=0.25, repulsion_mu=0.65, neighborhood_size=31).points_padded()

    def get_iso_points(self, points, sdf_net, ear=False, outlier_tolerance=0.01):
        if not ear:
            projection = self.projection
        else:
            # first resample uniformly
            projection = self.ear_projection
        with autograd.no_grad():
            proj_results = projection.project_points(points.view(1, -1, 3), sdf_net)
            mask_iso = proj_results['mask'].view(1, -1)
            iso_points = proj_results['levelset_points'].view(1, -1, 3)
            iso_points = iso_points[mask_iso].view(1, -1, 3)
            # iso_points = remove_outliers(iso_points, tolerance=outlier_tolerance, neighborhood_size=31).points_padded()
            return iso_points

def get_iso_bilateral_weights(points, normals, iso_points, iso_normals):
    """ find closest iso point, compute bilateral weight """
    search_radius = 0.1
    dim = iso_points.view(-1,3).norm(dim=-1).max()*2
    avg_spacing = iso_points.shape[1] / dim / 16
    dists, idxs, nn, _ = frnn.frnn_grid_points(
            points, iso_points, K=1,
            return_nn=True, grid=None, r=search_radius)
    iso_normals = F.normalize(iso_normals, dim=-1)
    iso_normals = frnn.frnn_gather(iso_normals, idxs).view(1, -1, 3)
    dists = torch.sum((nn.view_as(points) - points)*iso_normals,dim=-1)**2
    # dists[idxs<0] = 10 * search_radius **2
    # dists = dists.squeeze(-1)
    spatial_w = torch.exp(-dists*avg_spacing)
    normals = F.normalize(normals, dim=-1)
    normal_w = torch.exp(-((1-torch.sum(normals * iso_normals, dim=-1))/(1-np.cos(np.deg2rad(60))))**2)
    weight = spatial_w * normal_w
    weight[idxs.view_as(weight)<0] = 0
    if not valid_value_mask(weight).all():
        print("Illegal weights")
        breakpoint()
    return weight



def get_laplacian_weights(points, normals, iso_points, iso_normals, neighborhood_size=8):
    """
    compute distance based on iso local neighborhood
    """
    with autograd.no_grad():
        P, _ = points.view(-1, 3).shape
        search_radius = 0.15
        dim = iso_points.view(-1,3).norm(dim=-1).max()*2
        avg_spacing = iso_points.shape[1] / dim / 16
        dists, idxs, nn, _ = frnn.frnn_grid_points(
                points, iso_points, K=1,
                return_nn=True, grid=None, r=search_radius)
        nn_normals = frnn.frnn_gather(iso_normals, idxs)
        dists = torch.sum((points - nn.view_as(points))*(normals + nn_normals.view_as(normals)), dim=-1)
        dists = dists * dists
        spatial_w = torch.exp(-dists*avg_spacing)
        spatial_w[idxs.view_as(spatial_w)<0] = 0
    return spatial_w.view(points.shape[:-1])

def get_heat_kernel_weights(points, normals, iso_points, iso_normals, neighborhood_size=8, sigma_p=0.4, sigma_n=0.7):
    """
    find closest k points, compute point2face distance, and normal distance
    """
    P, _ = points.view(-1, 3).shape
    search_radius = 0.15
    dim = iso_points.view(-1,3).norm(dim=-1).max()
    avg_spacing = iso_points.shape[1] / (dim*2**2) / 16
    dists, idxs, nn, _ = frnn.frnn_grid_points(
            points, iso_points, K=neighborhood_size,
            return_nn=True, grid=None, r=search_radius)

    # features
    with autograd.no_grad():
        # normalize just to be sure
        iso_normals = F.normalize(iso_normals, dim=-1, eps=1e-15)
        normals = F.normalize(normals, dim=-1, eps=1e-15)

        # features are composite of points and normals
        features = torch.cat([points / sigma_p, normals / sigma_n], dim=-1)
        features_iso = torch.cat([iso_points / sigma_p, iso_normals / sigma_n], dim=-1)

        # compute kernels (N,P,K) k(x,xi), xi \in Neighbor(x)
        knn_idx = idxs
        # features_nb = knn_gather(features_iso, knn_idx)
        features_nb = frnn.frnn_gather(features_iso, knn_idx)
        # (N,P,K,D)
        features_diff = features.unsqueeze(2) - features_nb
        features_dist = torch.sum(features_diff**2, dim=-1)
        kernels = torch.exp(-features_dist)
        kernels[knn_idx < 0] = 0

        # N,P,K,K,D
        features_diff_ij = features_nb[:, :, :,
                                        None, :] - features_nb[:, :, None, :, :]
        features_dist_ij = torch.sum(features_diff_ij**2, dim=-1)
        kernel_matrices = torch.exp(-features_dist_ij)
        kernel_matrices[knn_idx < 0] = 0
        kernel_matrices[knn_idx.unsqueeze(-2).expand_as(kernel_matrices) < 0]
        kernel_matrices_inv = pinverse(kernel_matrices)

        weight = kernels.unsqueeze(-2) @ kernel_matrices_inv @ kernels.unsqueeze(-1)
        weight.clamp_max_(1.0)

    return weight.view(points.shape[:-1])

def gradient(points, net):
    points.requires_grad_(True)
    sdf_value = net(points)
    grad = torch.autograd.grad(sdf_value, [points], [
        torch.ones_like(sdf_value)], create_graph=True)[0]
    return grad

# %% Main script test
device = torch.device('cuda:0') #currently using GPU-0
writer = SummaryWriter() # plotting loss functions on tensorboard


# data
pointcloud_path='/research/nfs_ertin_1/nithin_data/3D_SAR/cvdomes_8pass_results/Camry.mat'
amps,points,normals,viewVector = load_data_cvdomes(pointcloud_path)
amps = amps.T
   
# %% bounding the points and recomputing the normals
pmax, pmin = points.max(axis=0), points.min(axis=0)
scale = (pmax - pmin).max()
pcenter = (pmax + pmin) /2
points = (points - pcenter) / scale * 1.5
scale_mat = scale_mat_inv = np.identity(4)
scale_mat[[0,1,2], [0,1,2]] = 1/scale * 1.5
scale_mat[[0,1,2], [3,3,3]] = - pcenter / scale * 1.5
scale_mat_inv = np.linalg.inv(scale_mat)
normals = normals @ np.linalg.inv(scale_mat[:3, :3].T)
object_bounding_sphere = np.linalg.norm(points, axis=1).max()
# %% data loader
assert(np.abs(points).max() < 1)
combinredInput = np.concatenate((points,viewVector),1)
combinredOutput = np.concatenate((normals,amps),1)

dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(combinredInput), torch.from_numpy(combinredOutput))
# The batch-size is kept as total number of points /3. 
# This should be reduced for larger scenes to fit in the GPU memory
batch_size = 512 #points.shape[0]//3
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, num_workers=1, shuffle=True,
    collate_fn=tolerating_collate,
)
# %% all the points in the dataset
gt_surface_pts_all = torch.from_numpy(points).unsqueeze(0).float()
gt_viewingVectors_all = torch.from_numpy(viewVector).unsqueeze(0).float()
gt_amps_all = torch.from_numpy(amps).unsqueeze(0)
gt_surface_normals_all = torch.from_numpy(normals).unsqueeze(0).float()
gt_surface_normals_all = F.normalize(gt_surface_normals_all, dim=-1)
# %% parameters
num_frequencies_SDF = 6
decoder_params = {
    'dim': 3,
    "out_dims": {'sdf': 1},
    "c_dim": 256,
    "hidden_size": 512,
    'n_layers': 9,
    'bias': 1.0,
    "num_frequencies": num_frequencies_SDF,
}
scatteringNet_params = {
    'dim': 9,
    "out_dims": {'complex': 2},
    "c_dim": 256,
    "hidden_size": 512,
    'n_layers': 4,
    "num_frequencies": 4,
}
decoder1 = SDF(**decoder_params)

decoder2 = SDF_feature(**decoder_params)
renderer = RenderingNetwork(**scatteringNet_params)
combinedModel1 = CombinedModel(decoder_params,scatteringNet_params)
print(decoder1)

print(decoder1) 
print(renderer)



sub_idx = torch.randperm(gt_surface_normals_all.shape[1])[:20000]
gt_surface_pts_sub = torch.index_select(gt_surface_pts_all, 1, sub_idx).to(device=device)
gt_surface_normals_sub = torch.index_select(gt_surface_normals_all, 1, sub_idx).to(device=device)
gt_surface_normals_sub = denoise_normals(gt_surface_pts_sub, gt_surface_normals_sub, neighborhood_size=30)
gt_amps_sub = torch.index_select(gt_amps_all, 1, sub_idx).to(device=device)
gt_viewingVectors_sub = torch.index_select(gt_viewingVectors_all, 1, sub_idx).to(device=device)

decoder1 = decoder1.to(device)
decoder2 = decoder2.to(device)
combinedModel1 = combinedModel1.to(device)
#print(decoder1(gt_surface_pts_sub[0,0,:]).sdf)
#print(decoder2(gt_surface_pts_sub[0,0,:]).sdf)
# %% loading  a batch
iterloader = iter(data_loader)
batch = next(iterloader)

gt_combinedInput, gt_combinedOutput = batch
gt_combinedInput.unsqueeze_(0)
gt_combinedOutput.unsqueeze_(0)
gt_combinedInput = gt_combinedInput.to(device=device).detach()
gt_combinedOutput = gt_combinedOutput.to(device=device).detach()

gt_surface_pts,gt_viewVector = torch.split(gt_combinedInput,3,dim=-1)
gt_surface_normals,gt_surface_amps = torch.split(gt_combinedOutput,3,dim=-1)


# %% testing gradient and normal 
x= combinedModel1.sdf_feature(gt_surface_pts)

# x = decoder2(gt_surface_pts).sdf
# y = decoder2(gt_surface_pts).spatial_feature

# %%
box_size = (object_bounding_sphere * 2 + 0.2, ) * 3
imgs = plot_cuts(lambda x: combinedModel1.sdf_feature(x).sdf.squeeze().detach(),
            box_size=box_size, max_n_eval_pts=10000, thres=0.0,
            imgs_per_cut=1, save_path='/research/nfs_ertin_1/nithin_data/test.png')
mesh = get_surface_high_res_mesh( lambda x: combinedModel1.sdf_feature(x).sdf.squeeze(), resolution=180)
mesh.apply_transform(scale_mat_inv)
#%%