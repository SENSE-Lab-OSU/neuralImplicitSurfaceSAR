
#%%% import 
%matplotlib widget
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
from DSS.models.common import Siren, SDF
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
import matplotlib.pyplot as plt
import frnn
#%% shape object 
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
            proj_max_iters=10, proj_tolerance=1e-5, total_iters=1, sample_iters=5, knn_k=16)
        self.ear_projection = EdgeAwareProjection(proj_max_iters=10, knn_k=16,
            proj_tolerance=1e-5, total_iters=1, resampling_clip=0.02, sample_iters=2, repulsion_mu=0.4,
            sharpness_angle=20, edge_sensitivity=1.0,upsample_ratio=2.5)
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

#%%% load point-cloud        
device = torch.device('cuda:0')

points, normals = np.split(
        read_ply("data/DTU_furu/parkingLot_full.ply").astype('float32'), (3,), axis=1)
pmax, pmin = points.max(axis=0), points.min(axis=0)
scale = (pmax - pmin).max()
pcenter = (pmax + pmin) /2
points = (points - pcenter) / scale * 1.5

gt_surface_pts_all = torch.from_numpy(points).unsqueeze(0).float().to(device=device)
gt_surface_normals_all = torch.from_numpy(normals).unsqueeze(0).float()
gt_surface_normals_all = F.normalize(gt_surface_normals_all, dim=-1)

shape = Shape(gt_surface_pts_all.cuda(), n_points=gt_surface_pts_all.shape[1]*10, normals=gt_surface_normals_all.cuda())
iso_points = shape.points


idx = torch.randperm(gt_surface_pts_all.shape[1]).to(device=gt_surface_pts_all.device)[:(gt_surface_pts_all.shape[1])]
tmp = torch.index_select(gt_surface_pts_all, 1, idx)
space_pts = torch.cat(
                [torch.rand_like(tmp.repeat(1,5,1)) * 2 - 1,
                 torch.randn_like(tmp, device=tmp.device, dtype=tmp.dtype) * 0.1+tmp], dim=1)

#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
iso_points_np = iso_points[0,:,:].cpu().numpy()
ax.scatter( iso_points_np[:,0],  iso_points_np[:,1],  iso_points_np[:,2])

ax.scatter(points[:,0], points[:,1], points[:,2])
plt.show()


#%%% plot figures
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2])

ax.scatter(space_pts[0,:,0], space_pts[0,:,1], space_pts[0,:,2])
plt.show()
# %%
