"""
[1] Feature Preserving Point Set Surfaces based on Non-Linear Kernel Regression
Cengiz Oztireli, Gaël Guennebaud, Markus Gross
[2] Consolidation of Unorganized Point Clouds for Surface Reconstruction
Hui Huang, Dan Li, Hao Zhang, Uri Ascher Daniel Cohen-Or
[3] Differentiable Surface Splatting for Point-based Geometry Processing
Wang Yifan, Felice Serena, Shihao Wu, Cengiz Oeztireli, Olga Sorkine-Hornung
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pytorch3d import ops
from DSS.core.cloud import PointClouds3D
from DSS.utils.mathHelper import eps_denom, eps_sqrt, estimate_pointcloud_normals
from DSS import get_debugging_mode, get_debugging_tensor
from DSS import logger_py
from pytorch3d.ops import padded_to_packed
from pytorch3d.ops.knn import _KNN as KNN
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform, MeshRasterizer, RasterizationSettings
from pytorch3d.loss.point_mesh_distance import point_face_distance

class BaseLoss(nn.Module):
    """
    Attributes:
        reduce (str): 'mean' | 'sum' | 'none'
        channel_dim (int): if not None, average this dimension before
            reduction
    """

    def __init__(self, reduction: str = 'mean', channel_dim: int = -1):
        super().__init__()
        self.reduction = reduction
        self.channel_dim = channel_dim
        self.hooks = []

    def compute(self, *args):
        raise NotImplementedError

    def _reduce(self, loss, reduction=None):
        reduction = reduction or self.reduction
        if reduction == 'none':
            return loss
        if reduction == 'sum':
            loss = torch.sum(loss)
        elif reduction == 'mean':
            loss = torch.mean(loss)
        else:
            raise ValueError(
                'Invalid reduction method ({})'.format(self.reduction))
        return loss

    def forward(self, *args, **kwargs):
        reduction = kwargs.get('reduction', self.reduction)
        self.channel_dim = kwargs.get('channel_dim', self.channel_dim)
        loss = self.compute(*args)
        if self.channel_dim is not None:
            loss = torch.sum(loss, dim=self.channel_dim)
        loss = self._reduce(loss, reduction=reduction)
        return loss

    def debug(self, is_debug, **kwargs):
        if is_debug:
            # nothing to do
            pass
        else:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()


class NormalLengthLoss(BaseLoss):
    """enforce normal length to be 1"""

    def __init__(self, reduction: str = 'mean', **kwargs):
        super().__init__(reduction=reduction, channel_dim=None)

    def compute(self, normals):
        assert(normals.shape[-1] == 3)
        loss = (normals.norm(p=2, dim=-1) - 1)**2
        return loss


class NormalLoss(BaseLoss):
    """ Normal same in the neighborhood """
    def __init__(self, reduction: str = 'mean', neighborhood_size=16):
        super().__init__(reduction=reduction, channel_dim=None)
        self.neighborhood_size = neighborhood_size

    def compute(self, pointclouds: PointClouds3D, neighborhood_size=None):
        if neighborhood_size is None:
            neighborhood_size = self.neighborhood_size
        num_points = pointclouds.num_points_per_cloud()
        normals_packed = pointclouds.normals_packed()
        assert(normals_packed is not None)
        normals_ref = estimate_pointcloud_normals(pointclouds, neighborhood_size=neighborhood_size)
        normals_ref = padded_to_packed(
            normals_ref, pointclouds.cloud_to_packed_first_idx(), num_points.sum().item())

        return 1-F.cosine_similarity(normals_packed, normals_ref).abs()


class CosSimilarityLoss(BaseLoss):

    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction=reduction, channel_dim=None)

    def compute(self, vec1, vec2):
        """
        F.cosine_similarity
        """
        return 1 - F.cosine_similarity(vec1, vec2)


class SmapeLoss(BaseLoss):
    """
    relative L1 norm
    http://drz.disneyresearch.com/~jnovak/publications/KPAL/KPAL.pdf eq(2)
    """

    def compute(self, x, y, mask=None, eps=1e-8):
        """ if reduce is true, return a 1-channel tensor, i.e. compute mean over the last dimension """
        lossImg = torch.abs(x - y) / (torch.abs(x) + torch.abs(y) + eps)
        if mask is not None:
            lossImg = lossImg[mask]
        return lossImg


class L1Loss(BaseLoss):
    def compute(self, x, y, weights=None, mask=None):
        lossImg = torch.abs(x - y)
        if weights is not None:
            lossImg = lossImg * weights
        if mask is not None:
            lossImg = lossImg[mask]
        return lossImg

class L2Loss(BaseLoss):
    def compute(self, x, y, weights=None, mask=None):
        lossImg = (x - y)**2
        if weights is not None:
            lossImg = lossImg * weights
        if mask is not None:
            lossImg = lossImg[mask]
        return lossImg

class SurfaceLoss(BaseLoss):
    def __init__(self, reduction='mean', knn_k: int = 33, filter_scale: float = 2.0, sharpness_sigma: float = 0.75):
        super().__init__(reduction=reduction, channel_dim=None)
        self.knn_tree = None
        self.knn_k = knn_k
        self.knn_mask = None
        self.filter_scale = filter_scale
        self.sharpness_sigma = sharpness_sigma

    def _build_knn(self, point_clouds):
        """
        search for KNN again set knn_tree and knn_mask attributes
        TODO(yifan): use a real Kd_tree library to be able to store the data tree and
        query at each forward pass?
        """
        # Find local neighborhood to compute weights
        with torch.autograd.enable_grad():
            points_padded = point_clouds.points_padded()

        lengths = point_clouds.num_points_per_cloud()
        knn_result = ops.knn_points(
            points_padded, points_padded, lengths, lengths, K=self.knn_k, return_nn=True)
        self.knn_mask = torch.full(
            knn_result.idx.shape, False, dtype=torch.bool, device=points_padded.device)
        # valid knn result
        for b in range(self.knn_mask.shape[0]):
            self.knn_mask[b, :lengths[b], :min(
                self.knn_k, lengths[b].item())] = True
            assert(torch.all(knn_result.dists[b][~self.knn_mask[b]] == 0))
        self.knn_tree = KNN(
            knn=knn_result.knn[:, :, 1:, :], dists=knn_result.dists[:, :, 1:], idx=knn_result.idx[:, :, 1:])
        self.knn_mask = self.knn_mask[:, :, 1:]
        assert(self.knn_mask.shape == self.knn_tree.dists.shape)

    def _denoise_normals(self, point_clouds, weights, point_clouds_filter=None):
        """
        robust normal mollification (Sec 4.4), i.e. replace normals with a weighted average
        from neighboring normals
        do this only for invisible points (?)
        Args:
            weights (tensors): (N,max_P,K)
        """
        lengths = point_clouds.num_points_per_cloud()
        P_total = lengths.sum().item()
        normals = point_clouds.normals_padded()
        batch_size, max_P, _ = normals.shape

        knn_normals = ops.knn_gather(normals, self.knn_tree.idx, lengths)
        normals_denoised = torch.sum(knn_normals * weights[:, :, :, None], dim=-2) / \
            eps_denom(torch.sum(weights, dim=-1, keepdim=True))

        # get point visibility so that we update only the non-visible or out-of-mask normals
        if point_clouds_filter is not None:
            try:
                reliable_normals_mask = point_clouds_filter.visibility & point_clouds_filter.inmask
                if len(point_clouds) != reliable_normals_mask.shape[0]:
                    if len(point_clouds) == 1 and reliable_normals_mask.shape[0] > 1:
                        reliable_normals_mask = reliable_normals_mask.any(
                            dim=0, keepdim=True)
                    else:
                        ValueError("Incompatible point clouds {} and mask {}".format(
                            len(point_clouds), reliable_normals_mask.shape))

                # found visibility 0/1 as the last dimension of the features
                # reset visible points normals to its original ones
                normals_denoised[pts_reliable_normals_mask ==
                                 1] = normals[reliable_normals_mask == 1]
            except KeyError as e:
                pass

        normals_packed = point_clouds.normals_packed()
        normals_denoised_packed = ops.padded_to_packed(
            normals_denoised, point_clouds.cloud_to_packed_first_idx(), P_total)
        point_clouds.update_normals_(normals_denoised_packed)
        return point_clouds

    def get_normal_w(self, point_clouds: PointClouds3D, normals: Optional[torch.Tensor] = None, **kwargs):
        """
        Weights exp(-\|n-ni\|^2/sharpness_sigma^2), for i in a local neighborhood
        Args:
            point_clouds: whose normals will be used for ni
            normals (tensor): (N, maxP, 3) padded normals as n, if not provided, use
                the normals from point_clouds
        Returns:
            weight per point per neighbor (N,maxP,K)
        """
        self.sharpness_sigma = kwargs.get(
            'sharpness_sigma', self.sharpness_sigma)
        inv_sigma_normal = 1 / (self.sharpness_sigma * self.sharpness_sigma)
        lengths = point_clouds.num_points_per_cloud()

        if normals is None:
            normals = point_clouds.normals_padded()
        knn_normals = ops.knn_gather(normals, self.knn_tree.idx, lengths)
        normals = torch.nn.functional.normalize(normals, dim=-1)
        knn_normals = torch.nn.functional.normalize(knn_normals, dim=-1)
        w = knn_normals - normals[:, :, None, :]

        w = torch.exp(-torch.sum(w * w, dim=-1) * inv_sigma_normal)
        return w

    def get_spatial_w(self, point_clouds: PointClouds3D, points: Optional[torch.Tensor] = None, ):
        """
        Weights exp(\|p-pi\|^2/sigma^2), h=0.5
        """
        point_spacing_sq = 2.0 / point_clouds.num_points_per_cloud()
        inv_sigma_spatial = 1.0 / point_spacing_sq
        if points is None:
            points = point_clouds.points_padded()
        deltap = self.knn_tree.knn - points[:, :, None, :]
        w = torch.exp(-torch.sum(deltap * deltap, dim=-1) * inv_sigma_spatial)
        return w

    def get_phi(self, point_clouds, **kwargs):
        """
        (1 - \|x-xi\|^2/hi^2)^4
        Return:
            weight per point per neighbor (N, maxP, K) [1] Eq.(12)
        """
        self.knn_tree = kwargs.get('knn_tree', self.knn_tree)
        self.filter_scale = kwargs.get('filter_scale', self.filter_scale)
        point_spacing_sq = self.knn_tree.dists[:, :, :1] * 2
        s = point_spacing_sq * self.filter_scale * self.filter_scale
        w = 1 - self.knn_tree.dists / s
        w[w < 0] = 0
        w = w * w
        w = w * w
        return w


# NOTE(yifan): Essentially an operation that updates point positions from normals.
# Can we formulate this as a pointflow (neural ODE)?
# i.e. we predict normals and integrate with neural ODE? TODO(yifan): think more!
class ProjectionLoss(SurfaceLoss):
    """
    Feature Preserving Point Set Surfaces based on Non-Linear Kernel Regression
    Cengiz Oztireli, Gaël Guennebaud, Markus Gross

    Attributes:
        filter_scale: variance of the low pass filter (default: 2)
        sharpness_sigma: [0.5 (sharp), 2 (smooth)]
    """

    def get_spatial_w(self, point_clouds, **kwargs):
        """
        meshlab implementation skip this step, we do so as well, especially
        since we don't really know what is the SDF function from points
        """
        w = torch.ones_like(self.knn_tree.dists)
        return w

    def compute(self, point_clouds: PointClouds3D, points_filters=None, rebuild_knn=False, **kwargs):
        """
        Args:
            point_clouds
            (optional) knn_tree: output from ops.knn_points excluding the query point itself
            (optional) knn_mask: mask valid knn results
        Returns:
            (P, N)
        """
        self.sharpness_sigma = kwargs.get(
            'sharpness_sigma', self.sharpness_sigma)
        self.filter_scale = kwargs.get('filter_scale', self.filter_scale)
        self.knn_tree = kwargs.get('knn_tree', self.knn_tree)
        self.knn_mask = kwargs.get('knn_mask', self.knn_mask)

        lengths = point_clouds.num_points_per_cloud()
        P_total = lengths.sum().item()
        points = point_clouds.points_padded()
        # - determine phi spatial with using local point spacing (i.e. 2*dist_to_nn)
        # - denoise normals
        # - determine w_normal
        # - mask out values outside ballneighbor i.e. d > filterSpatialScale * localPointSpacing
        # - projected distance dot(ni, x-xi)
        # - multiply and normalize the weights
        with torch.autograd.no_grad():
            if rebuild_knn or self.knn_tree is None or self.knn_tree.idx.shape[:2] != points.shape[:2]:
                self._build_knn(point_clouds)

            phi = self.get_phi(point_clouds, **kwargs)

            # robust normal mollification (Sec 4.4), i.e. replace normals with a weighted average
            # from neighboring normals Eq.(11)
            point_clouds = self._denoise_normals(
                point_clouds, phi, points_filters)

            # compute wn and wr
            # TODO(yifan): visibility weight?
            normal_w = self.get_normal_w(point_clouds, **kwargs)
            spatial_w = self.get_spatial_w(point_clouds, **kwargs)

            # update normals for a second iteration (?) Eq.(10)
            point_clouds = self._denoise_normals(
                point_clouds, phi * normal_w, points_filters)

            # compose weights
            weights = phi * spatial_w * normal_w
            weights[~self.knn_mask] = 0

            # outside filter_scale*local_point_spacing weights
            mask_ball_query = self.knn_tree.dists > (self.filter_scale *
                                                     self.knn_tree.dists[:, :, :1] * 2.0)
            weights[mask_ball_query] = 0.0

            # (B, P, k), dot product distance to surface
            # (we need to gather again because the normals have been changed in the denoising step)
            knn_normals = ops.knn_gather(
                point_clouds.normals_padded(), self.knn_tree.idx, lengths)

        # if points.requires_grad:
        #     from DSS.core.rasterizer import _dbg_tensor

        #     def save_grad(name):
        #         def _save_grad(grad):
        #             _dbg_tensor[name] = grad.detach().cpu()
        #         return _save_grad
        #     points.register_hook(save_grad('proj_grad'))

        dist_to_surface = torch.sum(
            (self.knn_tree.knn.detach() - points.unsqueeze(-2)) * knn_normals, dim=-1)

        if get_debugging_mode():
            # points.requires_grad_(True)

            def save_grad():
                lengths = point_clouds.num_points_per_cloud()

                def _save_grad(grad):
                    dbg_tensor = get_debugging_tensor()
                    if dbg_tensor is None:
                        logger_py.error("dbg_tensor is None")
                    if grad is None:
                        logger_py.error('grad is None')
                    # a dict of list of tensors
                    dbg_tensor.pts_world_grad['proj'] = [
                        grad[b, :lengths[b]].detach().cpu() for b in range(grad.shape[0])]
                return _save_grad

            dbg_tensor = get_debugging_tensor()
            dbg_tensor.pts_world['proj'] = [
                points[b, :lengths[b]].detach().cpu() for b in range(points.shape[0])]
            handle = points.register_hook(save_grad())
            self.hooks.append(handle)

        # convert everything to packed
        weights = ops.padded_to_packed(
            weights, point_clouds.cloud_to_packed_first_idx(), P_total)
        dist_to_surface = ops.padded_to_packed(
            dist_to_surface, point_clouds.cloud_to_packed_first_idx(), P_total)

        # compute weighted signed distance to surface
        dist_to_surface = torch.sum(
            weights * dist_to_surface, dim=-1) / eps_denom(torch.sum(weights, dim=-1))
        loss = dist_to_surface * dist_to_surface
        return loss


class RepulsionLoss(SurfaceLoss):
    """
    Intend to compute the repulsion term in DSS Eq(12)~Eq(15)
    without SVD
    """

    def get_density_w(self, point_clouds: PointClouds3D, points: Optional[torch.Tensor], **kwargs):
        """
        1 + sum_i (exp(-\|x-xi\|^2/(sigma*h)^2))
        """
        inv_sigma_spatial = point_clouds.num_points_per_cloud() / 2.0
        if points is None:
            with torch.autograd.enable_grad():
                points = point_clouds.points_padded()
        deltap = self.knn_tree.knn - points[:, :, None, :]
        w = 1 + torch.sum(torch.exp(-torch.sum(deltap * deltap, dim=-1)
                          * inv_sigma_spatial), dim=-1)
        return w

    def compute(self, point_clouds: PointClouds3D, points_filters=None, rebuild_knn=True, **kwargs):

        self.knn_tree = kwargs.get('knn_tree', self.knn_tree)
        self.knn_mask = kwargs.get('knn_mask', self.knn_mask)

        lengths = point_clouds.num_points_per_cloud()
        P_total = lengths.sum().item()
        points_padded = point_clouds.points_padded()

        # Compute necessary weights to project points to local plane
        # TODO(yifan): This part is same as ProjectionLoss
        # how can we at best save repetitive computation
        with torch.autograd.no_grad():
            if rebuild_knn or self.knn_tree is None or points_padded.shape[:2] != self.knn_tree.shape[:2]:
                self._build_knn(point_clouds)

            phi = self.get_phi(point_clouds, **kwargs)

            self._denoise_normals(point_clouds, phi, points_filters)

            # compute wn and wr
            # TODO(yifan): visibility weight?
            normal_w = self.get_normal_w(point_clouds, **kwargs)

            # update normals for a second iteration (?) Eq.(10)
            point_clouds = self._denoise_normals(
                point_clouds, phi * normal_w, points_filters)

            # compose weights
            weights = phi * normal_w
            weights[~self.knn_mask] = 0

            # outside filter_scale*local_point_spacing weights
            mask_ball_query = self.knn_tree.dists > (self.filter_scale *
                                                     self.knn_tree.dists[:, :, :1] * 2.0)
            weights[mask_ball_query] = 0.0

        # project the point to a local surface
        knn_normals = ops.knn_gather(
            point_clouds.normals_padded(), self.knn_tree.idx, lengths)
        dist_to_surface = torch.sum(
            (self.knn_tree.knn.detach() - points_padded.unsqueeze(-2)) * knn_normals, dim=-1)
        deltap = torch.sum(dist_to_surface[..., None] * weights[..., None]
                           * knn_normals, dim=-2) / eps_denom(torch.sum(weights, dim=-1, keepdim=True))
        points_projected = points_padded + deltap

        if get_debugging_mode():
            # points_padded.requires_grad_(True)

            def save_grad():
                lengths = point_clouds.num_points_per_cloud()

                def _save_grad(grad):
                    dbg_tensor = get_debugging_tensor()
                    if dbg_tensor is None:
                        logger_py.error("dbg_tensor is None")
                    if grad is None:
                        logger_py.error('grad is None')
                    # a dict of list of tensors
                    dbg_tensor.pts_world_grad['repel'] = [
                        grad[b, :lengths[b]].detach().cpu() for b in range(grad.shape[0])]
                return _save_grad

            dbg_tensor = get_debugging_tensor()
            dbg_tensor.pts_world['repel'] = [
                points_padded[b, :lengths[b]].detach().cpu() for b in range(points_padded.shape[0])]
            handle = points_padded.register_hook(save_grad())
            self.hooks.append(handle)

        with torch.autograd.no_grad():
            spatial_w = self.get_spatial_w(point_clouds, points_projected)
            # density_w = self.get_density_w(point_clouds)
            # density weight is actually spatial_w + 1
            density_w = torch.sum(spatial_w, dim=-1, keepdim=True) + 1.0
            weights = normal_w * spatial_w * density_w
            weights[~self.knn_mask] = 0
            weights[mask_ball_query] = 0

        deltap = points_projected[:, :, None, :] - self.knn_tree.knn.detach()
        point_to_point_dist = torch.sum(deltap * deltap, dim=-1)

        # convert everything to packed
        weights = ops.padded_to_packed(
            weights, point_clouds.cloud_to_packed_first_idx(), P_total)
        point_to_point_dist = ops.padded_to_packed(
            point_to_point_dist, point_clouds.cloud_to_packed_first_idx(), P_total)

        # we want to maximize this, so negative sign
        point_to_point_dist = -torch.sum(
            point_to_point_dist * weights, dim=1) / eps_denom(torch.sum(weights, dim=1))
        return point_to_point_dist


class IouLoss(BaseLoss):
    """
    Negative intersection/union,
    reduction applied only the the batch dimension
    """

    def compute(self, predict, target):
        """
        compute negative intersection/union, predict and target are same
        shape tensors [N,...]
        """
        dims = tuple(range(predict.ndimension())[1:])
        intersect = (predict * target).sum(dims)
        union = (predict + target - predict * target).sum(dims)
        result = 1.0 - intersect / eps_denom(union)
        return result


class SignedDistanceLoss(BaseLoss):
    """
    Given ground truth mesh and 3D points, compute the signed distance loss
    """

    def compute(self, points: torch.Tensor, sdf: torch.Tensor, mesh_gt: Meshes):
        """
        Rasterize mesh faces from an far camera facing the origin,
        transform the predicted points position to camera view and project to get the normalized image coordinates
        The number of points on the zbuf at the image coordinates that are larger than the predicted points
        determines the sign of sdf
        """
        assert(points.ndim == 2 and points.shape[-1] == 3)
        device = points.device
        faces_per_pixel = 4
        with torch.autograd.no_grad():
            # a point that is definitely outside the mesh as camera center
            ray0 = torch.tensor([2, 2, 2], device=device,
                                dtype=points.dtype).view(1, 3)
            R, T = look_at_view_transform(
                eye=ray0, at=((0, 0, 0),), up=((0, 0, 1),))
            cameras = PerspectiveCameras(R=R, T=T, device=device)
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(
                faces_per_pixel=faces_per_pixel, ))
            fragments = rasterizer(mesh_gt)

            z_predicted = cameras.get_world_to_view_transform().transform_points(
                points=points.unsqueeze(0))[..., -1:]
            # normalized pixel (top-left smallest values)
            screen_xy = -cameras.transform_points(points.unsqueeze(0))[..., :2]
            outside_screen = (screen_xy.abs() > 1.0).any(dim=-1)

            # pix_to_face, zbuf, bary_coords, dists
            assert(fragments.zbuf.shape[-1] == faces_per_pixel)
            zbuf = torch.nn.functional.grid_sample(fragments.zbuf.permute(0, 3, 1, 2),
                                                   screen_xy.clamp(-1.0,
                                                                   1.0).view(1, -1, 1, 2),
                                                   align_corners=False, mode='nearest')
            zbuf[outside_screen.unsqueeze(
                1).expand(-1, zbuf.shape[1], -1)] = -1.0
            sign = (((zbuf > z_predicted).sum(dim=1) % 2) ==
                    0).type_as(points).view(screen_xy.shape[1])
            sign = sign * 2 - 1

        pcls = PointClouds3D(points.unsqueeze(0)).to(device=device)

        points_first_idx = pcls.cloud_to_packed_first_idx()
        max_points = pcls.num_points_per_cloud().max().item()

        # packed representation for faces
        verts_packed = mesh_gt.verts_packed()
        faces_packed = mesh_gt.faces_packed()
        tris = verts_packed[faces_packed]  # (T, 3, 3)
        tris_first_idx = mesh_gt.mesh_to_faces_packed_first_idx()
        max_tris = mesh_gt.num_faces_per_mesh().max().item()

        # point to face distance: shape (P,)
        point_to_face = point_face_distance(
            points, points_first_idx, tris, tris_first_idx, max_points
        )
        point_to_face = sign * torch.sqrt(eps_sqrt(point_to_face))
        loss = (point_to_face - sdf) ** 2
        return loss


class SDF2DLoss(BaseLoss):
    def __init__(self, reduction='sum', channel_dim=None):
        super().__init__(reduction, channel_dim)

    @classmethod
    def cross_entropy_occupancy_loss(cls, logits_pred, is_occupied=True):
        ''' Calculates the cross entropy occupancy loss.

        Args:
            logits_pred (tensor): predicted logits
            is_occupied (bool): whether the points should be occupied or not
            reduction_method (string): reduction method (sum or mean)
        '''
        if is_occupied:
            occ_gt = torch.ones_like(logits_pred)
        else:
            occ_gt = torch.zeros_like(logits_pred)

        loss_out = F.binary_cross_entropy_with_logits(
            logits_pred, occ_gt, reduction='none')

        return loss_out

    @classmethod
    def occupancy_loss(cls, logits_pred):
        ''' Calculates the occupancy loss.

        Args:
            logits_pred (tensor): predicted logits
            weights (tensor): whether to weight the points with given tensor
            reduction_method (string): reduction method (sum or mean)
        '''
        return cls.cross_entropy_occupancy_loss(logits_pred)

    @classmethod
    def freespace_loss(cls, logits_pred):
        ''' Calculates the freespace loss.

        Args:
            logits_pred (tensor): predicted logits
            weights (tensor): whether to weight the points with given tensor
            reduction_method (string): reduction method (sum or mean)
        '''
        return cls.cross_entropy_occupancy_loss(logits_pred, is_occupied=False)

    def compute(self, logits_pred: torch.Tensor, freespace: bool):
        logits_pred = logits_pred.view(-1)
        if freespace:
            return self.freespace_loss(logits_pred)
        else:
            return self.occupancy_loss(logits_pred)
