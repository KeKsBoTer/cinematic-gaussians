from os import makedirs
import os
from typing import Self
from plyfile import PlyData, PlyElement

import numpy as np
from tqdm import tqdm
from gaussian_me.io.dataset_readers import BasicPointCloud, CameraInfo
from gaussian_me.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    inverse_sigmoid,
    mortonEncode,
    strip_symmetric,
)
import torch

from gaussian_me.utils.graphics_utils import fov2focal
from gaussian_me.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2


class GaussianModel(torch.nn.Module):

    def __init__(
        self,
        xyz: torch.Tensor,
        sh_coefs: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        kernel_size: float = 0.1,
        max_sh_degree: int = 3,
        gaussian_indices: torch.Tensor = None,
        feature_indices: torch.Tensor = None,
        active_sh_degree: int = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.active_sh_degree = active_sh_degree
        self.max_sh_degree = max_sh_degree

        self.scaling_activation = torch.exp
        self.scaling_activation_inverse = torch.log

        self.rotation_activation = torch.nn.functional.normalize
        self.rotation_activation_inverse = lambda x: x

        self.opacity_activation = torch.sigmoid
        self.opacity_activation_inverse = inverse_sigmoid

        self._xyz = torch.nn.Parameter(xyz.contiguous(), requires_grad=True)
        self._features_dc = torch.nn.Parameter(
            sh_coefs[:, :1].contiguous(), requires_grad=True
        )
        self._features_rest = torch.nn.Parameter(
            sh_coefs[:, 1:].contiguous(), requires_grad=True
        )
        self._scaling = torch.nn.Parameter(
            self.scaling_activation_inverse(scale.contiguous()), requires_grad=True
        )
        self._rotation = torch.nn.Parameter(
            self.rotation_activation_inverse(rotation.contiguous()), requires_grad=True
        )
        self._opacity = torch.nn.Parameter(
            self.opacity_activation_inverse(opacity.contiguous()), requires_grad=True
        )
        self.filter_3D = torch.nn.Parameter(
            torch.zeros_like(self._opacity), requires_grad=False
        )
        self.kernel_size = kernel_size

        # qunatization aware training
        self.opacity_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8)
        self.scaling_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8)
        self.rotation_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8)
        self.color_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8)

        # quantization related stuff
        self._feature_indices = (
            torch.nn.Parameter(feature_indices, requires_grad=False)
            if feature_indices is not None
            else None
        )
        self._gaussian_indices = (
            torch.nn.Parameter(gaussian_indices, requires_grad=False)
            if gaussian_indices is not None
            else None
        )

    @classmethod
    def from_pc(
        cls,
        max_sh_degree: int,
        pcd: BasicPointCloud,
        kernel_size: float = 0.1,
    ) -> Self:

        fused_point_cloud = torch.from_numpy(pcd.points).float()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        sh_0 = RGB2SH(torch.tensor(pcd.colors).float())
        features = torch.zeros((sh_0.shape[0], sh_num_coefs(max_sh_degree), 3)).float()
        features[:, 0] = sh_0
        dist2 = torch.clamp_min(
            distCUDA2(fused_point_cloud.cuda()),
            0.0000001,
        ).cpu()
        # initialize as isotropic splats
        scales = torch.sqrt(dist2)[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = 0.1 * torch.ones(
            (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
        )
        return cls(
            xyz=fused_point_cloud,
            sh_coefs=features,
            scale=scales,
            rotation=rots,
            opacity=opacities,
            kernel_size=kernel_size,
            max_sh_degree=max_sh_degree,
            active_sh_degree=0,
        )

    def num_points(self) -> int:
        return len(self._xyz)

    @property
    def rotation(self) -> torch.Tensor:
        rotation = self.rotation_activation(self.rotation_qa(self._rotation))
        if self._gaussian_indices is not None:
            return rotation[self._gaussian_indices]
        else:
            return rotation

    @property
    def xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def scaling(self) -> torch.Tensor:
        scales = self.scaling_activation(self._scaling)
        if self._gaussian_indices is not None:
            scales = scales[self._gaussian_indices]
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return self.scaling_activation(
            self.scaling_qa(self.scaling_activation_inverse(scales))
        )

    @property
    def color(self) -> torch.Tensor:
        color = self.color_qa(
            torch.cat([self._features_dc, self._features_rest], dim=1)
        )
        if self._feature_indices is not None:
            return color[self._feature_indices]
        else:
            return color

    @property
    def opacity(self) -> torch.Tensor:
        opacity = self.opacity_activation(self._opacity)

        # apply 3D filter
        scales = self.scaling_activation(self._scaling)
        if self._gaussian_indices is not None:
            scales = scales[self._gaussian_indices]

        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)

        scales_after_square = scales_square + torch.square(self.filter_3D)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        return self.opacity_qa(opacity * coef[..., None])

    def get_covariance(self, strip_sym=False) -> torch.Tensor:
        return build_covariance_from_scaling_rotation(
            self.scaling, 1.0, self.rotation, strip_sym=strip_sym
        )

    def oneupSHdegree(self):
        self.active_sh_degree = min(self.max_sh_degree, self.active_sh_degree + 1)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
        )

    @torch.no_grad()
    def compute_3D_filter(self, cameras: list[CameraInfo], resolution_scale: int = 1):
        xyz = self.xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)

        # we should use the focal length of the highest resolution camera
        focal_length = 0.0
        for camera in cameras:
            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
            # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]

            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2

            width = camera.width / resolution_scale
            height = camera.height / resolution_scale

            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)

            focal_x = fov2focal(camera.FovX, width)
            focal_y = fov2focal(camera.FovY, height)
            x = x / z * focal_x + width / 2.0
            y = y / z * focal_y + height / 2.0

            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(
                torch.logical_and(x >= -0.15 * width, x <= width * 1.15),
                torch.logical_and(y >= -0.15 * height, y <= 1.15 * height),
            )

            valid = torch.logical_and(valid_depth, in_screen)

            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < focal_x:
                focal_length = focal_x

        distance[~valid_points] = distance[valid_points].max()

        # TODO remove hard coded value
        # TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2**0.5)
        self.filter_3D = torch.nn.Parameter(filter_3D[..., None], requires_grad=False)

    def construct_list_of_attributes(self, exclude_filter=True):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        if not exclude_filter:
            l.append("filter_3D")
        return l

    def save_ply(self, path, background_color=None):
        makedirs(os.path.dirname(path), exist_ok=True)

        color_features = self.color.detach()

        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            color_features[:, :1]
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            color_features[:, 1:]
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self.opacity_activation_inverse(self.opacity).detach().cpu().numpy()
        scale = self.scaling_activation_inverse(self.scaling).detach().cpu().numpy()

        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")

        ply_data = PlyData(
            [el],
            comments=[
                f"mip=true",
                f"kernel_size={self.kernel_size}",
                f"background_color={background_color}",
            ],
        )

        ply_data.write(path)

    @classmethod
    def from_file(cls, path) -> Self:
        ext = os.path.splitext(path)[1]
        if ext == ".ply":
            return cls.from_ply(path)
        elif ext == ".npz":
            return cls.from_npz(path)
        else:
            raise NotImplementedError(f"file ending '{ext}' not supported")

    @classmethod
    def from_ply(cls, path) -> Self:
        plydata = PlyData.read(path)

        kernel_size = 0.1
        for comment in plydata.comments:
            if comment.startswith("mip="):
                mip_splatting = comment.split("=")[1] == "true"
                if not mip_splatting:
                    print(
                        "WARNING: Loaded model does not have mip splatting enabled. This might lead to unexpected results"
                    )
            if comment.startswith("kernel_size="):
                kernel_size = float(comment.split("=")[1])

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))

        features = torch.from_numpy(
            np.concatenate((features_dc, features_extra), axis=2)
        ).swapaxes(1, 2)

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_") and not p.name.startswith("scale_factor")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return cls(
            xyz=torch.from_numpy(xyz).float(),
            sh_coefs=features.float(),
            opacity=torch.from_numpy(opacities).float().sigmoid(),
            scale=torch.from_numpy(scales).float().exp(),
            rotation=torch.nn.functional.normalize(
                torch.from_numpy(rots).float(), dim=-1
            ),
            kernel_size=kernel_size,
            active_sh_degree=int((features.shape[1]) ** 0.5 - 1),
        )

    @torch.no_grad()
    def save_npz(
        self,
        path,
        compress: bool = True,
        sort_morton=False,
        background_color=None,
    ):
        if sort_morton:
            self._sort_morton()
        if isinstance(path, str):
            makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # call the property function to ensure init of qa
        self.scaling
        self.rotation
        self.color
        self.xyz
        self.opacity

        save_dict = dict()

        save_dict["quantization"] = True
        save_dict["mip_splatting"] = True
        save_dict["kernel_size"] = np.array(self.kernel_size, dtype=np.float32)
        if background_color is not None:
            save_dict["background_color"] = np.array(background_color, dtype=np.float32)

        # save position
        save_dict["xyz"] = self.xyz.detach().half().cpu().numpy()

        # save color features
        color_dc_q = torch.quantize_per_tensor(
            self._features_dc.detach(),
            self.color_qa.scale,
            self.color_qa.zero_point,
            self.color_qa.dtype,
        ).int_repr()
        color_rest_q = torch.quantize_per_tensor(
            self._features_rest.detach(),
            self.color_qa.scale,
            self.color_qa.zero_point,
            self.color_qa.dtype,
        ).int_repr()
        save_dict["features_dc"] = color_dc_q.cpu().numpy()
        save_dict["features_dc_scale"] = self.color_qa.scale.cpu().numpy()
        save_dict["features_dc_zero_point"] = self.color_qa.zero_point.cpu().numpy()

        save_dict["features_rest"] = color_rest_q.cpu().numpy()
        save_dict["features_rest_scale"] = self.color_qa.scale.cpu().numpy()
        save_dict["features_rest_zero_point"] = self.color_qa.zero_point.cpu().numpy()
        if self._feature_indices is not None:
            save_dict["feature_indices"] = (
                self._feature_indices.detach().contiguous().cpu().int().numpy()
            )
        if self._gaussian_indices is not None:
            save_dict["gaussian_indices"] = (
                self._gaussian_indices.detach().contiguous().cpu().int().numpy()
            )

        # save opacity
        opacity = self.opacity.detach()
        opacity_q = torch.quantize_per_tensor(
            opacity,
            scale=self.opacity_qa.scale,
            zero_point=self.opacity_qa.zero_point,
            dtype=self.opacity_qa.dtype,
        ).int_repr()
        save_dict["opacity"] = opacity_q.cpu().numpy()
        save_dict["opacity_scale"] = self.opacity_qa.scale.cpu().numpy()
        save_dict["opacity_zero_point"] = self.opacity_qa.zero_point.cpu().numpy()

        # save scaling

        # scales = self.scaling_activation(self._scaling)
        # if self._gaussian_indices is not None:
        #     scales = scales[self._gaussian_indices]
        # scales = torch.square(scales) + torch.square(self.filter_3D)
        # scales = torch.sqrt(scales)
        # return self.scaling_activation(
        #     self.scaling_qa(self.scaling_activation_inverse(scales))
        # )

        # scaling = self.scaling_activation_inverse(self.scaling.detach())
        scaling = self._scaling.detach()
        scaling_q = torch.quantize_per_tensor(
            scaling,
            scale=self.scaling_qa.scale,
            zero_point=self.scaling_qa.zero_point,
            dtype=self.scaling_qa.dtype,
        ).int_repr()
        save_dict["scaling"] = scaling_q.cpu().numpy()
        save_dict["scaling_scale"] = self.scaling_qa.scale.cpu().numpy()
        save_dict["scaling_zero_point"] = self.scaling_qa.zero_point.cpu().numpy()

        # save rotation
        rotation = self._rotation.detach()
        rotation_q = torch.quantize_per_tensor(
            rotation,
            scale=self.rotation_qa.scale,
            zero_point=self.rotation_qa.zero_point,
            dtype=self.rotation_qa.dtype,
        ).int_repr()
        save_dict["rotation"] = rotation_q.cpu().numpy()
        save_dict["rotation_scale"] = self.rotation_qa.scale.cpu().numpy()
        save_dict["rotation_zero_point"] = self.rotation_qa.zero_point.cpu().numpy()

        save_fn = np.savez_compressed if compress else np.savez
        save_fn(path, **save_dict)

    @classmethod
    def from_npz(cls, path):
        state_dict = np.load(path)

        kernel_size = state_dict.get("kernel_size", 0.1)
        # load position
        xyz = torch.from_numpy(state_dict["xyz"]).float()

        # load color
        features_rest_q = torch.from_numpy(state_dict["features_rest"]).int().cuda()
        features_rest_scale = torch.from_numpy(state_dict["features_rest_scale"]).cuda()
        features_rest_zero_point = torch.from_numpy(
            state_dict["features_rest_zero_point"]
        ).cuda()
        features_rest = (
            features_rest_q - features_rest_zero_point
        ) * features_rest_scale
        features_rest = torch.nn.Parameter(features_rest, requires_grad=True)

        features_dc_q = torch.from_numpy(state_dict["features_dc"]).int().cuda()
        features_dc_scale = torch.from_numpy(state_dict["features_dc_scale"]).cuda()
        features_dc_zero_point = torch.from_numpy(
            state_dict["features_dc_zero_point"]
        ).cuda()
        features_dc = (features_dc_q - features_dc_zero_point) * features_dc_scale
        features = torch.cat([features_dc, features_rest], dim=1)

        # load opacity
        opacity_q = torch.from_numpy(state_dict["opacity"]).int().cuda()
        opacity_scale = torch.from_numpy(state_dict["opacity_scale"]).cuda()
        opacity_zero_point = torch.from_numpy(state_dict["opacity_zero_point"]).cuda()
        opacity = (opacity_q - opacity_zero_point) * opacity_scale

        # self.opacity_qa.scale = opacity_scale
        # self.opacity_qa.zero_point = opacity_zero_point
        # self.opacity_qa.activation_post_process.min_val = opacity.min()
        # self.opacity_qa.activation_post_process.max_val = opacity.max()

        # load scaling
        scaling_q = torch.from_numpy(state_dict["scaling"]).int().cuda()
        scaling_scale = torch.from_numpy(state_dict["scaling_scale"]).cuda()
        scaling_zero_point = torch.from_numpy(state_dict["scaling_zero_point"]).cuda()
        scaling = (scaling_q - scaling_zero_point) * scaling_scale
        # self.scaling_qa.scale = scaling_scale
        # self.scaling_qa.zero_point = scaling_zero_point
        # self.scaling_qa.activation_post_process.min_val = scaling.min()
        # self.scaling_qa.activation_post_process.max_val = scaling.max()

        # load rotation
        rotation_q = torch.from_numpy(state_dict["rotation"]).int().cuda()
        rotation_scale = torch.from_numpy(state_dict["rotation_scale"]).cuda()
        rotation_zero_point = torch.from_numpy(state_dict["rotation_zero_point"]).cuda()
        rotation = (rotation_q - rotation_zero_point) * rotation_scale
        # self.rotation_qa.scale = rotation_scale
        # self.rotation_qa.zero_point = rotation_zero_point
        # self.rotation_qa.activation_post_process.min_val = rotation.min()
        # self.rotation_qa.activation_post_process.max_val = rotation.max()

        gaussian_indices = None
        feature_indices = None
        if "gaussian_indices" in list(state_dict.keys()):
            gaussian_indices = torch.from_numpy(state_dict["gaussian_indices"]).long()

        if "feature_indices" in list(state_dict.keys()):
            feature_indices = torch.from_numpy(state_dict["feature_indices"]).long()

        return cls(
            xyz=xyz,
            sh_coefs=features.float(),
            opacity=opacity.float(),
            scale=scaling.float().exp(),
            rotation=torch.nn.functional.normalize(rotation.float(), dim=-1),
            kernel_size=kernel_size,
            active_sh_degree=int((features.shape[1]) ** 0.5 - 1),
            gaussian_indices=gaussian_indices,
            feature_indices=feature_indices,
        )

    @torch.no_grad()
    def mask_splats(self, mask: torch.Tensor):

        if self._gaussian_indices is not None or self._feature_indices is not None:
            raise Exception("masking indexed Gaussians is not supported")
        self._xyz = torch.nn.Parameter(self._xyz[mask], requires_grad=True)
        self._opacity = torch.nn.Parameter(self._opacity[mask], requires_grad=True)
        self._features_dc = torch.nn.Parameter(
            self._features_dc[mask], requires_grad=True
        )

        self._features_rest = torch.nn.Parameter(
            self._features_rest[mask], requires_grad=True
        )
        self._scaling = torch.nn.Parameter(self._scaling[mask], requires_grad=True)
        self._rotation = torch.nn.Parameter(self._rotation[mask], requires_grad=True)
        self.filter_3D = torch.nn.Parameter(self.filter_3D[mask], requires_grad=False)

    def set_color_indexed(self, features: torch.Tensor, indices: torch.Tensor):
        self._feature_indices = torch.nn.Parameter(indices, requires_grad=False)
        self._features_dc = torch.nn.Parameter(
            features[:, :1].detach(), requires_grad=True
        )
        self._features_rest = torch.nn.Parameter(
            features[:, 1:].detach(), requires_grad=True
        )

    def set_gaussian_indexed(
        self, rotation: torch.Tensor, scaling: torch.Tensor, indices: torch.Tensor
    ):
        self._gaussian_indices = torch.nn.Parameter(
            indices.detach(), requires_grad=False
        )
        self._rotation = torch.nn.Parameter(rotation.detach(), requires_grad=True)
        self._scaling = torch.nn.Parameter(
            self.scaling_activation_inverse(scaling.detach()), requires_grad=True
        )

    @torch.no_grad()
    def _sort_morton(self):
        xyz_q = (
            (2**21 - 1)
            * (self._xyz - self._xyz.min(0).values)
            / (self._xyz.max(0).values - self._xyz.min(0).values)
        ).long()
        order = mortonEncode(xyz_q).sort().indices
        self._xyz = torch.nn.Parameter(self._xyz[order], requires_grad=True)
        self._opacity = torch.nn.Parameter(self._opacity[order], requires_grad=True)

        if self._feature_indices is not None:
            self._feature_indices = torch.nn.Parameter(
                self._feature_indices[order], requires_grad=False
            )
        else:
            self._features_rest = torch.nn.Parameter(
                self._features_rest[order], requires_grad=True
            )
            self._features_dc = torch.nn.Parameter(
                self._features_dc[order], requires_grad=True
            )

        if self._gaussian_indices is not None:
            self._gaussian_indices = torch.nn.Parameter(
                self._gaussian_indices[order], requires_grad=False
            )
        else:
            self._scaling = torch.nn.Parameter(self._scaling[order], requires_grad=True)
            self._rotation = torch.nn.Parameter(
                self._rotation[order], requires_grad=True
            )


def sh_num_coefs(sh_deg: int) -> int:
    return (sh_deg + 1) ** 2


def build_covariance_from_scaling_rotation(
    scaling, scaling_modifier, rotation, strip_sym=True
):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    if strip_sym:
        return strip_symmetric(actual_covariance)
    else:
        return actual_covariance
