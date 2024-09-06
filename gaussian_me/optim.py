import torch
from gaussian_me.args import OptimizationParams
from gaussian_me.io.dataset_readers import CameraInfo
from gaussian_me.model import GaussianModel

from gaussian_me.utils.general_utils import (
    build_rotation,
    get_expon_lr_func,
)


class GaussianOptimizer(torch.optim.Adam):
    def __init__(
        self,
        pc: GaussianModel,
        params: OptimizationParams,
        spatial_lr_scale: float,
    ):
        self.percent_dense = params.percent_dense
        self.params = params
        self.spatial_lr_scale = spatial_lr_scale

        l = [
            {
                "params": [pc._xyz],
                "lr": params.position_lr_init * spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [pc._features_dc],
                "lr": params.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [pc._features_rest],
                "lr": params.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [pc._opacity],
                "lr": params.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [pc._scaling],
                "lr": params.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [pc._rotation],
                "lr": params.rotation_lr,
                "name": "rotation",
            },
        ]

        super(GaussianOptimizer, self).__init__(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=params.position_lr_init * spatial_lr_scale,
            lr_final=params.position_lr_final * spatial_lr_scale,
            lr_delay_mult=params.position_lr_delay_mult,
            max_steps=params.position_lr_max_steps,
        )

        self.xyz_gradient_accum = torch.zeros((pc.num_points(), 1), device="cuda")
        self.denom = torch.zeros((pc.num_points(), 1), device="cuda")
        self.max_radii2D = torch.zeros((pc.num_points()), device="cuda")

    def update(
        self,
        iteration: int,
        gaussians: GaussianModel,
        viewspace_point_tensor: torch.Tensor,
        visibility_filter: torch.Tensor,
        radii: torch.Tensor,
        trainCameras: list[CameraInfo],
        resolution_scale: int = 1,
    ):
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if (iteration + 1) % 1000 == 0:
            gaussians.oneupSHdegree()

        if (
            iteration < self.params.densify_until_iter
        ):  # and gaussians.num_points() < 4_000_000:
            # Keep track of max radii in image-space for pruning
            self.max_radii2D[visibility_filter] = torch.max(
                self.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            # add densification stats
            self.xyz_gradient_accum[visibility_filter] += torch.norm(
                viewspace_point_tensor.grad[visibility_filter, :2], dim=-1, keepdim=True
            )
            self.denom[visibility_filter] += 1

            if (
                iteration > self.params.densify_from_iter
                and iteration % self.params.densification_interval == 0
            ):
                size_threshold = (
                    20 if iteration > self.params.opacity_reset_interval else None
                )
                self.densify_and_prune(
                    gaussians,
                    self.params.densify_grad_threshold,
                    0.005,
                    self.spatial_lr_scale,
                    size_threshold,
                    self.params.percent_dense,
                )
                gaussians.compute_3D_filter(
                    cameras=trainCameras, resolution_scale=resolution_scale
                )

            if iteration % self.params.opacity_reset_interval == 0:
                self._reset_opacity(gaussians)

        if iteration % 100 == 0 and iteration > self.params.densify_until_iter:
            if iteration < self.params.iterations - 100:
                # don't update in the end of training
                gaussians.compute_3D_filter(
                    cameras=trainCameras, resolution_scale=resolution_scale
                )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def replace_tensor(self, tensor: torch.Tensor, name: str):
        optimizable_tensors = {}
        for group in self.param_groups:
            if group["name"] == name:
                stored_state = self.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.state[group["params"][0]]
                group["params"][0] = torch.nn.Parameter(tensor, requires_grad=True)
                self.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _reset_opacity(self, pc: GaussianModel):
        # reset opacity to by considering 3D filter
        current_opacity_with_filter = pc.opacity
        opacities_new = torch.min(
            current_opacity_with_filter,
            torch.ones_like(current_opacity_with_filter) * 0.01,
        )

        # apply 3D filter
        scales = pc.scaling_activation(pc._scaling)

        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1).clamp(min=1e-6)

        scales_after_square = scales_square + torch.square(pc.filter_3D)
        det2 = scales_after_square.prod(dim=1).clamp(min=1e-6)
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = pc.opacity_activation_inverse(opacities_new)

        optimizable_tensors = self.replace_tensor(opacities_new, "opacity")
        pc._opacity = optimizable_tensors["opacity"]

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.param_groups:
            stored_state = self.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.state[group["params"][0]]
                group["params"][0] = torch.nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_points(self, pc: GaussianModel, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        pc._xyz = optimizable_tensors["xyz"]
        pc._features_dc = optimizable_tensors["f_dc"]
        pc._features_rest = optimizable_tensors["f_rest"]
        pc._opacity = optimizable_tensors["opacity"]
        pc._scaling = optimizable_tensors["scaling"]
        pc._rotation = optimizable_tensors["rotation"]
        pc.filter_3D.data = pc.filter_3D[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def _cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.state[group["params"][0]]
                group["params"][0] = torch.nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _densification_postfix(
        self,
        pc: GaussianModel,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_filter3D,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self._cat_tensors_to_optimizer(d)
        pc._xyz = optimizable_tensors["xyz"]
        pc._features_dc = optimizable_tensors["f_dc"]
        pc._features_rest = optimizable_tensors["f_rest"]
        pc._opacity = optimizable_tensors["opacity"]
        pc._scaling = optimizable_tensors["scaling"]
        pc._rotation = optimizable_tensors["rotation"]
        pc.filter_3D.data = torch.cat([pc.filter_3D, new_filter3D], dim=0)

        self.xyz_gradient_accum = torch.zeros((pc.num_points(), 1), device="cuda")
        self.denom = torch.zeros((pc.num_points(), 1), device="cuda")
        self.max_radii2D = torch.zeros((pc.num_points()), device="cuda")

    def densify_and_split(
        self,
        pc: GaussianModel,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
        percent_dense: float,
        N=2,
    ):
        n_init_points = pc.num_points()
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        scaling = pc.scaling_activation(pc._scaling)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scaling, dim=1).values > percent_dense * scene_extent,
        )

        stds = scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(pc._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + pc.xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = pc.scaling_activation_inverse(
            scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = pc._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = pc._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = pc._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = pc._opacity[selected_pts_mask].repeat(N, 1)
        new_filter3D = pc.filter_3D[selected_pts_mask].repeat(N, 1)

        self._densification_postfix(
            pc,
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_filter3D,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self._prune_points(pc, prune_filter)

    def densify_and_clone(
        self,
        pc: GaussianModel,
        grads,
        grad_threshold,
        scene_extent,
        percent_dense: float,
    ):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        scaling = pc.scaling_activation(pc._scaling)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scaling, dim=1).values <= percent_dense * scene_extent,
        )

        new_xyz = pc._xyz[selected_pts_mask]
        new_features_dc = pc._features_dc[selected_pts_mask]
        new_features_rest = pc._features_rest[selected_pts_mask]
        new_opacities = pc._opacity[selected_pts_mask]
        new_scaling = pc._scaling[selected_pts_mask]
        new_rotation = pc._rotation[selected_pts_mask]
        new_filter3D = pc.filter_3D[selected_pts_mask]

        self._densification_postfix(
            pc,
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_filter3D,
        )

    def densify_and_prune(
        self,
        pc: GaussianModel,
        max_grad,
        min_opacity: float,
        extent,
        max_screen_size,
        percent_dense: float,
    ):

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(pc, grads, max_grad, extent, percent_dense)
        self.densify_and_split(pc, grads, max_grad, extent, percent_dense)

        prune_mask = (pc.opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = pc.scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self._prune_points(pc, prune_mask)

        torch.cuda.empty_cache()
