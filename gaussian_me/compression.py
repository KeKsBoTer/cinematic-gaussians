from dataclasses import dataclass
import math
import time
import torch
from torch import nn
from torch_scatter import scatter
from typing import Tuple, Optional
from tqdm import trange
import tqdm
import gc
from torch.utils.tensorboard import SummaryWriter

from .model import GaussianModel
from .utils.splats import to_full_cov, extract_rot_scale
from simple_nn._C import nearestNeighbor


class VectorQuantize(nn.Module):

    def __init__(
        self,
        channels: int,
        codebook_size: int = 2**12,
        decay: float = 0.5,
        k_expire: int = 10,
    ) -> None:
        super().__init__()
        self.decay = decay
        self.codebook = nn.Parameter(
            torch.empty(codebook_size, channels), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.codebook)
        self.entry_importance = nn.Parameter(
            torch.zeros(codebook_size), requires_grad=False
        )
        self.eps = 1e-5
        self.k_expire = k_expire

    def uniform_init(self, x: torch.Tensor):
        amin, amax = x.aminmax()
        self.codebook.data = torch.rand_like(self.codebook) * (amax - amin) + amin

    def kmeans_pp_init(
        self,
        x: torch.Tensor,
        importance: torch.Tensor,
        steps: int = 10,
        batch_size: int = 2**15,
    ):
        self.codebook = nn.Parameter(
            kmeanpp_init(
                x,
                importance,
                self.codebook.shape[0],
                steps,
                batch_size,
            ),
            requires_grad=True,
        )

    def update(self, x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            min_dists, idx = nearestNeighbor(x.detach(), self.codebook.detach())
            acc_importance = scatter(
                importance, idx, 0, reduce="sum", dim_size=self.codebook.shape[0]
            )

            ema_inplace(self.entry_importance, acc_importance, self.decay)

            codebook = scatter(
                x * importance[:, None],
                idx,
                0,
                reduce="sum",
                dim_size=self.codebook.shape[0],
            )

            ema_inplace(
                self.codebook,
                codebook / (acc_importance[:, None] + self.eps),
                self.decay,
            )

            self._replace(x, importance)

            return min_dists

    def _replace(self, x: torch.Tensor, importance: torch.Tensor):
        if self.k_expire > 0:
            least_important = self.entry_importance.topk(self.k_expire, largest=False)[
                1
            ]
            most_important = importance.topk(self.k_expire, largest=True)[1]
            self.codebook[least_important] = x[most_important]

    def forward(
        self,
        x: torch.Tensor,
        return_dists: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        min_dists, idx = nearestNeighbor(x.detach(), self.codebook.detach())
        if return_dists:
            return self.codebook[idx], idx, min_dists
        else:
            return self.codebook[idx], idx


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def vq_features(
    features: torch.Tensor,
    importance: torch.Tensor,
    codebook_size: int,
    vq_chunk: int = 2**16,
    k_expire: int = 10,
    steps: int = 1000,
    decay: float = 0.8,
    kpp_init: bool = True,
    plot_error: bool = False,
    tb_writer: SummaryWriter = None,
    name="",
    scale_normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    importance_n = importance / importance.max()
    vq_model = VectorQuantize(
        channels=features.shape[-1],
        codebook_size=codebook_size,
        decay=decay,
        k_expire=k_expire,
    ).to(device=features.device)

    if kpp_init:
        vq_model.kmeans_pp_init(
            features,
            importance_n,
            batch_size=vq_chunk,
            steps=int(math.ceil(codebook_size / 32)),
        )
    else:
        vq_model.uniform_init(features)

    errors = []
    for i in trange(steps):
        batch = torch.randint(low=0, high=features.shape[0], size=[vq_chunk])
        vq_feature = features[batch]
        error = (
            vq_model.update(vq_feature, importance=importance_n[batch]).mean().item()
        )
        # if tb_writer is not None:
        tb_writer.add_scalar(f"cluster_loss_{name}", error, i)
        errors.append(error)
        if scale_normalize:
            # this computes the trace of the codebook covariance matrices
            # we devide by the trace to ensure that matrices have normalized eigenvalues / scales
            tr = vq_model.codebook[:, [0, 3, 5]].sum(-1)
            vq_model.codebook /= tr[:, None]

    gc.collect()
    torch.cuda.empty_cache()

    if plot_error:
        from matplotlib import pyplot as plt

        plt.plot(errors)

    start = time.time()
    _, vq_indices = vq_model(features)
    torch.cuda.synchronize(device=vq_indices.device)
    end = time.time()
    print(f"calculating indices took {end-start} seconds ")
    return vq_model.codebook.data.detach(), vq_indices.detach()


def join_features(
    all_features: torch.Tensor,
    keep_mask: torch.Tensor,
    codebook: torch.Tensor,
    codebook_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keep_features = all_features[keep_mask]
    compressed_features = torch.cat([codebook, keep_features], 0)

    indices = torch.zeros(
        len(all_features), dtype=torch.long, device=all_features.device
    )
    indices[~keep_mask] = codebook_indices
    indices[keep_mask] = torch.arange(len(keep_features), device=indices.device) + len(
        codebook
    )

    return compressed_features, indices


@dataclass
class CompressionSettings:
    codebook_size: int
    importance_prune: float
    importance_include: float
    steps: int
    k_expire: int
    decay: float
    batch_size: int
    kmeanspp_init: bool


def compress_color(
    gaussians: GaussianModel,
    color_importance: torch.Tensor,
    color_comp: CompressionSettings,
    color_compress_non_dir: bool,
    tb_writer: SummaryWriter = None,
):
    keep_mask = color_importance > color_comp.importance_include

    print(f"color keep: {keep_mask.float().mean()*100:.2f}%")

    vq_mask_c = ~keep_mask

    # remove zero sh component
    if color_compress_non_dir:
        n_sh_coefs = gaussians.color.shape[1]
        color_features = gaussians.color.detach().flatten(-2)
    else:
        n_sh_coefs = gaussians.color.shape[1] - 1
        color_features = gaussians.color[:, 1:].detach().flatten(-2)
    if vq_mask_c.any():
        print("compressing color...")
        color_codebook, color_vq_indices = vq_features(
            color_features[vq_mask_c],
            color_importance[vq_mask_c],
            color_comp.codebook_size,
            color_comp.batch_size,
            color_comp.k_expire,
            color_comp.steps,
            kpp_init=color_comp.kmeanspp_init,
            tb_writer=tb_writer,
            name="color",
        )
    else:
        color_codebook = torch.empty(
            (0, color_features.shape[-1]), device=color_features.device
        )
        color_vq_indices = torch.empty(
            (0,), device=color_features.device, dtype=torch.long
        )

    all_features = color_features
    compressed_features, indices = join_features(
        all_features, keep_mask, color_codebook, color_vq_indices
    )

    gaussians.set_color_indexed(compressed_features.reshape(-1, n_sh_coefs, 3), indices)


def compress_covariance(
    gaussians: GaussianModel,
    gaussian_importance: torch.Tensor,
    gaussian_comp: CompressionSettings,
    tb_writer: SummaryWriter = None,
):

    keep_mask_g = gaussian_importance > gaussian_comp.importance_include

    vq_mask_g = ~keep_mask_g

    print(f"gaussians keep: {keep_mask_g.float().mean()*100:.2f}%")

    # covariance = gaussians.get_normalized_covariance(strip_sym=True).detach()
    covariance = gaussians.get_covariance(strip_sym=True).detach()

    if vq_mask_g.any():
        print("compressing gaussian splats...")
        cov_codebook, cov_vq_indices = vq_features(
            covariance[vq_mask_g],
            gaussian_importance[vq_mask_g],
            gaussian_comp.codebook_size,
            gaussian_comp.batch_size,
            gaussian_comp.k_expire,
            gaussian_comp.steps,
            kpp_init=gaussian_comp.kmeanspp_init,
            tb_writer=tb_writer,
            name="gaussians",
            scale_normalize=False,
        )
    else:
        cov_codebook = torch.empty(
            (0, covariance.shape[1], 1), device=covariance.device
        )
        cov_vq_indices = torch.empty((0,), device=covariance.device, dtype=torch.long)

    compressed_cov, cov_indices = join_features(
        covariance,
        keep_mask_g,
        cov_codebook,
        cov_vq_indices,
    )

    rot_vq, scale_vq = extract_rot_scale(to_full_cov(compressed_cov))

    gaussians.set_gaussian_indexed(
        rot_vq.to(compressed_cov.device),
        scale_vq.to(compressed_cov.device),
        cov_indices,
    )


def compress_gaussians(
    gaussians: GaussianModel,
    color_importance: torch.Tensor,
    gaussian_importance: torch.Tensor,
    color_comp: Optional[CompressionSettings],
    gaussian_comp: Optional[CompressionSettings],
    color_compress_non_dir: bool,
    prune_threshold: float = 0.0,
    tb_writer: SummaryWriter = None,
):
    with torch.no_grad():
        if prune_threshold >= 0:
            non_prune_mask = color_importance > prune_threshold
            print(f"non_prune_color: {non_prune_mask.float().mean()*100:.2f}%")
            gaussians.mask_splats(non_prune_mask)
            gaussian_importance = gaussian_importance[non_prune_mask]
            color_importance = color_importance[non_prune_mask]

        if color_comp is not None:
            compress_color(
                gaussians,
                color_importance,
                color_comp,
                color_compress_non_dir,
                tb_writer,
            )
        if gaussian_comp is not None:
            compress_covariance(
                gaussians,
                gaussian_importance,
                gaussian_comp,
                tb_writer,
            )


def kmeanpp_init(
    features: torch.Tensor,
    importance: torch.Tensor,
    codebook_size: int,
    steps: int = 10,
    batch_size: int = 2**15,
) -> torch.Tensor:
    with torch.no_grad():
        codebook = torch.empty(
            (codebook_size, features.shape[-1]), device=features.device
        )
        samples_per_step = math.ceil(codebook_size / steps)
        codebook[:samples_per_step] = features[
            torch.multinomial(importance, samples_per_step)
        ]

        for i in trange(1, steps, desc="Kmeans++ init"):
            batch = torch.randint(low=0, high=features.shape[0], size=[batch_size])
            min_dists, idx = nearestNeighbor(
                features[batch], codebook[: i * samples_per_step]
            )

            selected_idx = torch.multinomial(
                min_dists.square() * importance[batch], 1, replacement=False
            )
            codebook[i * samples_per_step : (i + 1) * samples_per_step] = features[
                batch
            ][idx[selected_idx]]

        return codebook


@torch.no_grad()
def init_cdf_mask(
    importance: torch.Tensor, prune=0.0, keep=float("inf")
) -> Tuple[torch.Tensor, torch.Tensor]:
    print("start cdf three split")
    non_prune_mask = importance > prune

    keep_mask = importance > keep

    return non_prune_mask, keep_mask
