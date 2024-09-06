import gc
from os import makedirs
import os
from typing import Dict, Optional
import numpy as np
import torch
from tqdm import tqdm
from gaussian_me.args import PipelineParams
from gaussian_me.io import CamerasDataset
from gaussian_me.model import GaussianModel
from gaussian_me.renderer import render
from gaussian_me.utils.image_utils import psnr
from gaussian_me.utils.loss_utils import ssim
from matplotlib import pyplot as plt
from PIL import Image


def psnr_masked(img1, img2, mask):
    mse = ((img1 - img2) ** 2).mean(0)[mask].mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


@torch.no_grad
def render_and_eval(
    gaussians: GaussianModel,
    dataset: CamerasDataset,
    pipeline_params: PipelineParams,
    img_folder: Optional[str] = None,
) -> Dict[str, float]:
    ssims = []
    psnrs = []
    psnrs_alpha = []

    bg_color = [0.0, 0.0, 0.0, 1.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for view in tqdm(dataset, desc="Rendering progress"):
        view = view.cuda()
        rendering = render(view, gaussians, pipeline_params, background)[
            "render"
        ].clamp(0, 1)
        gt = view.original_image.float() / 255

        if img_folder is not None:
            makedirs(
                os.path.join(img_folder, "renders"),
                exist_ok=True,
            )
            makedirs(
                os.path.join(img_folder, "gt"),
                exist_ok=True,
            )  #
            im = Image.fromarray(
                (
                    rendering.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255
                ).astype(np.uint8)
            )
            im.save(
                os.path.join(
                    img_folder,
                    "renders",
                    f"img_{view.uid}.png",
                )
            )

            im = Image.fromarray(
                (gt.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255).astype(
                    np.uint8
                )
            )
            im.save(
                os.path.join(
                    img_folder,
                    "gt",
                    f"img_{view.uid}.png",
                )
            )

        mask = (rendering > 0).any(0) | (gt > 0).any(0)

        psnrs_alpha.append(psnr_masked(rendering[3:4], gt[3:4], mask))

        rendering_rgb = rendering[:3]
        gt_rgb = gt[:3]

        ssim_pp = ssim(rendering_rgb, gt_rgb, size_average=False)[mask].mean()
        psnr_pp = psnr_masked(rendering_rgb, gt_rgb, mask)

        ssims.append(ssim_pp)
        psnrs.append(psnr_pp)
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "SSIM": torch.tensor(ssims).mean().item(),
        "PSNR": torch.tensor(psnrs).mean().item(),
        "PSNR_ALPHA": torch.tensor(psnrs_alpha).mean().item(),
    }
