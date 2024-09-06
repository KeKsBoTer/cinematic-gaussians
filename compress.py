# %%
import gc
import glob
import json
import os
import random
import time
from argparse import ArgumentParser, Namespace
from itertools import cycle
from os import path
from shutil import copyfile
from typing import Dict, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from tqdm import tqdm

# %%
from gaussian_me.args import (
    CompressionParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from gaussian_me.compression import CompressionSettings, compress_gaussians
from gaussian_me.eval import render_and_eval
from gaussian_me.io import CamerasDataset
from gaussian_me.model import GaussianModel
from gaussian_me.optim import GaussianOptimizer
from gaussian_me.renderer import render
from gaussian_me.utils.general_utils import build_scaling_rotation, strip_symmetric
from gaussian_me.utils.loss_utils import l1_loss, ssim
from gaussian_me.utils.system_utils import searchForMaxIteration
from train import prepare_output_and_logger, training_report


def calc_importance(
    gaussians: GaussianModel, dataset: CamerasDataset, pipeline_params
) -> Tuple[torch.Tensor, torch.Tensor]:

    scaling = gaussians.scaling.detach()
    cov3d = build_covariance(
        scaling, 1.0, gaussians.rotation.detach(), True
    ).requires_grad_(True)

    h1 = gaussians._features_dc.register_hook(lambda grad: grad.abs())
    h2 = gaussians._features_rest.register_hook(lambda grad: grad.abs())
    h3 = cov3d.register_hook(lambda grad: grad.abs())
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    gaussians._features_dc.grad = None
    gaussians._features_rest.grad = None
    num_pixels = 0
    for camera in tqdm(dataset, desc="Calculating sensitivity"):
        rendering = render(
            camera.cuda(),
            gaussians,
            pipeline_params,
            background,
            clamp_color=False,
            cov3d=cov3d,
        )["render"]
        loss = rendering.sum()
        loss.backward()
        num_pixels += rendering.shape[1] * rendering.shape[2]

    importance = (
        torch.cat(
            [gaussians._features_dc.grad, gaussians._features_rest.grad],
            1,
        ).flatten(-2)
        / num_pixels
    )
    cov_grad = cov3d.grad / num_pixels
    h1.remove()
    h2.remove()
    h3.remove()
    torch.cuda.empty_cache()
    return importance.detach(), cov_grad.detach()


def build_covariance(scaling, scaling_modifier, rotation, strip_sym=True):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    if strip_sym:
        return strip_symmetric(actual_covariance)
    else:
        return actual_covariance


def save_hpt(
    tb_writer: SummaryWriter,
    comp_params: CompressionParams,
    optim_params: OptimizationParams,
    metrics: Dict[str, int],
    iteration: int,
    first=True,
):
    print(f"saving hpt for iteration {iteration}")
    optim_dict = {f"optim/{name}": value for name, value in vars(optim_params).items()}
    comp_dict = {f"comp/{name}": value for name, value in vars(comp_params).items()}
    if first:
        exp, ssi, sei = hparams(
            {**optim_dict, **comp_dict}, {name: 0 for name, _ in metrics.items()}
        )
        tb_writer.file_writer.add_summary(exp)
        tb_writer.file_writer.add_summary(ssi)
        tb_writer.file_writer.add_summary(sei)

    for name, value in metrics.items():
        tb_writer.add_scalar(name, value, iteration)




def run_vq(
    model_params: ModelParams,
    optim_params: OptimizationParams,
    pipeline_params: PipelineParams,
    comp_params: CompressionParams,
):
    if comp_params.load_iteration == -1:
        comp_params.load_iteration = searchForMaxIteration(
            os.path.join(model_params.model_path, "point_cloud")
        )
    print("Loading trained model at iteration {}".format(comp_params.load_iteration))
    ply_file = glob.glob(
        os.path.join(
            model_params.model_path,
            "point_cloud",
            "iteration_" + str(comp_params.load_iteration),
            "point_cloud.*",
        )
    )[0]
    gaussians = GaussianModel.from_file(ply_file).cuda()

    dataset = CamerasDataset.from_folder(model_params.source_path, model_params.images)

    if model_params.eval:
        train_dataset, val_dataset = dataset.split_train_val()
    else:
        train_dataset = dataset
        val_dataset = []

    timings = {}

    # %%

    start_time = time.time()
    color_importance, gaussian_sensitivity = calc_importance(
        gaussians, train_dataset, pipeline_params
    )
    end_time = time.time()
    timings["sensitivity_calculation"] = end_time - start_time
    tb_writer = SummaryWriter(comp_params.output_vq)

    # %%
    print("vq compression..")
    with torch.no_grad():
        start_time = time.time()
        color_importance_n = color_importance.amax(-1)

        gaussian_importance_n = gaussian_sensitivity.amax(-1)

        torch.cuda.empty_cache()

        color_compression_settings = CompressionSettings(
            codebook_size=comp_params.color_codebook_size,
            importance_prune=comp_params.color_importance_prune,
            importance_include=comp_params.color_importance_include,
            steps=int(comp_params.color_cluster_iterations),
            k_expire=int(comp_params.color_k_expire),
            kmeanspp_init=comp_params.color_kmeanspp_init,
            decay=comp_params.color_decay,
            batch_size=comp_params.color_batch_size,
        )

        gaussian_compression_settings = CompressionSettings(
            codebook_size=comp_params.gaussian_codebook_size,
            importance_prune=None,
            importance_include=comp_params.gaussian_importance_include,
            steps=int(comp_params.gaussian_cluster_iterations),
            k_expire=int(comp_params.gaussian_k_expire),
            kmeanspp_init=comp_params.gaussian_kmeanspp_init,
            decay=comp_params.gaussian_decay,
            batch_size=comp_params.gaussian_batch_size,
        )

        compress_gaussians(
            gaussians,
            color_importance_n,
            gaussian_importance_n,
            color_compression_settings if not comp_params.not_compress_color else None,
            (
                gaussian_compression_settings
                if not comp_params.not_compress_gaussians
                else None
            ),
            comp_params.color_compress_non_dir,
            prune_threshold=comp_params.prune_threshold,
            tb_writer=tb_writer,
        )
        end_time = time.time()
        timings["clustering"] = end_time - start_time

    gc.collect()
    torch.cuda.empty_cache()
    os.makedirs(comp_params.output_vq, exist_ok=True)

    copyfile(
        path.join(model_params.model_path, "cfg_args"),
        path.join(comp_params.output_vq, "cfg_args"),
    )
    model_params.model_path = comp_params.output_vq

    out_file = path.join(
        comp_params.output_vq,
        f"point_cloud/iteration_{comp_params.load_iteration}/point_cloud.npz",
    )

    gaussians.save_npz(out_file, sort_morton=False)
    file_size = os.path.getsize(out_file) / 1024**2

    # eval model
    print("evaluating...")
    metrics = render_and_eval(
        gaussians,
        val_dataset if len(val_dataset) > 0 else train_dataset,
        pipeline_params,
    )
    metrics["size"] = file_size
    metrics["duration"] = end_time - start_time
    save_hpt(
        tb_writer,
        comp_params,
        optim_params,
        metrics,
        comp_params.load_iteration,
        first=True,
    )
    print("done")
    with open(os.path.join(comp_params.output_vq, "cfg_args_comp"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(comp_params))))

    iteration = comp_params.load_iteration + comp_params.finetune_iterations
    if comp_params.finetune_iterations > 0:

        start_time = time.time()
        finetune(
            model_params,
            optim_params,
            pipeline_params,
            comp_params,
            train_dataset,
            val_dataset,
            gaussians,
            testing_iterations=torch.linspace(comp_params.load_iteration, comp_params.load_iteration+comp_params.finetune_iterations, 10).long().tolist(),
            saving_iterations=[comp_params.load_iteration+comp_params.finetune_iterations]
        )
        end_time = time.time()
        timings["finetune"] = end_time - start_time

        # %%
        out_file = path.join(
            comp_params.output_vq,
            f"point_cloud/iteration_{iteration}/point_cloud.npz",
        )
        start_time = time.time()
        gaussians.save_npz(out_file, sort_morton=not comp_params.not_sort_morton)
        end_time = time.time()
        timings["encode"] = end_time - start_time
        timings["total"] = sum(timings.values())
        with open(f"{comp_params.output_vq}/times.json", "w") as f:
            json.dump(timings, f)
        file_size = os.path.getsize(out_file) / 1024**2
        print(f"saved vq finetuned model to {out_file}")

        # eval model
        print("evaluating...")
        metrics = render_and_eval(
            gaussians,
            val_dataset if len(val_dataset) > 0 else train_dataset,
            pipeline_params,
        )
        metrics["size"] = file_size
        print(metrics)
        save_hpt(tb_writer, comp_params, optim_params, metrics, iteration, first=False)
        print("done")
    tb_writer.close()


def finetune(
    model_params: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    comp: CompressionParams,
    dataset_train: CamerasDataset,
    dataset_val: CamerasDataset,
    gaussians: GaussianModel,
    testing_iterations: list[int],
    saving_iterations: list[int],
):
    tb_writer = prepare_output_and_logger(model_params.model_path, model_params)

    first_iter = comp.load_iteration
    max_iter = first_iter + comp.finetune_iterations

    background = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device="cuda")

    dataloader = iter(
        torch.utils.data.DataLoader(
            dataset_train,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            collate_fn=lambda x: x,
            pin_memory=True,
        )
    )
    dataloader_train = cycle(dataloader)

    optimizer = GaussianOptimizer(gaussians, opt, dataset_train.cameras_extent())
    optimizer.update_learning_rate(first_iter)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    gaussians.compute_3D_filter(cameras=dataset_train.scene_info.cameras)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, max_iter), desc="Training progress")
    for iteration in range(first_iter, max_iter + 1):
        iter_start.record()

        viewpoint_cam = next(dataloader_train)[0].cuda()

        bg = (
            torch.tensor(
                [random.random(), random.random(), random.random(), 1.0], device="cuda"
            )
            if opt.random_background
            else background
        )
        # Render
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
        )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        # image+=bg[:,None,None]*(1-image[3:4])

        # Loss
        with torch.no_grad():
            gt_image = viewpoint_cam.original_image / 255

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == max_iter:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                gaussians,
                dataset_train,
                dataset_val,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                render,
                (pipe, background, 1, 1),
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                point_cloud_path = os.path.join(
                    model_params.model_path,
                    "point_cloud/iteration_{}".format(iteration),
                )
                gaussians.save_npz(os.path.join(point_cloud_path, "point_cloud.npz"))


if __name__ == "__main__":
    parser = ArgumentParser(description="Compression script parameters")
    model = ModelParams(parser, sentinel=True)
    model.data_device = "cuda"
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    comp = CompressionParams(parser)
    args = get_combined_args(parser)

    model_params = model.extract(args)
    optim_params = op.extract(args)
    pipeline_params = pipeline.extract(args)
    comp_params = comp.extract(args)

    run_vq(model_params, optim_params, pipeline_params, comp_params)
