import gc
from itertools import cycle
import json
import os
import random
import numpy as np
from gaussian_me.utils.camera_utils import camera_to_JSON
import torch
import sys

from gaussian_me.io import CamerasDataset
from gaussian_me.model import GaussianModel
from gaussian_me.optim import GaussianOptimizer
from gaussian_me.utils.loss_utils import l1_loss, ssim
from gaussian_me.renderer import render
import sys
from gaussian_me.model import GaussianModel
from tqdm import tqdm
from gaussian_me.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from gaussian_me.args import ModelParams, PipelineParams, OptimizationParams

from torch.utils.tensorboard import SummaryWriter


def training(
    model_params: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    testing_iterations: list[int],
    saving_iterations: list[int],
):
    tb_writer = prepare_output_and_logger(model_params.model_path, model_params)

    dataset = CamerasDataset.from_folder(model_params.source_path, model_params.images)

    json_cams = []
    for id, cam in enumerate(dataset.scene_info.cameras):
        json_cams.append(camera_to_JSON(id, cam))
    with open(os.path.join(args.model_path, "cameras.json"), "w") as file:
        json.dump(json_cams, file)

    if model_params.eval:
        train_dataset, val_dataset = dataset.split_train_val()
    else:
        train_dataset = dataset
        val_dataset = []

    gaussians = GaussianModel.from_pc(
        pcd=dataset.init_pc(),
        kernel_size=model_params.kernel_size,
        max_sh_degree=model_params.sh_degree,
    ).cuda()

    dataloader = iter(
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=8,
            collate_fn=lambda x: x,
            pin_memory=True,
        )
    )

    scales = [1]
    current_scale = scales.pop(0)

    dataloader_train = cycle([d[0].cuda() for d in dataloader])  # cycle(dataloader)  #

    optimizer = GaussianOptimizer(gaussians, opt, dataset.cameras_extent())

    background = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    gaussians.compute_3D_filter(
        cameras=dataset.scene_info.cameras, resolution_scale=current_scale
    )

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):

        iter_start.record()

        optimizer.update_learning_rate(iteration)

        viewpoint_cam = next(dataloader_train)

        bg = (
            torch.tensor(
                [random.random(), random.random(), random.random(), 1.0], device="cuda"
            )
            if opt.random_background
            else background
        )

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            resolution_scale=current_scale,
        )

        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss calculation

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
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                gaussians,
                train_dataset,
                val_dataset,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                render,
                (pipe, background, 1, current_scale),
            )

            # Densification

            optimizer.update(
                iteration,
                gaussians,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                dataset.scene_info.cameras,
                current_scale,
            )

            if iteration < opt.iterations:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                point_cloud_path = os.path.join(
                    model_params.model_path,
                    "point_cloud/iteration_{}".format(iteration),
                )
                gaussians.save_npz(os.path.join(point_cloud_path, "point_cloud.npz"))


def prepare_output_and_logger(output_folder: str, args: ModelParams) -> SummaryWriter:
    # Set up output folder
    print("Output folder: {}".format(output_folder))
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    return SummaryWriter(output_folder)


def training_report(
    tb_writer,
    iteration: int,
    gaussians: GaussianModel,
    train_dataset: CamerasDataset,
    val_dataset: CamerasDataset,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {
                "name": "test",
                "cameras": val_dataset,
            },
            {
                "name": "train",
                "cameras": [
                    train_dataset[idx % len(train_dataset)] for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    result = renderFunc(viewpoint.cuda(), gaussians, *renderArgs)
                    image = torch.clamp(
                        result["render"][:3],
                        0.0,
                        1.0,
                    )
                    scale = renderArgs[-1]
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda")[:3, ::scale, ::scale] / 255,
                        0.0,
                        1.0,
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None, :3],
                            global_step=iteration,
                        )

                        # if iteration == testing_iterations[0]:
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/ground_truth".format(viewpoint.image_name),
                            gt_image[None, :3],
                            global_step=iteration,
                        )

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", gaussians.opacity, iteration
            )
            tb_writer.add_scalar("total_points", gaussians.num_points(), iteration)
        torch.cuda.empty_cache()



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
    )

    # All done
    print("\nTraining complete.")
