
<div align="center" >

# Application of 3D Gaussian Splatting for Cinematic Anatomy on Consumer Class Devices



<font size="4">
 Simon Niedermayr<sup>1</sup>, Christoph Neuhauser<sup>1</sup>, Kaloian Petkov<sup>2</sup>, Klaus Engel<sup>2</sup>, Rüdiger Westermann<sup>1</sup>

</font>
<br>

<font size="4">
<sup>1</sup>Technical University of Munich, <sup>2</sup>Siemens Healthineers 
</font>

<a href="https://keksboter.github.io/cinematic-gaussians/">Webpage</a> | <a href="https://arxiv.org/abs/2404.11285">arXiv</a> 

<img src="docs/static/img/preview.svg">

</div>

## Abstract
<div style="text-align:justify">
Interactive photorealistic visualization of 3D anatomy (i.e., Cinematic Anatomy) is used in medical education to explain the structure of the human body. It is currently restricted to frontal teaching scenarios, where the demonstrator needs a powerful GPU and high-speed access to a large storage device where the dataset is hosted. We demonstrate the use of novel view synthesis via compressed 3D Gaussian splatting to overcome this restriction and to enable students to perform cinematic anatomy on lightweight mobile devices and in virtual reality environments. We present an automatic approach for finding a set of images that captures all potentially seen structures in the data. By mixing closeup views with images from a distance, the splat representation can recover structures up to the voxel resolution. The use of Mip-Splatting enables smooth transitions when the focal length is increased. Even for GB datasets, the final renderable representation can usually be compressed to less than 70 MB, enabling interactive rendering on low-end devices using rasterization. 
</div>


## Citation
If you find our work useful, please cite:
```
@misc{niedermayr2024novel,
    title={Application of 3D Gaussian Splatting for Cinematic Anatomy on Consumer Class Devices},
    author={Simon Niedermayr and Christoph Neuhauser and Kaloian Petkov and Klaus Engel and Rüdiger Westermann},
    year={2024},
    eprint={2404.11285},
    archivePrefix={arXiv},
    primaryClass={cs.GR}
}
```

## Installation

Best to be installed with anaconda or miniconda:

1. Clone the repo:
```
git clone https://https://github.com/KeKsBoTer/cinematic-gaussians.git --recursive
cd cinematic-gaussians
```

2. Create environment and install dependencies
```
conda create -n cin3dgs python=3.12 -y
conda activate cin3dgs
```

## Training

```
python train.py
    -s <scene folder> \\
    -m <model output folder> \\
    --eval \\
    --test_iterations 7000 15000 30000 \\
    --densify_grad_threshold 0.00005 \\
    --save_iterations 30000 \\
```

If you run into VRAM issues try increasing the `densify_grad_threshold` parameter to reduce the number of Gaussians.

## Compression

To compress the model with [our method](https://keksboter.github.io/c3dgs/) run the compression script on the reconstructed scenes:

```
python compress.py \\
    -m <training_output_folder> \\
    --eval \\
    --output_vq <compression_output_folder> \\
    --load_iteration 30000
```

This script will also evaluate the scene on the training and test images and report PSNR and SSIM.

## View Selection

The code for the view selection can be found [here](https://github.com/chrismile/vpt_denoise/tree/main/pydens2d)
