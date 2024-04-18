<div align="center" >

# Novel View Synthesis for Cinematic Anatomy on Mobile and Immersive Displays


<font size="4">
 Simon Niedermayr<sup>1</sup>, Christoph Neuhauser<sup>1</sup>, Kaloian Petkov<sup>2</sup>, Klaus Engel<sup>2</sup>, Rüdiger Westermann<sup>1</sup>

</font>
<br>

<font size="4">
<sup>1</sup>Technical University of Munich, <sup>2</sup>Siemens Healthineers 
</font>

<a href="https://keksboter.github.io/cinematic-gaussians/">Webpage</a> | <a href="https://arxiv.org/abs/2404.11285">arXiv</a> 

<img style="backgroud-color:white" src="docs/static/img/pipeline.svg" alt="Comrpression Pipeline"/>
</div>

## Abstract
Interactive photorealistic visualization of 3D anatomy (i.e., Cinematic Anatomy) is used in medical education to explain the structure of the human body. It is currently restricted to frontal teaching scenarios, where the demonstrator needs a powerful GPU and high-speed access to a large storage device where the dataset is hosted. We demonstrate the use of novel view synthesis via compressed 3D Gaussian splatting to overcome this restriction and to enable students to perform cinematic anatomy on lightweight mobile devices and in virtual reality environments. We present an automatic approach for finding a set of images that captures all potentially seen structures in the data. By mixing closeup views with images from a distance, the splat representation can recover structures up to the voxel resolution. The use of Mip-Splatting enables smooth transitions when the focal length is increased. Even for GB datasets, the final renderable representation can usually be compressed to less than 70 MB, enabling interactive rendering on low-end devices using rasterization. 

## Citation
If you find our work useful, please cite:
```
@misc{niedermayr2024novel,
    title={Novel View Synthesis for Cinematic Anatomy on Mobile and Immersive Displays},
    author={Simon Niedermayr and Christoph Neuhauser and Kaloian Petkov and Klaus Engel and Rüdiger Westermann},
    year={2024},
    eprint={2404.11285},
    archivePrefix={arXiv},
    primaryClass={cs.GR}
}
```

## Code

**Code will be released soon** 