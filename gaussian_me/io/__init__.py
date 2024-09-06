import os
import random
from typing import List, Self, Tuple
import torch
from torchvision.transforms.functional import pil_to_tensor
from gaussian_me.cameras import Camera
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


from gaussian_me.io.dataset_readers import (
    BasicPointCloud,
    CameraInfo,
    SceneInfo,
    readColmapSceneInfo,
    readNerfSyntheticInfo,
    readSiemensSceneInfo,
    readVolumeSceneInfo,
)


class CamerasDataset(torch.utils.data.Dataset):

    @classmethod
    def from_folder(cls, source_path: str, images="images") -> Self:
        if os.path.exists(os.path.join(source_path, "cameras.json")):
            print("Found cameras.json file, assuming Volume data set!")
            scene_info = readVolumeSceneInfo(source_path, images)
        elif os.path.exists(os.path.join(source_path, "sparse")):
            print("Found sparse folder, assuming colmap data set!")
            scene_info = readColmapSceneInfo(source_path, images)
        elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = readNerfSyntheticInfo(source_path)
        elif os.path.exists(os.path.join(source_path, "transforms.json")):
            print("Found transforms.json file, assuming Siemens data set!")
            scene_info = readSiemensSceneInfo(source_path, images)
        else:
            raise Exception("Could not recognize scene type!")

        return cls(scene_info)

    def __init__(
        self,
        scene_info: SceneInfo,
    ):
        self.scene_info = scene_info

    def __len__(self):
        return len(self.scene_info.cameras)

    def __getitem__(self, idx) -> CameraInfo:
        cam_info = self.scene_info.cameras[idx]
        with open(cam_info.image_path, "rb") as f:
            img = Image.open(f)
            image = pil_to_tensor(img.convert("RGBA"))

        if image.shape[0] == 3:
            image = torch.cat([image, torch.ones_like(image[:1])], 0)

        return Camera(
            colmap_id=cam_info.uid,
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            image=image,
            image_name=cam_info.image_name,
            uid=cam_info.uid,
            data_device=image.device,
        )

    def cameras_extent(self) -> float:
        return self.scene_info.nerf_normalization["radius"]

    def init_pc(self) -> BasicPointCloud:
        return self.scene_info.point_cloud

    def split_train_val(self) -> Tuple[Self, Self]:
        train_dataset = []
        val_dataset = []
        for i, cam_info in enumerate(self.scene_info.cameras):
            if i % 8 == 0:
                val_dataset.append(cam_info)
            else:
                train_dataset.append(cam_info)

        return (
            CamerasDataset(
                SceneInfo(
                    ply_path=self.scene_info.ply_path,
                    cameras=train_dataset,
                    point_cloud=self.scene_info.point_cloud,
                    nerf_normalization=self.scene_info.nerf_normalization,
                )
            ),
            CamerasDataset(
                SceneInfo(
                    ply_path=self.scene_info.ply_path,
                    cameras=val_dataset,
                    point_cloud=self.scene_info.point_cloud,
                    nerf_normalization=self.scene_info.nerf_normalization,
                )
            ),
        )

def sample_rate(step: int, max_steps: int) -> torch.Tensor:
    clamp = lambda x: min(max(x, 0), 1)
    x = clamp(step / max_steps)
    rate = torch.tensor(
        [
            clamp((1 - x) * (1 + 0.5) - 0.5),
            x * 0.2,
            x * 0.3,
            x * 0.4,
        ]
    )
    rate /= rate.sum()
    return rate
