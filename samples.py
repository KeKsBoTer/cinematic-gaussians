from PIL import Image
import itertools
from tqdm import tqdm
from os import makedirs, path

scenes = ["brain", "kidney", "whole_body", "skull"]
resolutions = [2]
presets = [1, 2, 3, 4]
images_idx = {"brain": 48, "kidney": 0, "whole_body": 56, "skull": 56}
args = itertools.product(scenes, presets, resolutions)
for scene, preset, res in tqdm(list(args)):
    if scene == "whole_body" and preset == 4 or scene == "skull":
        res = 4
    scene_name = f"{scene}_preset{preset}_{res}"
    if not path.exists(
        path.join(
            "/home/niedermayr/Projects/gaussian_me/eval",
            scene_name,
        )
    ):
        continue
    render_path = path.join(
        "/home/niedermayr/Projects/gaussian_me/eval",
        scene_name,
        f"test/renders/img_{images_idx[scene]}.png",
    )
    gt_path = path.join(
        "/home/niedermayr/Projects/gaussian_me/eval",
        scene_name,
        f"test/gt/img_{images_idx[scene]}.png",
    )
    render = Image.open(render_path).resize((1024, 1024))
    gt = Image.open(gt_path).resize((1024, 1024))
    target_dir = path.join("docs/static/img/samples", scene_name)
    makedirs(target_dir, exist_ok=True)
    render.save(path.join(target_dir, "render.png"))
    gt.save(path.join(target_dir, "gt.png"))

    target_dir = path.join("docs/static/img/thumbnails")
    makedirs(target_dir, exist_ok=True)
    render.resize((512, 512)).convert("RGB").save(
        path.join(target_dir, f"{scene_name}.jpg")
    )
