import gc
import itertools
import os
import random
import subprocess
import sys
from glob import glob
from itertools import product
from os import makedirs, path

import torch
from tqdm import tqdm

ALL_GPUS = [0]#list(range(8))
FREE_GPUS = ALL_GPUS.copy()
RUNNING = []

def run_cmd(cmd,logdir):
    global RUNNING
    global FREE_GPUS
    if len(FREE_GPUS)==0:
        while True:
            found = False
            RUNNNING_NEW = []
            for (job,gpu_id) in RUNNING:
                if job.poll() != None:
                    FREE_GPUS.append(gpu_id)
                    found = True
                else:
                    RUNNNING_NEW.append((job,gpu_id))
            RUNNING = RUNNNING_NEW
            if found:
                break
    gpu_id = FREE_GPUS.pop()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
    makedirs(logdir,exist_ok=True)
    stdout = open(os.path.join(logdir,"stdout.txt"),"wb")
    stderr = open(os.path.join(logdir,"stderr.txt"),"wb")
    p = subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr)
    RUNNING.append((p,gpu_id))
    
def wait_for_all():
    for job,_ in RUNNING:
        job.wait()


scenes = ["brain""torso","brain", "kidney", "torso", "whole_body"]
resolutions = [1,4]
presets = [1,2]

args = list(itertools.product(scenes,presets,resolutions))

# args+=list(itertools.product(["whole_body"],[4],[4]))

for (scene,preset,res) in tqdm(args):
    model_path =  path.join(
            "eval_brain_hq",
            f"{scene}_preset{preset}_{res}"
        )

    if os.path.exists(model_path):
        print(f"skipping {model_path} as it already exists")
        continue

    scene_path = os.path.join("/home/niedermayr/datasets/siemens",scene,f"preset{preset}")
    images = f"images_{res}" if res != 1 else "images"
    if os.path.exists(os.path.join(scene_path,images)):

        cmd = ["python","train.py",
            "-s", scene_path,
            "-m", model_path,
            "--eval",
            "--test_iterations", "7000", "15000", "30000",
            "--images",images, 
            "--densify_grad_threshold", "0.00005" if res!=1 else "0.0001",
            "--save_iterations", "7000", "15000", "30000"
        ]
        print(" ".join(cmd))
        run_cmd(cmd,os.path.join(model_path,"log"))

            
wait_for_all()