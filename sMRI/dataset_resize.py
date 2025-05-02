import os
import glob
import numpy as np
from tqdm import tqdm
from dataAug import apply_canny_and_crop, augment_slice, is_majority_black
import cv2

base_dir = "preproc_out"
groups = [f"group{i}" for i in range(1, 6)]

for group in groups:
    group_path = os.path.join(base_dir, group)
    for root, _, files in os.walk(group_path):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.endswith("aug0.npy"):
                file_path = os.path.join(root, file)
                data = np.load(file_path)
                if data.shape != (224, 224):
                    data = cv2.resize(data, (224, 224), interpolation=cv2.INTER_AREA)
                    np.save(file_path, data)
                    