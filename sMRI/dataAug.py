import os
import glob
import random
import numpy as np
import nibabel as nib
import cv2

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate
from tqdm import tqdm

def load_nii_slices(nii_path):
    nii_obj = nib.load(nii_path)
    volume = nii_obj.get_fdata()
    volume = np.asarray(volume, dtype=np.float32)

    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-7)
    
    slices = []
    for z in range(volume.shape[-1]):
        slice_2d = volume[:, :, z]
        slices.append(slice_2d)
    return slices

def apply_canny_and_crop(slice_2d, threshold1=50, threshold2=150, crop_pad=5):
    slice_uint8 = (slice_2d * 255).astype(np.uint8)
    edges = cv2.Canny(slice_uint8, threshold1, threshold2)
    ys, xs = np.where(edges > 0)
    if len(xs) == 0 or len(ys) == 0:
        return slice_2d
    
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_min = max(0, x_min - crop_pad)
    x_max = min(slice_2d.shape[1], x_max + crop_pad)
    y_min = max(0, y_min - crop_pad)
    y_max = min(slice_2d.shape[0], y_max + crop_pad)
    
    return slice_2d[y_min:y_max, x_min:x_max]

def add_salt_noise(img, salt_prob=0.05):
    noisy = img.copy()
    h, w = noisy.shape
    num_salt = int(h * w * salt_prob)
    coords = [
        (random.randint(0, h - 1), random.randint(0, w - 1))
        for _ in range(num_salt)
    ]
    for (y, x) in coords:
        noisy[y, x] = 1.0
    return noisy

def augment_slice(slice_2d):

    aug_slices = []
    # original
    aug_slices.append(slice_2d)
    # flip left-right
    aug_slices.append(np.fliplr(slice_2d))
    # rotate 90
    aug_slices.append(np.rot90(slice_2d, k=1))
    # rotate 180
    aug_slices.append(np.rot90(slice_2d, k=2))
    # salt
    aug_slices.append(add_salt_noise(slice_2d, 0.05))
    # rotate angles
    aug_slices.append(rotate(slice_2d, angle=60))



    # IMPORTANT: resize each variant to a fixed size (224x224) so shapes match
    resized_augs = []
    for aug in aug_slices:
        aug_resized = cv2.resize(aug, (224, 224), interpolation=cv2.INTER_AREA)
        resized_augs.append(aug_resized)
        
    return resized_augs

def is_majority_black(slice_2d, black_thresh=0.05, ratio_thresh=0.9):
    n_black = np.count_nonzero(slice_2d < black_thresh)
    total = slice_2d.size
    black_ratio = n_black / float(total)
    return black_ratio > ratio_thresh

class AutismDataset(Dataset):
    def __init__(self, list_of_files, transform=None, resize_shape=(224,224)):
        self.list_of_files = list_of_files
        self.transform = transform
        self.resize_shape = resize_shape
        self.data = []
        
        
        for nii_path, label in tqdm(self.list_of_files, desc="Processing files"):
            print(f"Processing {nii_path}...")
            slices_2d = load_nii_slices(nii_path)
            for slc in slices_2d:
                # # 1) Canny+crop
                cropped = apply_canny_and_crop(slc)
                if np.allclose(cropped, 0.0):
                    continue
                # 2) augment
                aug_slices = augment_slice(cropped)
                # 3) accumulate
                if is_majority_black(slc, black_thresh=0.15, ratio_thresh=0.7):
                    continue
                for aug_slc in aug_slices:
                    self.data.append((aug_slc, label))

                

                # if self.resize_shape is not None:
                #     # (width, height) = (224,224) in OpenCV
                #     slc_resized = cv2.resize(
                #         slc, 
                #         dsize=self.resize_shape, 
                #         interpolation=cv2.INTER_AREA
                #     )
                #     self.data.append((slc_resized, label))
                # else:
                #     self.data.append((slc, label))

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = img.astype(np.float32)
        
        img = np.expand_dims(img, axis=0)  # [1, H, W]
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.float64)

def show_images(images, labels, num_images=4):

    num_images = min(num_images, images.shape[0])
    

    plt.figure(figsize=(4 * num_images, 4))
    
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)

        img = images[i, 0].cpu().numpy()
        
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

class AutismDataset_raw(Dataset):
    def __init__(self, list_of_files, transform=None, resize_shape=(224,224)):

        self.list_of_files = list_of_files
        self.transform = transform
        self.resize_shape = resize_shape
        self.data = []
        
        for nii_path, label in self.list_of_files:
            slices_2d = load_nii_slices(nii_path)
            for slc in slices_2d:
                if is_majority_black(slc, black_thresh=0.15, ratio_thresh=0.7):
                    continue
                
                if self.resize_shape is not None:
                    slc_resized = cv2.resize(
                        slc, 
                        dsize=self.resize_shape, 
                        interpolation=cv2.INTER_AREA
                    )
                    self.data.append((slc_resized, label))
                else:
                    self.data.append((slc, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = img.astype(np.float32)
        
        img = np.expand_dims(img, axis=0)  # [1, H, W]
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.float64)