import os 
import io
import torch
from torch.utils.data import Dataset
import math
from glob import glob
import re
from .meta import DEVICE_INFOS
import numpy as np
import pandas as pd
from PIL import Image
import random
import pathlib


class WFASDataset(Dataset):    
    def __init__(self, 
                 root_dir, 
                 transform=None, 
                 is_train=False,
                 map_size=32, 
                 UUID=-1,
                 img_size=256):
        self.root_dir = root_dir
        self.is_train = is_train
        annotation_file = ["dev_and_test.txt", "train.txt"][self.is_train]
        self.video_df = pd.read_csv(os.path.join(self.root_dir,
                                                 annotation_file),
                                    header=None)
        self.transform = transform
        self.dataset_name = "WFAS"
        self.map_size = map_size
        self.UUID = UUID
        self.image_size = img_size

    def __len__(self):
        return len(self.video_df)
    
    def shuffle(self):
        if self.is_train:
            self.video_df = self.video_df.sample(frac=1)
        
    def get_client_from_video_name(self, video_name):
        return video_name.rpartition('/')[-1].rpartition('.')[0]

    def __getitem__(self, idx):
        idx = idx % len(self.video_df) # Incase testing with many frame per video
        video_path = self.video_df.iloc[idx, 1]
        # 1 = live, 0 = spoof
        spoofing_label = int(self.video_df.iloc[idx, 0])
        image_dir = video_path

        device_tag = 'live' if spoofing_label else 'spoof'
        client_id = self.get_client_from_video_name(image_dir)

        image_x = self.sample_image(image_dir,is_train=self.is_train, rep=None)
        if self.is_train:
            transformed_image1 = self.transform(image_x)
            transformed_image2 = self.transform(image_x)
        else:
            transformed_image1, transformed_image2 = (self.transform(image_x),)*2

        sample = {"image_x_v1": transformed_image1,
                  "image_x_v2": transformed_image2,
                  "label": spoofing_label,
                  "UUID": self.UUID,
                  'device_tag': device_tag,
                  'video': video_path.rpartition('/')[-1],
                  'client_id': client_id,
                  'video_path': video_path}
        return sample


    def sample_image(self, image_dir, is_train=False, rep=None):
        try:
            image = Image.open(image_dir)
            image = image.convert("RGB")
        except OSError:
            raise OSError(image_dir)
        return image

class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im
    
class RandomCutout(object):
    def __init__(self, n_holes, p=0.5):
        """
        Args:
            n_holes (int): Number of patches to cut out of each image.
            p (int): probability to apply cutout
        """
        self.n_holes = n_holes
        self.p = p

    def rand_bbox(self, W, H, lam):
        """
        Return a random box
        """
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
    
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if  np.random.rand(1) > self.p:
            return img
        
        h = img.size(1)
        w = img.size(2)
        lam = np.random.beta(1.0, 1.0)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(w, h, lam)
        for n in range(self.n_holes):
            img[:,bby1:bby2, bbx1:bbx2] = img[:,bby1:bby2, bbx1:bbx2].mean(dim=[-2,-1],keepdim=True)
        return img
    
class RandomJPEGCompression(object):
    def __init__(self, quality_min=30, quality_max=90, p=0.5):
        assert 0 <= quality_min <= 100 and 0 <= quality_max <= 100
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p
    def __call__(self, img):
        if  np.random.rand(1) > self.p:
            return img
        # Choose a random quality for JPEG compression
        quality = np.random.randint(self.quality_min, self.quality_max)
        
        # Save the image to a bytes buffer using JPEG format
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        
        # Reload the image from the buffer
        img = Image.open(buffer)
        return img
    
class RoundRobinDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_len = sum(self.lengths)
        
    def __getitem__(self, index):
        # Determine which dataset to sample from
        dataset_id = index % len(self.datasets)
        
        # Adjust index to fit within the chosen dataset's length
        inner_index = index // len(self.datasets)
        inner_index = inner_index % self.lengths[dataset_id]
        return self.datasets[dataset_id][inner_index]
    
    def shuffle(self):
        for dataset in self.datasets:
            dataset.shuffle()

    def __len__(self):
        # Return the length of the largest dataset times the number of datasets
        return max(self.lengths) * len(self.datasets)
