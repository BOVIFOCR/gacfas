import os 
import io
import torch
from torch.utils.data import Dataset
import math
from glob import glob
import re
from .meta import DEVICE_INFOS
import numpy as np
from PIL import Image
import random
import pathlib


USE_CENTER_FRAME_FOR_TESTING = False

def list_dirs_at_depth(p, depth):
    p = pathlib.Path(p)
    if depth < 0:
        return []
    elif depth == 0:
        return [p]
    else:
        ans = []
        for q in p.iterdir():
            if q.is_dir():
                ans += list_dirs_at_depth(q, depth-1)
        return ans
    
class RoseDataset(Dataset):    
    def __init__(self, 
                 dataset_name, 
                 root_dir, 
                 is_train=True, 
                 label=None, 
                 transform=None, 
                 map_size=32, 
                 UUID=-1,
                 img_size=256,
                 test_per_video=1):
        self.is_train = is_train
        self.video_list = list(filter(lambda x: x.is_dir(), pathlib.Path(root_dir).iterdir()))

        if label is not None and label != 'all':
            if label == 'live':
                self.video_list = list(filter(lambda x: self.is_live(x), self.video_list))
            else:
                self.video_list = list(filter(lambda x: not self.is_live(x), self.video_list))

        # train_subjects = ['2', '3', '4', '5', '6', '7', '9', '10', '11', '12']
        # if self.is_train:
        #     self.video_list = list(filter(
        #         lambda x: self.get_client_from_video_name(x) in train_subjects,
        #         self.video_list))
        # else:
        #     self.video_list = list(filter(
        #         lambda x: self.get_client_from_video_name(x) not in train_subjects,
        #         self.video_list))

        real_count = len(list(filter(lambda x: self.is_live(x), self.video_list)))
        spoof_count = len(self.video_list) - real_count
        print(f"({dataset_name}) Total video: {len(self.video_list)}: {real_count} vs. {spoof_count}")
            
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.transform = transform
        self.map_size = map_size
        self.UUID = UUID
        self.image_size = img_size

        if not is_train:
            self.frame_per_video = test_per_video
            self.video_list = sum([self.video_list]*test_per_video, [])
        else:
            self.frame_per_video = 1

        self.init_frame_list()

    def is_live(self, video_path):
        return video_path.name.startswith('G')

    def __len__(self):
        return len(self.video_list)
    
    def shuffle(self):
        if self.is_train:
            random.shuffle(self.video_list)
        
    def init_frame_list(self):
        """
        Create dictionary mapping video paths to a list of face image paths

        In test mode (when self.is_train is false), only one frame
        is needed from each video, but in the code made available
        by the authors they use all frames
        """
        self.video_frame_list = {video: [] for video in self.video_list}
        for p in self.video_frame_list:
            face_img_list = list(map(pathlib.Path, p.iterdir()))
            # obs: in test mode (not self.is_train) only one frame is needed
            if self.is_train or not USE_CENTER_FRAME_FOR_TESTING:
                self.video_frame_list[p] = face_img_list
            else:
                sz = len(crop_faces_list)
                self.video_frame_list[p] = [crop_faces_list[sz//2]]
        return True
    
    def get_client_from_video_name(self, video_name):
        video_name = str(video_name).rpartition('/')[-1]
        client_id = video_name.split('_')[-2]
        return client_id
    
    def __getitem__(self, idx):
        idx = idx % len(self.video_list) # Incase testing with many frame per video
        video_name = self.video_list[idx]
        spoofing_label = int(self.is_live(video_name))
        device_tag = 'live' if spoofing_label else 'spoof'

        client_id = self.get_client_from_video_name(video_name)

        image_dir = video_name

        if self.is_train:
            image_x, _, _, = self.sample_image(image_dir, is_train=True)
            transformed_image1 = self.transform(image_x)           
            transformed_image2 = self.transform(image_x, )
        else:
            image_x, _, _ = self.sample_image(image_dir, is_train=False,
                                              rep=None)
            transformed_image1 = transformed_image2 = self.transform(image_x)

        sample = {"image_x_v1": transformed_image1,
                  "image_x_v2": transformed_image2,
                  "label": spoofing_label,
                  "UUID": self.UUID,
                  'device_tag': device_tag,
                  'video': str(video_name),
                  'video_path': str(video_name),
                  'client_id': client_id}
        return sample


    def sample_image(self, image_dir, is_train=False, rep=None):
        """
        rep is the parameter from the __getitem__ function to reduce randomness of test phase
        """
        image_path = str(np.random.choice(self.video_frame_list[image_dir]))
        image_id = int(image_path.split('/')[-1].split('_')[-1].split('.')[0])

        try: 
            info = None
            image = Image.open(image_path)
        except:
            if is_train: 
                return self.sample_image(image_dir, is_train)
            else:
                raise ValueError(f"Error in the file {image_path}")
        return image, info, image_id * 5

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
