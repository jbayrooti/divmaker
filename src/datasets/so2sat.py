import os
from skimage import io
import copy
import numpy as np
import random
from glob import glob
import h5py

import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from src.datasets.root_paths import DATA_ROOTS


class BaseSo2Sat(data.Dataset):
    CLASSES = ['Compact High-Rise',
            'Compact Midrise',
            'Compact Low-Rise',
            'Open High-Rise',
            'Open Midrise',
            'Open Low-Rise',
            'Lightweight Low-Rise',
            'Large Low-Rise',
            'Sparsely Built',
            'Heavy Industry',
            'Dense Trees',
            'Scattered Trees',
            'Brush, Scrub',
            'Low Plants',
            'Bare Rocks or Paved',
            'Bare Soil or Sand',
            'Water']

    def __init__(
            self, 
            root=DATA_ROOTS["so2sat"], 
            train=True, 
            image_transforms=None,
            seed=42,
            sen='sen1',
        ):
        super().__init__()
        self.root = root
        self.train = train
        self.image_transforms = image_transforms

        if self.train:
            fileName = 'training.h5'
        else:
            fileName = 'validation.h5'

        fid = h5py.File(self.root + "/" + fileName,'r')
        self.data = np.array(fid[sen], dtype=np.float32).transpose(0,3,1,2)
        self.labels = np.argmax(np.array(fid['label']), axis=1)
        self.targets = copy.deepcopy(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        image = torch.tensor(self.data[index])
        if self.image_transforms:
            image = self.image_transforms(image)
        return image, label

    def __len__(self):
        return self.data.shape[0]


class So2Sat_Sen1(BaseSo2Sat):
    NUM_CLASSES = 17
    MULTI_LABEL = False
    NUM_CHANNELS = 8
    FILTER_SIZE = 32

    def __init__(
            self, 
            root=DATA_ROOTS["so2sat"], 
            train=True, 
            image_transforms=None,
        ):
        super().__init__()
        self.dataset = BaseSo2Sat(
            root=root, 
            train=train, 
            image_transforms=image_transforms,
            sen='sen1',
        )
    
    def __getitem__(self, index):
        img_data, label = self.dataset.__getitem__(index)
        img2_data, _ = self.dataset.__getitem__(index)
        data = [index, img_data.float(), img2_data.float(), label, label]
        return tuple(data)

    def __len__(self):
        return len(self.dataset)


class So2Sat_Sen2(BaseSo2Sat):
    NUM_CLASSES = 17
    MULTI_LABEL = False
    NUM_CHANNELS = 10
    FILTER_SIZE = 32

    def __init__(
            self, 
            root=DATA_ROOTS["so2sat"], 
            train=True, 
            image_transforms=None,
        ):
        super().__init__()
        self.dataset = BaseSo2Sat(
            root=root, 
            train=train, 
            image_transforms=image_transforms,
            sen='sen2',
        )
    
    def __getitem__(self, index):
        img_data, label = self.dataset.__getitem__(index)
        img2_data, _ = self.dataset.__getitem__(index)
        data = [index, img_data.float(), img2_data.float(), label, label]
        return tuple(data)

    def __len__(self):
        return len(self.dataset)