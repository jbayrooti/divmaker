import os
import copy
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
from os.path import join
from itertools import chain
from collections import defaultdict

import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from src.datasets.root_paths import DATA_ROOTS


class BaseRESISC45(data.Dataset):
    NUM_CLASSES = 45
    MULTI_LABEL = False
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    CLASSES = ['airplane', 'airport', 'baseball diamond', 'basketball court', 'beach', 'bridge', 'chaparral', 'church',
                'circular farmland', 'cloud', 'commercial area', 'dense residential', 'desert', 'forest', 'freeway',
                'golf course', 'ground track field', 'harbour', 'industrial area', 'intersection', 'island', 'lake', 
                'meadow', 'medium residential', 'mobile home park', 'mountain', 'overpass', 'palace', 'parking lot', 
                'railway', 'railway station', 'rectangular farmland', 'river', 'roundabout', 'runway', 'sea ice', 'ship',
                'snowberg', 'sparse residential', 'stadium', 'storage tank', 'tennis court', 'terrace', 
                'thermal power station', 'wetland']

    def __init__(
            self, 
            root=DATA_ROOTS["resisc45"], 
            train=True, 
            image_transforms=None,
            seed=42,
        ):
        super().__init__()
        self.root = root
        self.train = train
        self.image_transforms = image_transforms

        self.rs = np.random.RandomState(seed)
        train_paths, test_paths, train_labels, test_labels = self.train_test_split()
        if train:
            self.paths = train_paths
            self.labels = train_labels
        else:
            self.paths = test_paths
            self.labels = test_labels
        self.targets = copy.deepcopy(self.labels)
    
    def train_test_split(self, train_frac=0.8):
        class_dirs = os.listdir(self.root)
        class_to_label = dict(zip(class_dirs, range(len(class_dirs))))
        train_paths, test_paths = [], []
        train_labels, test_labels = [], []
        for class_dir in class_dirs:
            label = class_to_label[class_dir]
            class_img_paths = glob(os.path.join(self.root, class_dir, '*.jpg'))
            class_img_paths = np.array(class_img_paths)
            num_class_img = len(class_img_paths)
            indices = np.arange(num_class_img)
            self.rs.shuffle(indices)
            train_indices = indices[:int(num_class_img * train_frac)]
            test_indices = indices[int(num_class_img * train_frac):]
            train_img_paths = class_img_paths[train_indices]
            test_img_paths = class_img_paths[test_indices]
            train_paths.append(train_img_paths)
            test_paths.append(test_img_paths)
            train_labels.append(np.ones(len(train_img_paths)) * label)
            test_labels.append(np.ones(len(test_img_paths)) * label)
        train_paths = np.concatenate(train_paths)
        test_paths = np.concatenate(test_paths)
        train_labels = np.concatenate(train_labels)
        test_labels = np.concatenate(test_labels)
        return train_paths, test_paths, train_labels, test_labels

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        image = Image.open(path)
        if self.image_transforms:
            image = self.image_transforms(image)
        return image, label

    def __len__(self):
        return len(self.paths)


class RESISC45(BaseRESISC45):

    def __init__(
            self, 
            root=DATA_ROOTS["resisc45"], 
            train=True, 
            image_transforms=None,
        ):
        super().__init__()
        self.dataset = BaseRESISC45(
            root=root, 
            train=train, 
            image_transforms=image_transforms,
        )
    
    def __getitem__(self, index):
        img_data, label = self.dataset.__getitem__(index)
        img2_data, _ = self.dataset.__getitem__(index)
        data = [index, img_data.float(), img2_data.float(), label, label]
        return tuple(data)

    def __len__(self):
        return len(self.dataset)