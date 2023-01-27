import os
from skimage import io
import copy
import numpy as np
import random
from glob import glob
import json
from sklearn.preprocessing import MultiLabelBinarizer

import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from src.datasets.root_paths import DATA_ROOTS

CLASSES = ['Sea and ocean',
            'Coniferous forest', 
            'Mixed forest', 
            'Moors and heathland', 
            'Transitional woodland/shrub', 
            'Sparsely vegetated areas', 
            'Discontinuous urban fabric', 
            'Non-irrigated arable land', 
            'Pastures', 
            'Complex cultivation patterns', 
            'Broad-leaved forest', 
            'Water bodies', 
            'Land principally occupied by agriculture, with significant areas of natural vegetation', 
            'Vineyards', 
            'Agro-forestry areas', 
            'Industrial or commercial units', 
            'Airports', 
            'Water courses', 
            'Natural grassland', 
            'Construction sites', 
            'Sclerophyllous vegetation', 
            'Peatbogs', 
            'Rice fields', 
            'Continuous urban fabric', 
            'Olive groves', 
            'Permanently irrigated land', 
            'Mineral extraction sites', 
            'Annual crops associated with permanent crops', 
            'Dump sites', 
            'Green urban areas', 
            'Intertidal flats', 
            'Bare rock', 
            'Fruit trees and berry plantations', 
            'Salt marshes', 
            'Road and rail networks and associated land', 
            'Estuaries', 
            'Inland marshes', 
            'Sport and leisure facilities', 
            'Beaches, dunes, sands', 
            'Coastal lagoons', 
            'Salines', 
            'Port areas', 
            'Burnt areas']

class BaseBigEarthNet(data.Dataset):
    NUM_CLASSES = 43
    MULTI_LABEL = True
    NUM_CHANNELS = 12
    FILTER_SIZE = 120

    def __init__(
            self, 
            root=DATA_ROOTS["bigearthnet"], 
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
        all_sample_paths = np.array(os.listdir(self.root))
        num_samples = len(all_sample_paths)
        labels = []
        for i in range(num_samples):
            sample_path = all_sample_paths[i]
            metadata_path = glob(os.path.join(self.root, sample_path, '*.json'))[0]
            class_names = set(json.load(open(metadata_path))['labels'])
            labels.append(class_names)

        encoder = MultiLabelBinarizer(classes=CLASSES, sparse_output=False)
        encoded_labels = encoder.fit_transform(labels)
        num_samples = len(all_sample_paths)
        indices = np.arange(num_samples)
        self.rs.shuffle(indices)
        train_indices = indices[:int(num_samples * train_frac)]
        test_indices = indices[int(num_samples * train_frac):]
        train_paths = all_sample_paths[train_indices]
        test_paths = all_sample_paths[test_indices]
        train_labels = encoded_labels[train_indices]
        test_labels = encoded_labels[test_indices]
        return train_paths, test_paths, train_labels, test_labels

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        img_paths = glob(os.path.join(self.root, path, '*.tif'))
        image = []
        for i, img_path in enumerate(img_paths):
            img = np.asarray(io.imread_collection(img_path), dtype=np.float32) # one of (1, 20, 20), (1, 60, 60), (1, 120, 120)
            resized_img = transforms.Resize(120)(torch.tensor(img))
            image.append(resized_img)
        image = torch.vstack(image)  # (12, 120, 120)
        if self.image_transforms:
            image = self.image_transforms(image)
        return image, label

    def __len__(self):
        return len(self.paths)


class BigEarthNet(BaseBigEarthNet):

    def __init__(
            self, 
            root=DATA_ROOTS["bigearthnet"], 
            train=True, 
            image_transforms=None,
        ):
        super().__init__()
        self.dataset = BaseBigEarthNet(
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
