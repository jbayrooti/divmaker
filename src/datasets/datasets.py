from torchvision import transforms
import torch
import random
from PIL import ImageFilter, Image

from src.datasets.resisc45 import RESISC45
from src.datasets.eurosat import EuroSAT
from src.datasets.so2sat import So2Sat_Sen1, So2Sat_Sen2
from src.datasets.bigearthnet import BigEarthNet
from src.datasets.data_statistics import get_data_mean_and_stdev

DATASET = {
    'eurosat': EuroSAT,
    'so2sat_sen1': So2Sat_Sen1,
    'so2sat_sen2': So2Sat_Sen2,
    'bigearthnet': BigEarthNet,
    'resisc45': RESISC45,
}

def get_image_datasets(
        dataset_name,
        default_augmentations='none',
        resize_imagenet_to_112=False,
        resize_imagenet_to_64=False,
        resize_imagenet_to_32=False,
        resize_to_224=False,
        mask=False,
        zscore=False
    ):
    load_transforms = TRANSFORMS[default_augmentations]
    train_transforms, test_transforms = load_transforms(
        dataset=dataset_name, 
        resize_imagenet_to_112=resize_imagenet_to_112,
        resize_imagenet_to_64=resize_imagenet_to_64,
        resize_imagenet_to_32=resize_imagenet_to_32,
        resize_to_224=resize_to_224
    )
    train_dataset = DATASET[dataset_name](
        train=True,
        image_transforms=train_transforms
    )
    val_dataset = DATASET[dataset_name](
        train=False,
        image_transforms=test_transforms,
    )
    return train_dataset, val_dataset


def load_image_transforms(
        dataset,
        resize_imagenet_to_112=False, 
        resize_imagenet_to_64=False,
        resize_imagenet_to_32=False,
        resize_to_224=False,
    ):
    if dataset == 'resisc45':
        if resize_imagenet_to_112:
            crop_size = 112
        elif resize_imagenet_to_64:
            crop_size = 64
        elif resize_imagenet_to_32:
            crop_size = 32
        else:
            crop_size = 224
        train_transforms = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])
    elif dataset in ['eurosat', 'so2sat_sen1', 'so2sat_sen2', 'bigearthnet']:
        if resize_imagenet_to_112:
            crop_size = 112
        elif resize_imagenet_to_64:
            crop_size = 64
        elif resize_imagenet_to_32:
            crop_size = 32
        else:
            crop_size = 224
        train_transforms = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
        ])
    else:
        return None, None

    return train_transforms, test_transforms


def load_default_transforms(
        dataset, 
        resize_imagenet_to_112=False, 
        resize_imagenet_to_64=False,
        resize_imagenet_to_32=False,
        resize_to_224=False,
    ):
    if dataset == 'resisc45':
        mean, std = get_data_mean_and_stdev(dataset)
        if resize_imagenet_to_112:
            resize_size = 128
            crop_size = 112
        elif resize_imagenet_to_64:
            resize_size = 73
            crop_size = 64
        elif resize_imagenet_to_32:
            resize_size = 37
            crop_size = 32
        else:
            resize_size = 256
            crop_size = 224
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif dataset in ['eurosat', 'so2sat_sen1', 'so2sat_sen2', 'bigearthnet']:
        mean, std = get_data_mean_and_stdev(dataset)
        if resize_imagenet_to_112:
            resize_size = 128
            crop_size = 112
        elif resize_imagenet_to_64:
            resize_size = 73
            crop_size = 64
        elif resize_imagenet_to_32:
            resize_size = 37
            crop_size = 32
        else:
            resize_size = 256
            crop_size = 224
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return None, None
    return train_transforms, test_transforms


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


TRANSFORMS = {
    True: load_default_transforms, # For legacy models
    False: load_image_transforms, # For legacy models
    'all': load_default_transforms,
    'none': load_image_transforms, 
}