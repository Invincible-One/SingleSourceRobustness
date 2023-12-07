import os
import PIL import Image

import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset

from models import model_config
from data.datasets import CelebA
from data.datasets import CUB



data_config = {
        "celebA": {
            "constructor": CelebA,
            "orig_shape": (178, 218),   #shape is (width, height)
            },
        "cub": {
            "constructor": CUB,
            },
        }



def get_celebA_transform(target_resolution, train):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)

    if not train:
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
            ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2,
                ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], 
                [0.229, 0.224, 0.225]),
            ])

    return transform

def get_cub_transform(target_resolution, train):
    scale = 256.0 / 224.0

    if not train:
        transform = transforms.Compose([
            transforms.Resize((
                int(target_resolution[0] * scale),
                int(target_resolution[1] * scale),
            )),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
            ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2,
                ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
            ])
        
    return transform



def get_data(args, split):
    assert split in ('train', 'valid', 'test', "all")
    if split == 'train':
        train = True
    elif split in ('valid', 'test'):
        train = False
    else:
        raise NotImplementedError
    
    target_resolution = model_config[args.model]["target_resolution"]
    assert target_resolution is not None
    
    if args.dataset == "celebA":
        transform = get_celebA_transform(target_resolution=target_resolution, train=train)
    elif args.dataset == "cub":
        trainsform = get_cub_transform(target_resolution=target_resolution, train=train)
    else:
        raise NotImplementedError

    dataset = data_config[args.dataset]["constructor"](root=args.root, split=split, transform=transform)
    if train and args.train_fraction < 1:
        n_to_retain = int(np.round(float(len(dataset)) * args.train_fraction))
        indices = range(n_to_retain)
        dataset = Subset(dataset, indices)
    return dataset



def get_loader(dataset, split, **kwargs):
    assert split in ('train', 'valid', 'test', "all")
    if split == 'train':
        train = True
    elif split in ('valid', 'test'):
        train = False
    else:
        raise NotImplementedError

    loader = DataLoader(dataset, shuffle=train, **kwargs)
    return loader
