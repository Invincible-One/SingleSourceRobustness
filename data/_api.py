import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from models import model_config
from result_collect import res_config
from data.datasets import CelebA
from data.datasets import CUB
from utils.data import UpsampleSampler
from data.transforms import PatchMasking



data_config = {
        "celebA": {
            "constructor": CelebA,
            "orig_shape": (178, 218),   #shape is (width, height)
            },
        "cub": {
            "constructor": CUB,
            },
        }



def _get_celebA_transforms_list(target_resolution, train):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)

    if not train:
        transforms_list = [
                transforms.CenterCrop(orig_min_dim),
                transforms.Resize(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
                ]
    else:
        transforms_list = [
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
                ]

    return transforms_list

def _get_cub_transforms_list(target_resolution, train):
    scale = 256.0 / 224.0

    if not train:
        transforms_list = [
                transforms.Resize((
                    int(target_resolution[0] * scale),
                    int(target_resolution[1] * scale),
                    )),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
                ]
    else:
        transforms_list = [
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
                ]
    
    return transforms_list



def revise_transform(args, transforms_list, train):
    if train:
        if args.method in {"AUG", "AUG_TT", "AUG_JTT"}:
            transforms_list.append(
                    PatchMasking(
                        grid_size=(args.grid_height, args.grid_width),
                        ratio=args.mask_rate1,
                        auto_fill=args.auto_fill,
                        value=(args.fill_r, args.fill_g, args.fill_b),
                        )
                    )
            if args.method in {"AUG_TT", "AUG_JTT"}:
                transforms_list2 = list(transforms_list)
                transforms_list2.append(
                        PatchMasking(
                            grid_size=(args.grid_height, args.grid_width),
                            ratio=args.mask_rate2,
                            auto_fill=args.auto_fill,
                            value=(args.fill_r, args.fill_g, args.fill_b),
                            )
                        )
                return transforms.Compose(transforms_list), transforms.Compose(transforms_list2)
    
    else:
        if args.method in {"AUG", "AUG_TT", "AUG_JTT"}:
            transforms_list.append(
                    PatchMasking(
                        grid_size=(args.grid_height, args.grid_width),
                        ratio=args.test_mask_rate,
                        auto_fill=args.auto_fill,
                        value=(args.fill_r, args.fill_g, args.fill_b),
                        )
                    )

    return transforms.Compose(transforms_list), None



def _get_guidance_series(args):
    """
    return:
    guidance (pd.Series): containing boolean values, True if the result is different from the target, False, otherwise
    """
    fpath = os.path.join(
            args.root,
            res_config[args.dataset]["res_folder"],
            args.exp_name,
            )
    fdf = pd.read_csv(fpath)
    
    res_name = f"e{args.conditional_epoch:02d}"
    target_name = res_config[args.dataset]["target_col"]
    mask = fdf[res_name] != fdf[target_name]
    
    return mask



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
        transforms_list = _get_celebA_transforms_list(target_resolution=target_resolution, train=train)
    elif args.dataset == "cub":
        transforms_list = _get_cub_transforms_list(target_resolution=target_resolution, train=train)
    else:
        raise NotImplementedError

    transform, transform2 = revise_transform(args, transforms_list, train)
    dataset = data_config[args.dataset]["constructor"](root=args.root, split=split, transform=transform)
    
    if transform2 is not None:
        mask = _get_guidance_series(args)
        guidance_list = mask.astype(int).to_numpy()
        dataset.add_guidance(guidance_list, transform=transform2)

    if train and args.train_fraction < 1:
        n_to_retain = int(np.round(float(len(dataset)) * args.train_fraction))
        indices = range(n_to_retain)
        dataset = Subset(dataset, indices)
    return dataset



def get_loader(dataset, train, args, **kwargs):
    if train:
        if args.method == "JTT":
            mask = _get_guidance_series(args)
            upsample_indices = np.where(mask)[0]

            upsample_sampler = UpsampleSampler(
                    upsample_indices = upsample_indices,
                    n_samples=len(dataset),
                    up_weight=args.up_weight,
                    )

            loader = DataLoader(dataset, sampler=upsample_sampler, **kwargs)
        else:
            loader = DataLoader(dataset, shuffle=True, **kwargs)
    else:
        loader = DataLoader(dataset, shuffle=False, **kwargs)
    return loader



def get_group_repr(group, n_classes, n_groups, target_attr, confounder_attr):
    n_confounder = n_groups // n_classes

    target = group // n_confounder
    c = group % n_confounder

    group_repr = f"{target_attr}{int(target)}"
    c_bin_str = format(int(c), f"0{n_confounder}b")[::-1]
    for i, attr in enumerate(confounder_attr):
        group_repr += f"{attr}{c_bin_str[i]}"
    return group_repr
    
