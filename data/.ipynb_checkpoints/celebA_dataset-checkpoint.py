import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset

from models import model_attributes
from data.confounder_dataset import ConfounderDataset



class CelebADataset(ConfounderDataset):
    def __init__(self, root_dir, augment_data, model_type, target_name="Blond_Hair", confounder_names=["Male",], metadata_csv="metadata.csv"):
        self.root_dir = os.path.join(root_dir, "celebA")
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data

        self.data_dir = os.path.join(self.root_dir, 'data', "img_align_celeba")
        self.n_classes = 2
        self.split_dict = {
                'train': 0,
                'val': 1,
                'test': 2,
                }

        self.attr_df = pd.read_csv(os.path.join(self.root_dir, 'data', metadata_csv))
        self.filename_array = self.attr_df["image_id"].values
        self.split_array = self.attr_df["partition"].values

        self.attr_df = self.attr_df.drop(labels=["image_id", "partition"], axis="columns")
        self.attr_names = self.attr_df.columns.copy()
        self.attr_df = self.attr_df.values # self.attr_df is an array now
        self.attr_df[self.attr_df == -1] = 0

        target_idx = self.get_attr_idx(self.target_name)
        self.label_array = self.attr_df[:, target_idx]

        confounder_indices = [self.get_attr_idx(a) for a in self.confounder_names]
        self.n_confounders = len(confounder_indices)
        confounders = self.attr_df[:, confounder_indices]
        self.confounder_array = np.matmul(
                confounders.astype(np.int64),
                np.power(2, np.arange(self.n_confounders)))

        self.n_groups = self.n_classes * pow(2, self.n_confounders)
        self.group_array = (self.label_array * (self.n_groups / self.n_classes) + self.confounder_array).astype(np.int64)

        self.train_transform = get_celebA_transform(model_type, train=True, augment_data=augment_data)
        self.eval_transform = get_celebA_transform(model_type, train=False, augment_data=augment_data)

    def get_attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)



def get_celebA_transform(model_type, train, augment_data):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)

    if model_attributes[model_type]["target_resolution"] is not None:
        target_resolution = model_attributes[model_type]["target_resolution"]
    else:
        target_resolution = (orig_w, orig_h)

    if (not train) or (not augment_data):
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
