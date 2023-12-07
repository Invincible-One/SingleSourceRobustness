import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset

from models import model_config



class ConfounderDataset(Dataset):
    _split_map = {
            'train': 0,
            'valid': 1,
            'test': 2,
            "all": None,
            }

    def __init__(self, root, split, transform, target_transform, conditional_name, target_attr, confounder_attr):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        #WARNING: conditional not properly handled
        self.conditional_name = conditional_name
        self.target_attr = target_attr
        if isinstance(confounder_attr, list):
            self.confounder_attr = confounder_attr
        else:
            self.confounder_attr = [confounder_attr]

        self.guidance_list = None
        self.transform2 = None

    def __getitem__(self, idx):
        target = self.target_arr[idx]
        g = self.group_arr[idx]

        X = Image.open(os.path.join(self._data_folder, self.filename[idx])).convert('RGB')

        if self.guidance_list:
            if self.guidance_list[idx] == 0 and self.transform is not None:
                X = self.transform(X)
            #Comment: 1 means hard samples
            elif self.guidance_list[idx] == 1 and self.transform2 is not None:
                X = self.transform2(X)
        else:
            if self.transform is not None:
                X = self.transform(X)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        return X, target, g, idx
            
    def __len__(self):
        return len(self.filename)

    def add_guidance(self, guidance_list, **kwargs):
        """
        Special method to add guidance to the dataset

        Arguments:
        guidance_list: Provide guidance to datapoints processing
        kwargs:
            transform: processing is transform
        """
        assert len(guidance_list) == len(self)
        self.guidance_list = guidance_list

        if "transform" in kwargs:
            self.transform2 = kwargs["transform"]




class CelebA(ConfounderDataset):
    """CelebA
    attributes:
        n_classes (int): Number of classes.
        n_groups (int): Number of groups.
        filename (array): Array of image file names.
        target_arr (array): Array of targets, as int.
        group_arr (array): Array of group names, as int.
    """
    _base_folder = "celebA/data"
    _metadata = ["list_attr_celeba.csv", "list_eval_partition.csv"]
    n_classes = 2
    
    def __init__(
            self,
            root,
            split,
            transform=None,
            target_transform=None,
            conditional_name=None,
            target_attr="Blond_Hair",
            confounder_attr="Male",
            ):
        super().__init__(root, split, transform, target_transform, conditional_name, target_attr, confounder_attr)
        self._data_folder = os.path.join(self.root, self._base_folder, "img_align_celeba")   #_data_folder is where data points are kept
        
        #split and select attrs, map values from {-1, 1} to {0, 1} and map multiple confounders to a single int
        split_df = pd.read_csv(os.path.join(self.root, self._base_folder, "list_eval_partition.csv")) 
        split_arr = split_df["partition"].to_numpy()
        assert self.split in ('train', 'valid', 'test', "all"), f"{self.split} is not valid!"
        split_ = self._split_map[self.split]
        mask = (split_arr == split_) if split_ is not None else np.ones_like(split_arr, dtype=bool)

        attr_df = pd.read_csv(os.path.join(self.root, self._base_folder, "list_attr_celeba.csv"))
        self.filename = attr_df.loc[mask, "image_id"].to_numpy()
        target_arr = attr_df.loc[mask, self.target_attr].to_numpy()
        confounder_arr = attr_df.loc[mask, self.confounder_attr].to_numpy()

        self.target_arr = ((target_arr + 1) // 2).astype(int)
        confounder_arr = ((confounder_arr + 1) // 2).astype(int)
        n_confounder = len(self.confounder_attr)
        confounder_arr = np.matmul(
                confounder_arr,
                np.power(2, np.arange(n_confounder)))

        #groups
        self.n_groups = self.n_classes * np.power(2, n_confounder)
        self.group_arr = self.target_arr * np.power(2, n_confounder) + confounder_arr



class CUB(ConfounderDataset):
    """Caltech-UCSD Birds
    attributes:
        n_classes (int): Number of classes.
        n_groups (int): Number of groups.
        filename (array): Array of image file names, w.r.t. _data_folder.
        target_arr (array): Array of targets, as int.
        group_arr (array): Array of group names, as int.
    """
    _base_folder = "cub/waterbird_complete95_forest2water2"
    _metadata = ["metadata.csv"]
    n_classes = 2

    def __init__(
            self,
            root,
            split,
            transform=None,
            target_transform=None,
            conditional_name=None,
            target_attr='y',
            confounder_attr="place",
            ):
        super().__init__(root, split, transform, target_transform, conditional_name, target_attr, confounder_attr)
        self._data_folder = os.path.join(self.root, self._base_folder)

        #split. CUB only has one confounder.
        assert len(self.confounder_attr) == 1
        self.n_groups = 4

        metadata_df = pd.read_csv(os.path.join(self.root, self._base_folder, "metadata.csv"))
        split_arr = metadata_df["split"].to_numpy()
        assert self.split in ('train', 'valid', 'test', "all"), f"{self.split} is not valid!"
        split_ = self._split_map[self.split]
        mask = (split_arr == split_) if split_ is not None else np.ones_like(split_arr, dtype=bool)

        self.filename = metadata_df.loc[mask, "img_filename"].to_numpy()
        self.target_arr = metadata_df.loc[mask, 'y'].to_numpy().astype(int)
        confounder_arr = metadata_df.loc[mask, "place"].to_numpy().astype(int)
        self.group_arr = self.target_arr * 2 + confounder_arr
