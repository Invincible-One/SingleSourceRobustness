import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from models import model_attributes
from data.utils import Subset



class ConfounderDataset(Dataset):
    def __init__(self, root_dir, augment_data, model_type, target_name, confounder_names, metadata_csv):
        # WARNING: not a good base class
        raise NotImplementedError

    def __getitem__(self, idx):
        y = self.label_array[idx]
        g = self.group_array[idx]

        img_filename = os.path.join(self.data_dir, self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')

        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            img = self.train_transform(img)
        elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']]
                and self.eval_transform):
            img = self.eval_transform(img)

        return img, y, g, idx

    def __len__(self):
        return len(self.filename_array)

    def get_group_array(self):
        return self.group_array

    def get_label_array(self):
        return self.label_array

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ('train', 'val', 'test'), f"{split} is not valid."
            
            mask = self.split_array == self.split_dict[split]
            n_split = np.sum(mask)
            indices = np.where(mask)[0]

            if split == 'train' and train_frac < 1:
                n_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:n_to_retain])

            subsets[split] = Subset(self, indices)

        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups / self.n_classes)
        c = group_idx % (self.n_groups // self.n_classes)
        
        group_name = f"{self.target_name} = {int(y)}"
        bin_str = format(int(c), f"0{self.n_confounders}b")[::-1]
        
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f", {attr_name} = {bin_str[attr_idx]}"
        
        return group_name
