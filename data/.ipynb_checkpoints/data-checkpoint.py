import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset

from models import model_attributes
from data.celebA_dataset import CelebADataset 
from data.dro_dataset import DRODataset



confounder_settings = {
        "celebA": {
            "constructor": CelebADataset,
            },
        }



def prepare_confounder_data(args, train):
    if args.dataset == "celebA" and not os.path.exists(os.path.join(args.root_dir, "celebA/data/", "metadata.csv")):
        gen_celebA_metadata(root_dir=os.path.join(args.root_dir, "celebA/data/"), metadata_csv="metadata.csv")

    full_dataset = confounder_settings[args.dataset]["constructor"](
            root_dir=args.root_dir,
            augment_data=args.augment_data,
            model_type=args.model,)

    if train:
        splits = ['train', 'val']
    else:
        splits = ['test']
    subsets = full_dataset.get_splits(splits, train_frac=args.fraction)
    dro_subsets = [DRODataset(subsets[split], n_groups=full_dataset.n_groups, n_classes=full_dataset.n_classes) for split in splits]

    return dro_subsets



def gen_celebA_metadata(root_dir, metadata_csv, attr_csv="list_attr_celeba.csv", split_csv="list_eval_partition.csv"):
    attr_df = pd.read_csv(os.path.join(root_dir, attr_csv))
    split_df = pd.read_csv(os.path.join(root_dir, split_csv))
    metadata_df = pd.merge(attr_df, split_df[["image_id", "partition"]], on="image_id", how="left")
    
    metadata_df.to_csv(os.path.join(root_dir, metadata_csv), index=False)
    
    metadata_df = pd.read_csv(os.path.join(root_dir, metadata_csv))
    for col in attr_df.columns:
        assert all(metadata_df[col] == attr_df[col])
    assert all(metadata_df["image_id"] == split_df["image_id"])
    assert all(metadata_df["partition"] == split_df["partition"])



#def log_data(data, logger):
#    logger.write("Training Data...\n")
#    for group_idx in range(data["train_data"].n_groups):
#        logger.write(f'    {data["train_data"].group_str(group_idx)}: n = {data["train_data"].group_counts()[group_idx]:.0f}\n')
#    
#    logger.write("Validation Data...\n")
#    for group_idx in range(data["val_data"].n_groups):
#        logger.write(f'    {data["val_data"].group_str(group_idx)}: n = {data["val_data"].group_counts()[group_idx]:.0f}\n')
