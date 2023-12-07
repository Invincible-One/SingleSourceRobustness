import os
import sys

import numpy as np
import pandas as pd

import torch

join =  os.path.join

res_config = {
        "celebA": {
            "res_folder": "celebA/results",
            "base_folder": "celebA/data",
            "metadata": ["list_attr_celeba.csv", "list_eval_partition.csv"],
            "img_col": "image_id",
            "target_col": "Blond_Hair",
            "split_col": "partition",
            "train_split_map": 0,
            },
        "cub": {
            "res_folder": "cub/results",
            "base_folder": "cub/waterbird_complete95_forest2water2",
            "metadata": ["metadata.csv"],
            "img_col": "image_filename",
            "target_col": 'y',
            "split_col": 'split',
            "train_split_map": 0,
            },
        }


class ResCollector:
    def __init__(self, epoch, total_data_points, args):
        res_folder = join(
                args.root,
                res_config[args.dataset]["res_folder"],
                )
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        self.fpath = join(
                res_folder,
                args.exp_name,
                )
        if not os.path.exists(self.fpath):
            metadata_path = [
                    join(
                        args.root,
                        res_config[args.dataset]["base_folder"],
                        res_config[args.dataset]["metadata"][i],
                        )
                    for i in range(len(res_config[args.dataset]["metadata"]))
                    ]
            if len(metadata_path) == 1:
                metadata_df = pd.read_csv(metadata_path[0])
                fdf = metadata_df[[
                    res_config[args.dataset]["img_col"],
                    res_config[args.dataset]["target_col"],
                    res_config[args.dataset]["split_col"]
                    ]]
            elif len(metadata_path) == 2:
                attr_df = pd.read_csv(metadata_path[0])
                split_df = pd.read_csv(metadata_path[1])
                if not all(attr_df[res_config[args.dataset]["img_col"]] == split_df[res_config[args.dataset]["img_col"]]):
                    raise Exception("Columns in two files do not match!")
                fdf = pd.concat(
                        [
                            attr_df[[
                                res_config[args.dataset]["img_col"],
                                res_config[args.dataset]["target_col"]]],
                            split_df[
                                res_config[args.dataset]["split_col"]]
                            ],
                        axis=1)        
            else:
                raise NotImplementedError
            fdf = fdf[fdf[res_config[args.dataset]["split_col"]] == 0][[res_config[args.dataset]["img_col"], res_config[args.dataset]["target_col"]]]
            
            if args.dataset == "celebA":
                fdf[res_config[args.dataset]["target_col"]] = (fdf[res_config[args.dataset]["target_col"]] + 1) // 2
            
            fdf.to_csv(self.fpath, index=False)

        self.res_name = f"e{epoch:02d}"
        self.all_predictions = torch.zeros(total_data_points, dtype=torch.int64, device='cuda')
        self.all_data_idx = torch.zeros(total_data_points, dtype=torch.int64, device='cuda')
        self.curr_idx = 0

    def update_res(self, output, y, data_idx):
        predicted = output.argmax(dim=1)
        #predicted = (output.argmax(dim=1) == y)   It depends on whether you want to save the predictions or results
        batch_size = len(predicted)
        self.all_predictions[self.curr_idx : self.curr_idx + batch_size] = predicted
        self.all_data_idx[self.curr_idx : self.curr_idx + batch_size] = data_idx
        self.curr_idx += batch_size

    def save_res(self):
        predicted_arr = self.all_predictions.cpu().numpy()
        data_idx_arr = self.all_data_idx.cpu().numpy()
        fdf = pd.read_csv(self.fpath)
        fdf.loc[data_idx_arr, self.res_name] = predicted_arr
        fdf[self.res_name] = fdf[self.res_name].astype(int)
        fdf.to_csv(self.fpath, index=False)
