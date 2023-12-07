import os
import sys
import time
import argparse
from PIL import Image
import csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Subset

from train import run_epoch
from models import get_model, model_config
from loss import hinge_loss, LossComputer
from data import get_data, get_loader, get_group_repr
from utils.helper import progress_bar
from utils.logging import Logger, log_args
from result_collect import ResCollector



def get_args():
    parser = argparse.ArgumentParser()

    ##basic settings
    parser.add_argument("--method", choices={"ERM", "JTT", "AUG", "AUG_TT", "AUG_JTT"}, required=True)
    parser.add_argument("--dataset", choices={"celebA", "cub"}, required=True)
    parser.add_argument("--model", choices=model_config.keys(), default="resnet50")

    parser.add_argument("--root", type=str, default="/scratch/ym2380/data/")
    parser.add_argument("--n_workers", type=int, default=6)
    parser.add_argument("--train_fraction", type=float, default=1.0)

    ##result collecting
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--res_epochs", type=int, nargs='*')
    parser.add_argument("--save_model_epochs", type=int, nargs='*')
    ###JTT conditional epoch
    parser.add_argument("--conditional_epoch", type=int, default=None)

    ##masking settings
    parser.add_argument("--mask_rate1", type=float, default=0.8)
    parser.add_argument("--mask_rate2", type=float, default=0.3)
    parser.add_argument("--test_mask_rate", type=float, default=0.3)
    parser.add_argument("--grid_height", type=int, default=7)
    parser.add_argument("--grid_width", type=int, default=7)
    parser.add_argument("--fill_r", type=float, default=1)
    parser.add_argument("--fill_g", type=float, default=1)
    parser.add_argument("--fill_b", type=float, default=1)
    parser.add_argument("--auto_fill", action="store_true")

    ##hyperparameters
    parser.add_argument("--up_weight", type=int, default=None)
    parser.add_argument("--hinge", action="store_true")

    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", choices={"SGD", "Adam"}, required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0)

    args = parser.parse_args()
    return args



def check_args(args):
    if args.save_model_epochs:
        assert max(args.save_model_epochs) <= args.n_epochs
    if args.method == 'ERM':
        if args.res_epochs:
            assert args.exp_name is not None
            assert max(args.res_epochs) <= args.n_epochs
    elif args.method == "JTT":
        assert args.exp_name is not None
        assert args.conditional_epoch is not None
        assert args.up_weight is not None
        assert not args.res_epochs
        assert args.train_fraction == 1
    elif args.method == "AUG":
        pass
    else:
        raise NotImplementedError



def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if __name__ == "__main__":
    #Settings
    args = get_args()
    check_args(args)
    log_args(args)

    device = get_device()

    ###save model settings
    saved_model_root = "/scratch/ym2380/saved_models/"   #WARNING: machine specific
    save_folder = os.path.join(saved_model_root, args.model, args.dataset, args.exp_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    #Data Section
    train_set = get_data(args, 'train')
    valid_set = get_data(args, 'valid')

    loader_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": args.n_workers,
            "pin_memory": True,
            }
    train_loader = get_loader(dataset=train_set, train=True, args=args, **loader_kwargs)
    valid_loader = get_loader(dataset=valid_set, train=False, args=args, **loader_kwargs)

    n_classes = train_set.n_classes
    n_groups = train_set.n_groups

    #Model Section
    network = get_model(model=args.model, n_classes=n_classes).to(device)

    #Loss & Optimization Section
    get_group_repr_fn = lambda g: get_group_repr(
            g,
            n_classes=n_classes,
            n_groups=n_groups,
            target_attr=train_set.target_attr,
            confounder_attr=train_set.confounder_attr,
            )
    if args.hinge:
        assert args.dataset in ["celebA", "cub"]
        criterion = hinge_loss
    else:
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
    loss_computer = LossComputer(criterion, n_groups, get_group_repr_fn=get_group_repr_fn)

    if args.optimizer == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.scheduler_gamma)
    else:
        scheduler = None

    #Run Epoch Section
    count = 0
    for _ in range(args.n_epochs):
        beg = time.time()
        count += 1

        if args.method == 'ERM' and count in args.res_epochs:
            collect_res = True
        else:
            collect_res = False

        train_log = run_epoch(
                dataloader=train_loader,
                network=network,
                loss_computer=loss_computer,
                optimizer=optimizer,
                train=True,
                collect_res=collect_res,
                epoch=count,
                total_data_points=len(train_set),
                args=args,
                )

        if args.save_model_epochs is not None and count in args.save_model_epochs:
            model_path = os.path.join(save_folder, f"e{count}.pth")
            torch.save(network.state_dict(), model_path)

        if scheduler is not None:
            scheduler.step()

        valid_log = run_epoch(
                dataloader=valid_loader,
                network=network,
                loss_computer=loss_computer,
                optimizer=optimizer,
                train=False,
                )
        
        runtime = time.time() - beg
        print(f"Epoch [{count:02d}/{args.n_epochs}] {progress_bar(count, args.n_epochs)}, Time {runtime:.0f}, {train_log}, {valid_log}.")
        sys.stdout.flush()


