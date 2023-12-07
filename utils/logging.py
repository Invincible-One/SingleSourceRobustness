import os
import sys
import csv


import numpy as np

import torch



class Logger:
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()



def log_args(args):
    max_len = max(len(arg_name) for arg_name in vars(args))
    line = '*' * (max_len + 2 + 20)

    print(line)

    print("Arguments & Hyperparameters".center(max_len + 2 + 20))

    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name.ljust(max_len)}: {arg_value}")

    print(line)
