# importing packages
from PIL import Image
import numpy as np
import math



# PatchMasking transformations
class PatchMasking:
    '''
    customizd transform
    description
        randomly masking out patches of image with given values
    __init__
        grid_size (tuple or int): size of the grid
        ratio (float): masking ratio
        auto_fill (bool): whether to do autofill
        value (tuple with length 3 or int): fill-in value
    __call__
        img (torch.tensor): single image with shape (num_channel, height, width)
    '''
    def __init__(self, grid_size, ratio, auto_fill=False, value=0):
        if not isinstance(grid_size, tuple):
            grid_size = (grid_size, ) * 2
        if not isinstance(value, tuple):
            value = (value, ) * 3
        self.grid_size = grid_size
        self.ratio = ratio
        self.value = value
        self.auto_fill = auto_fill
        
        self.num_patch = self.grid_size[0] * self.grid_size[1]
        self.num_mask = int(self.ratio * self.num_patch)
    def __call__(self, img):
        mask_indicator = np.hstack([
            np.zeros(self.num_patch - self.num_mask),
            np.ones(self.num_mask),
            ])
        np.random.shuffle(mask_indicator)
        
        img_height, img_width = img.shape[1], img.shape[2]
        patch_height, patch_width = img_height // self.grid_size[0], img_width // self.grid_size[1]
        for (idx, mask_it) in enumerate(mask_indicator):
            if mask_it:
                row_idx, col_idx  = idx // self.grid_size[1], idx % self.grid_size[1]
                row_slice = (row_idx * patch_height, None) if row_idx == (self.grid_size[0] - 1) else (row_idx * patch_height, row_idx * patch_height + patch_height)
                col_slice = (col_idx * patch_width, None) if col_idx == (self.grid_size[1] - 1) else (col_idx * patch_width, col_idx * patch_width + patch_width)
                if self.auto_fill:
                    true_value = img[:, row_slice[0]:row_slice[1], col_slice[0]:col_slice[1]].mean(dim=(1, 2))
                else:
                    true_value = self.value
                if img.shape[0] == 3:
                    for channel_idx in range(3):
                        img[channel_idx, row_slice[0]:row_slice[1], col_slice[0]:col_slice[1]] = true_value[channel_idx]
                else:
                    img[0, row_slice[0]:row_slice[1], col_slice[0]:col_slice[1]] = true_value[0]
        return img
