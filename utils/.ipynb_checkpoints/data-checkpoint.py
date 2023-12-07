import numpy as np

import torch
from torch.utils.data.sampler import Sampler



class UpsampleSampler(Sampler):
    def __init__(self, upsample_indices, n_samples, up_weight):
        self.upsample_indices = np.array(upsample_indices)
        self.n_samples = n_samples
        self.up_weight = up_weight

        self.upsampled_indices = np.concatenate([
            np.nrange(self.n_samples),
            np.repeat(self.upsample_indices, self.up_wegiht - 1)
            ])

    def __iter__(self):
        return iter(np.random.permutation(self.upsampled_indices))

    def __len__(self):
        return len(self.upsampled_indices)



#WARNING: DebugSampler is for debugging purposes
class DebugSampler(Sampler):
    def __init__(self, upsample_indices, n_samples, up_weight):
        self.upsample_indices = np.array(upsample_indices)
        self.n_samples = n_samples
        self.up_weight = up_weight

        self.upsampled_indices = np.repeat(self.upsample_indices, self.up_wegiht)

    def __iter__(self):
        return iter(np.random.permutation(self.upsampled_indices))

    def __len__(self):
        return len(self.upsampled_indices)

