import torch.utils.data as data
import os.path
import numpy as np


class FeatureDataset(data.Dataset):

    def __init__(self, filename, root_path, transform=None):

        fname = os.path.join(root_path, filename)
        self.data = np.load(fname)
        self.transform = transform

    def __len__(self):

        return self.data.shape[0]

    def __getitem__(self, index):

        return self.data[index]
