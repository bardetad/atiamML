# To manage Dataset (especially batch structure) on .npz data

import torch
from torch.utils.data import Dataset

import numpy as np


class NPZ_Dataset(Dataset):

    def __init__(self, npz_file, root_dir, dataName='Spectrums', labelName='labels', transform=None):


        self.dataset_name = npz_file
        self.root_dir = root_dir
        self.path = self.root_dir + self.dataset_name

        npz_dict = np.load(self.path)

        # the data
        self.imgs_stack = npz_dict[dataName]
        # the infos on data
        self.labels_stack = npz_dict[labelName]

    # to support the indexing such that dataset[i] can be used to get ith
    # sample
    def __getitem__(self, idx):
        image = self.imgs_stack[:, idx]
        label = str(self.labels_stack[idx])
        singleData = {'image': image, 'label': label}
        return singleData

    # returns the size of the dataset
    def __len__(self):
        return len(self.labels_stack)
