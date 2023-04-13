from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class YCBDataset(Dataset):
    """Custom dataset containing YCB data"""

    def __init__(self, data_file, root_dir):
        """
        Arguments:
            data_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.img_names = []
        with open(data_file, "r") as f:
            for line in f.readlines():
                self.img_names.append(str(line).strip()+"-color.png")
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor()]) 

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.img_names[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image
    
class ARMDataset(Dataset):
    """Custom dataset containing ARM data"""

    def __init__(self, data_file, root_dir):
        """
        Arguments:
            data_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.img_names = []
        with open(data_file, "r") as f:
            for line in f.readlines():
                self.img_names.append(str(line).strip()+"-color.png")
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor()]) 

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.img_names[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image