from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class CustomDataset(Dataset):

    def __init__(self, data_file, root_dir, transform=None):
        """
        Arguments:
            data_file (string): Path to the file containing the name of the images to process
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_names = []

        with open(data_file, "r") as f:
            for line in f.readlines():
                self.image_names.append(str(line.strip())+".png")
        self.root_dir = root_dir
        if transform:
            self.transform = transform
        else: 
            self.transform = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.image_names[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image