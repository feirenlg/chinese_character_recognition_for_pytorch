import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils import data


class DataSet(data.Dataset):

    def __init__(self, config, train=True, test=False, transform=None):
        self.config = config
        self.images = list()
        self.labels = list()

        if not test:
            root = config.TRAIN_DATA_ROOT
        else:
            root = config.TEST_DATA_ROOT

        categories = os.listdir(root)
        for category in categories:
            images_category = os.listdir(os.path.join(root, category))
            images_category = [os.path.join(root, category, str(image)) for image in images_category]
            if train:
                images_category = images_category[: int(0.8 * len(images_category))]
            elif not train and not test:
                images_category = images_category[int(0.8 * len(images_category)):]
            self.images += images_category
            self.labels += [category for i in range(len(images_category))]

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((config.RE_SIZE, config.RE_SIZE)),
                transforms.Pad(config.PADDING_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[config.MEAN_R, config.MEAN_G, config.MEAN_B],
                                     std=[config.STD_R, config.STD_G, config.STD_B])
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):

        image = Image.open(self.images[index])
        label = int(self.labels[index][1:])
        data = self.transform(image)
        return data, label


    def __len__(self):
        return len(self.labels)
