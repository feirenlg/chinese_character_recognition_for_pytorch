import torch.nn as nn
import torch
import time
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from config import conf


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64 * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64 * 2, 64 * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64 * 4, 64 * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64 * 8, 64 * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 3755)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 512 * 2 * 2)
        x = self.classifier(x)
        return x

