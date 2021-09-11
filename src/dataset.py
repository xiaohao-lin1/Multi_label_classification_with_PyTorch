import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


data_dir = ''


def load_data(data_dir):
    '''

    :param data_dir: data directory path
    :return: train_loader, val_loader, test_loader in PyTorch Loader format
    '''

    train_data = torch.nn.data(
        root=''
        transforms='',
        train=True
    )
    val_data =
    test_data =

    transforms =

    train_loader = torch.utils.data.DataLoader()
    return train_loader, val_loader, test_loader