import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os

#TODO: check the batchsize


def data_directory():
    '''

    :param data_dir:
    :return: train_dir, valid_dir, test_dir
    '''
    img_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'inputs\img'))
    # split_path =
    train_dir = '{0}\Train'.format(img_path)
    valid_dir = img_path + '\Val'
    return train_dir, valid_dir


def load_data(batchsize=64):
    '''

    :param data_dir: data directory path
    :return: train_loader, val_loader, test_loader in PyTorch Loader format
    '''
    train_dir, valid_dir = data_directory()

    # Define transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    #TODO: this is how you should get the mean and sd.
    # loader = DataLoader(train_set, batch_size=len(train_set), num_workers=1)
    # data[0].mean(), data[0].std()
    # data = next(iter(loader))
    # (tensor(0.2860), tensor(0.3530))
    #

    # testing_transforms = transforms.Compose([transforms.Resize(256),
    #                                          transforms.CenterCrop(224),
    #                                          transforms.ToTensor(),
    #                                          transforms.Normalize([0.485, 0.456, 0.406],
    #                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    # testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batchsize, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batchsize)
    # test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=batchsize)

    return train_loader, validate_loader #,test_loader
