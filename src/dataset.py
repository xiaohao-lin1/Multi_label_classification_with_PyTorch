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


def load_data(data_dir = 'flowers', batchsize=64):
    '''

    :param data_dir: data directory path
    :return: train_loader, val_loader, test_loader in PyTorch Loader format
    '''


    # train_dir = data_dir + '/train'
    # valid_dir = data_dir + '/valid'
    # test_dir = data_dir + '/test'

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

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=32)

    return train_loader, validate_loader, test_loader