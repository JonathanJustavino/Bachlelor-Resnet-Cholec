import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms, models
import matplotlib as plt
import numpy as np

import re
import os
import sys

from cholec80 import Cholec80


IMG_EXTENSIONS = ['.png']
path = '/media/data/ToolClassification/cholec80/frames'
folder = '1'
annotations_path = '/media/data/ToolClassification/cholec80/phase_annotations'


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validate': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# img_datasets = {x: datasets.ImageFolder(os.path.join(path, x), data_transforms[x]) for x in ['train', 'validate']}

training_folder = ['1', '2', '3']
validation_folder = ['4']


dataset = {x: Cholec80((os.path.join(path, x)), annotations_path, IMG_EXTENSIONS, data_transforms['train']) for x in training_folder}
for sub in validation_folder:
    dataset[sub] = Cholec80((os.path.join(path, sub)), annotations_path, IMG_EXTENSIONS, data_transforms['validate'])

print(dataset)

dataloaders = {x: torch.utils.data.DataLoader(dataset[x], batch_size=32, shuffle=True, num_workers=5) for x in training_folder + validation_folder }




# dataloaders = {x: torch.utils.data.DataLoader(training_datasets[x], batch_size=72, shuffle=True, num_workers=6) for x in ['train', 'validate']}
#
# data_sizes = {x: len(training_datasets[x]) for x in ['train', 'validate']}
# class_names = training_datasets['train'].classes
# print(class_names)


def img_show(inp, title=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp.clip(inp, 0, 1)
    plt.show(inp)
    if title:
        plt.title

inputs, classes = next(iter(dataloaders['1']))

out = torchvision.utils.make_grid(inputs)
img_show(out)
