import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

import os
import time
import copy
import sys

from cholec80 import Cholec80

plt.ion()

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

training_folder = ['1', '2', '3']
validation_folder = ['4']

training_folder = ['3']
validation_folder = ['4']
training_phase = '3'
dataset_folders = training_folder + validation_folder


dataset = {x: Cholec80((os.path.join(path, x)), annotations_path, IMG_EXTENSIONS, data_transforms['train']) for x in training_folder}
for sub in validation_folder:
    dataset[sub] = Cholec80((os.path.join(path, sub)), annotations_path, IMG_EXTENSIONS, data_transforms['validate'])

print(dataset)

dataloaders = {x: torch.utils.data.DataLoader(dataset[x], batch_size=72, shuffle=True, num_workers=5) for x in dataset_folders }
data_sizes = {x: len(dataset[x]) for x in dataset_folders}
class_names = dataset[training_phase].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def img_show(inp, title=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    # inp.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title
    plt.pause(1)


def progress_out(current, total):
    fraction = float(current) / total
    sys.stdout.write("\r[{}/{}]{:.2f}%".format(current, total, (fraction * 100)))

# inputs, classes = next(iter(dataloaders['3']))
#
# out = torchvision.utils.make_grid(inputs)
# img_show(out)


def train(model, criterion, optimizer, scheduler, epochs=10):
    since = time.time()
    batch_size = 72

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('HELLO!!!')
        print("Epoch {}/{}".format(epoch, epochs - 1))
        print("-" * 10)

        for phase in dataset_folders:
            d_size = data_sizes[phase]
            print('Phase: ', phase)
            if phase != '4':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            num_run = 0
            for inputs, labels in dataloaders[phase]:
                #progress output
                num_run += 1
                current = num_run * batch_size
                if current > 7000:
                    break
                progress_out(current, d_size)

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase != '4'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase != '4':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == '4' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

        time_elapsed = time.time() - since
        print("Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        print("Best val Acc: {:4f}".format(best_acc))

        model.load_state_dict(best_model_wts)
        return model


model_conv = torchvision.models.resnet18(pretrained=True)

# for param in model_conv.parameters():
#     param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 7)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)


exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train(model_conv, criterion, optimizer_conv, exp_lr_scheduler, epochs=5)
