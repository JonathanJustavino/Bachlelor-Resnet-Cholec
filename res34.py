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
import datetime

from cholec80 import Cholec80

# plt.ion()

IMG_EXTENSIONS = ['.png']
path = '/media/data/ToolClassification/cholec80/frames'
annotations_path = '/media/data/ToolClassification/cholec80/phase_annotations'
result_path = '/media/data/ToolClassification/results/resnet34'


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((480,270)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validate': transforms.Compose([       
        transforms.Resize((480,270)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def write_epoch_predictions(path, preds, labels):
    with open(os.path.join(result_path, path), 'a') as results:
        results.write("\nPredictions: ")
        results.write(str(preds))
        results.write("\nLabels:      ")
        results.write(str(labels))


training_folder = ['4', '2', '3']
validation_folder = ['1']

training_phase = '3'
dataset_folders = training_folder + validation_folder

dataset = {x: Cholec80((os.path.join(path, x)), annotations_path, IMG_EXTENSIONS, data_transforms['train']) for x in training_folder}
for sub in validation_folder:
    dataset[sub] = Cholec80((os.path.join(path, sub)), annotations_path, IMG_EXTENSIONS, data_transforms['validate'])

loader_batch_size = 72

dataloaders = {x: torch.utils.data.DataLoader(dataset[x], batch_size=loader_batch_size, shuffle=True, num_workers=5) for x in dataset_folders }
data_sizes = {x: len(dataset[x]) for x in dataset_folders}
class_names = dataset[training_phase].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

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
    if current > total:
        current = total
    sys.stdout.write("\r[{}/{}]{:.2f}%".format(current, total, (fraction * 100)))


def train(model, criterion, optimizer, scheduler, batch_size, learning_rate, validation_set, epochs=10):
    since = time.time()
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    predictions = date + "predictions.txt"
    predictions_path = os.path.join(result_path, predictions)
    print('Validation Set: ', validation_set)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs - 1))
        print("-" * 10)

        with open(os.path.join(result_path, date), 'a') as result_file:

            result_file.write("Epoch {}:\n".format(epoch))

            for set in dataset_folders:
                d_size = data_sizes[set]
                print('Set: ', set)
                if set != validation_set:
                	# maybe remove the scheduler (only necessary for last top percentages)
                    # scheduler.step()
                    model.train()
                else:
                    print('Validation Set', validation_set)
                    model.eval()
                    with open(predictions_path, 'a') as file:
                        file.write("\nEpoch: {}\n".format(epoch))

                running_loss = 0.0
                running_corrects = 0

                num_run = 0
                for inputs, labels in dataloaders[set]:
                    # progress output
                    num_run += 1
                    current = num_run * batch_size
                    progress_out(current, d_size)

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(set != validation_set):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if set != validation_set:
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    if set == validation_set and epoch % 5 == 0:
                        write_epoch_predictions(predictions_path, preds.cpu().numpy(), labels.data.cpu().numpy())

                epoch_loss = running_loss / data_sizes[set]
                epoch_acc = running_corrects.double() / data_sizes[set]

                print("{} Loss: {:.4f} Acc: {:.4f}".format(set, epoch_loss, epoch_acc))

                if set == validation_set and epoch_acc > best_acc:
                    epoch_of_best_acc = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                result_file.write("Set: {} Loss: {:.4f} Acc: {:.4f}\n".format(set, epoch_loss, epoch_acc))

    with open(os.path.join(result_path, date), 'a') as result_file:
        result_file.write("ResNet34 Best val Acc: {:4f} in epoch: {} Learning rate: {}\n".format(best_acc, epoch_of_best_acc, learning_rate))

    model.load_state_dict(best_model_wts)
    return model


model_conv = torchvision.models.resnet34(pretrained=True)

# train only the last block
for name, layer in model_conv._modules.items():
	if name != 'layer4':
		for param in layer.parameters():
			param.requires_grad = False


num_ftrs = model_conv.fc.in_features
num_ftrs = num_ftrs * 27 # in order to reach the 13.824

model_conv.fc = nn.Linear(num_ftrs, 7)
model_conv = model_conv.to(device)

print(model_conv.layer4)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
validation_set = '1'

# optim Adam

optimizer_conv = optim.SGD(model_conv.layer4.parameters(), lr=learning_rate, momentum=0.9)

# maybe later on relevant
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train(model_conv, criterion, optimizer_conv, exp_lr_scheduler, loader_batch_size, learning_rate, validation_set, epochs=15)
