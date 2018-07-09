import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.optim import lr_scheduler
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from bot import *

import os
import time
import copy
import sys
import datetime
import csv

from cholec80 import Cholec80


IMG_EXTENSIONS = ['.png']
path = '/media/data/ToolClassification/cholec80/frames'
annotations_path = '/media/data/ToolClassification/cholec80/phase_annotations'
result_path = '/media/data/ToolClassification/results/resnet18'
net_path = '/media/TCO/TCO-Studenten/justaviju/results/resnet18'

picture_size = (384, 216)


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(picture_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validate': transforms.Compose([       
        transforms.Resize(picture_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def write_epoch_predictions(path, preds, labels):
    with open(os.path.join(result_path, path), 'a', newline='') as results:
        writer = csv.writer(results, delimiter=',')
        results.write("Predictions: ")
        writer.writerow(preds)
        results.write("Labels:      ")
        writer.writerow(labels)


training_folder = ['4', '2', '3']
validation_folder = ['1']

training_phase = '3'
dataset_folders = training_folder + validation_folder


dataset = {x: Cholec80((os.path.join(path, x)), annotations_path, IMG_EXTENSIONS, data_transforms['train']) for x in training_folder}
for sub in validation_folder:
    dataset[sub] = Cholec80((os.path.join(path, sub)), annotations_path, IMG_EXTENSIONS, data_transforms['validate'])

loader_batch_size = 64

dataloaders = {x: torch.utils.data.DataLoader(dataset[x], batch_size=loader_batch_size, shuffle=False, num_workers=5) for x in dataset_folders }
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
    plt.pause(5)


def progress_out(current, total):
    fraction = float(current) / total
    if current > total:
        current = total
        fraction = 1.0
    sys.stdout.write("\r[{}/{}]{:.2f}%".format(current, total, (fraction * 100)))


def train(model, criterion, optimizer, scheduler, batch_size, learning_rate, validation_set, date, epochs=10):
    since = time.time()
    predictions = date + "_predictions.csv"
    predictions_path = os.path.join(result_path, predictions)
    print('Validation Set: ', validation_set)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs - 1))
        print("-" * 10)

        with open(os.path.join(result_path, date), 'a') as result_file:

            result_file.write("Epoch {}:\n\n\n".format(epoch))

            for set in dataset_folders:
                d_size = data_sizes[set]
                print('Set: ', set)
                if set != validation_set:
                	# maybe remove the scheduler (only necessary for last top percentages)
                    scheduler.step()
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
                        # maybe wrap them into variables?
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
                send_message("ResNet18 Acc: {:4f}, Loss: {} in epoch: {} in set: {}".format(epoch_acc, epoch_loss, epoch, set))
                result_file.write("Set: {} Loss: {:.4f} Acc: {:.4f}\n".format(set, epoch_loss, epoch_acc))

                if set == validation_set and epoch_acc > best_acc:
                    epoch_of_best_acc = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())  
                    try:
                        print('\nSaving ResNet18')
                        torch.save(model.state_dict(), os.path.join(net_path, "model"))
                        torch.save(optimizer.state_dict(), os.path.join(net_path, "optimizer"))
                        torch.save(scheduler.state_dict(), os.path.join(net_path, "scheduler"))
                    except Exception as e:
                        print("attempt to save the network failed")
                        send_message("Error {}".format(e))
                        send_message("Failed attempt to save the network.")

    with open(os.path.join(result_path, date), 'a') as result_file:
        result_file.write("ResNet18 Best val Acc: {:4f} in epoch: {} Learning rate: {}\n".format(best_acc, epoch_of_best_acc, learning_rate))
        result_file.write("Optimizer: {}".format(optimizer_conv))

    model.load_state_dict(best_model_wts)
    return model


model_conv = torchvision.models.resnet18(pretrained=True)

# train only the last block
for name, layer in model_conv._modules.items():
	if 'layer4' not in name and 'fc' not in name:
		for param in layer.parameters():
			param.requires_grad = False

# for name, layer in model_conv._modules.items():
# 	for param in layer.parameters():
# 		print(name)
# 		print(param.requires_grad)


model_conv.fc = nn.Linear(3072, 7)
print(model_conv)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.0005
validation_set = '1'

trainable_layers = list(model_conv.layer4.parameters()) + list(model_conv.fc.parameters())

# optim Adam
adam = True
if adam:
    optimizer_conv = optim.Adam(trainable_layers, lr=learning_rate)
# optim SGD
else:
    optimizer_conv = optim.SGD(trainable_layers, lr=learning_rate, momentum=0.9)


exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
print("Optimizer", optimizer_conv)
send_message("Training Started...(ResNet18)")

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

try:
    model_conv = train(model_conv, criterion, optimizer_conv, exp_lr_scheduler, loader_batch_size, learning_rate, validation_set, date, epochs=150)
    torch.save(model_conv.state_dict(), os.path.join(net_path, 'model'))
    send_message("Training Finished. (ResNet18)")
except Exception as e:
    print(e)
    print("ResNet18 crashed...")
    send_message("ResNet18 crashed...")



