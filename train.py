import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torch import Tensor
from torch.optim import lr_scheduler
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms, models

import numpy as np
import random
import os
import time
import copy
import sys
import datetime
import csv

#from cholec80 import Cholec80
from cholec80 import *
from bot import *


IMG_EXTENSIONS = ['.png']
path = '/media/data/ToolClassification/cholec80/frames'
annotations_path = '/media/data/ToolClassification/cholec80/phase_annotations'

parent_result_folder = '/media/data/ToolClassification/results'
parent_network_folder = '/media/TCO/TCO-Studenten/justaviju/results'


picture_size = (384, 216)
data_transforms = {
    'default_transformation': transforms.Compose([
        transforms.Resize(picture_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def get_result_path(net_type, parent_folder=parent_result_folder):
    return os.path.join(parent_folder, net_type)


def get_net_path(net_type, parent_folder=parent_network_folder):
    return os.path.join(parent_folder, net_type)


def write_epoch_predictions(path, preds, labels):
    '''Write model predictions and ground truth to a csv file'''
    with open(os.path.join(path), 'a', newline='') as results:
        writer = csv.writer(results, delimiter=',')
        results.write("Predictions: ")
        writer.writerow(preds)
        results.write("Labels:      ")
        writer.writerow(labels)


def pick_validation_folder(epoch, data_folders, data_folders_length):
    return data_folders.pop(epoch % data_folders_length)


def generate_dataset(data_folders, transformation='default_transformation'):
    print(data_folders)
    dataset = {x: Cholec80((os.path.join(path, x)), annotations_path,
                           IMG_EXTENSIONS,
                           data_transforms[transformation])
               for x in data_folders}
    return dataset


def generate_dataloader(dataset, data_folders, batch_size, shuffling):
    dataloaders = {x: torch.utils.data.DataLoader(dataset[x],
                                                  batch_size=batch_size,
                                                  shuffle=shuffling, num_workers=5)
                   for x in data_folders}
    return dataloaders


def get_dataset_sizes(dataset, data_folders):
    return {x: len(dataset[x]) for x in data_folders}


def get_class_names(dataset):
    return dataset[0].classes


def set_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def progress_out(current, total):
    fraction = float(current) / total
    if current > total:
        current = total
        fraction = 1.0
    sys.stdout.write("\r[{}/{}]{:.2f}%".format(
                     current, total, (fraction * 100)))


def train(model, criterion, optimizer, scheduler, batch_size, learning_rate, data_sizes, dataloaders,
          dataset_folders, date, net_type, device, epochs=10):
    result_path = get_result_path(net_type.lower())
    net_path = get_net_path(net_type.lower())
    print("\nPath: ", net_path)
    predictions_path = os.path.join(result_path, "{}_predictions.csv".format(date))
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    folder_length = len(dataset_folders)

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs - 1))
        print("-" * 10)
        validation_set = pick_validation_folder(epoch, dataset_folders, folder_length)
        data_folders.append(validation_set)


        with open(os.path.join(result_path, date), 'a') as result_file:
            result_file.write("Epoch {}:\n\n\n".format(epoch))
            print(dataset_folders)
            for set in dataset_folders:
                d_size = data_sizes[set]
                print('Set: ', set)
                if set != validation_set:
                    # maybe remove the scheduler
                    # (only necessary for last top percentages)
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

                        loss = criterion(outputs, labels)

                        if set != validation_set:
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    if set == validation_set and epoch % 5 == 0:
                        write_epoch_predictions(predictions_path,
                                                preds.cpu().numpy(),
                                                labels.data.cpu().numpy())

                epoch_loss = running_loss / data_sizes[set]
                epoch_acc = running_corrects.double() / data_sizes[set]

                print("{} Loss: {:.4f} Acc: {:.4f}".format(
                      set, epoch_loss, epoch_acc)
                      )
                result_file.write("Set: {} Loss: {:.4f} Acc: {:.4f}\n".format(
                                  set, epoch_loss, epoch_acc))
                try:
                    send_message("{} Acc: {:4f}, Loss: {} in epoch: {} \
                                 in set: {}".format(
                                 net_type, epoch_acc, epoch_loss, epoch, set))
                except:
                    print("Sending results was not successful")

                if set == validation_set and epoch_acc > best_acc:
                    epoch_of_best_acc = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    try:
                        print('\nSaving', net_type)
                        net_type_lower = net_type.lower()
                        torch.save(model.state_dict(),
                                   os.path.join(net_path, "model_{}".format(net_type_lower)))
                        torch.save(optimizer.state_dict(),
                                   os.path.join(net_path, "optimizer_{}".format(net_type_lower)))
                        torch.save(scheduler.state_dict(),
                                   os.path.join(net_path, "scheduler_{}".format(net_type_lower)))
                    except Exception as e:
                        print("attempt to save the network failed")
                        send_message("Error {}".format(e))
                        send_message("Failed attempt to save the network.")

    with open(os.path.join(result_path, date), 'a') as result_file:
        result_file.write("{} Best val Acc: {:4f} in epoch: {} \
        Learning rate: {}\n".format(net_type, best_acc,
                                    epoch_of_best_acc, learning_rate))
        result_file.write("Optimizer: {}".format(optimizer))

    model.load_state_dict(best_model_wts)
    return model
