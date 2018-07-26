import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.optim as optim
from torchvision import models
from train import *
import numpy as np


net_path = '/media/TCO/TCO-Studenten/justaviju/results/resnet18/model'
parent_result_folder = '/media/data/ToolClassification/results'
parent_network_folder = '/media/TCO/TCO-Studenten/justaviju/results'

resnet = None
net = None

class LSTM(nn.Module):
    def __init__(self, num_classes=7, hidden_size=1):
        super(LSTM, self).__init__()
        resnet = models.resnet18()
        resnet.fc = nn.Linear(3072, 7)
        resnet.load_state_dict(torch.load(net_path))
        #self.hidden_size = 3072
        self.hidden_size = hidden_size
        self.hidden = self.init_hidden(1)
        self.conv = nn.Sequential(*(list(resnet.children())[:-1]))
        self.fc = nn.Linear(hidden_size, 7)
        self.lstm = nn.LSTM(3072, self.hidden_size, batch_first=True)
        #self.classifier = nn.Linear(self.hidden_size, 7)


    def forward(self, inputs, hidden=None):
        inputs = self.conv(inputs)
        inputs = inputs.view(1, inputs.size(0), -1)
        output, hidden = self.lstm(inputs, self.hidden)
        output = output.view(output.size(1), output.size(2))
        output = self.fc(output)
        return output, hidden


    def init_hidden(self, batch_size):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))


def train(epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    total = data_sizes['1']
    num_run = 0
    batch_size = 64

    for ep in range(epoch):

        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader['1']:
            #inputs.to(device)
            #labels.to(device)

            num_run += 1
            current = num_run * batch_size
            progress_out(current, total)
            
            optimizer.zero_grad()

            output, hidden = rnn(inputs)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, pred = output.max(1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(pred == labels)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total

        print("Epoch: {} Loss: {:.4f} Acc: {:.4f}".format(
                  ep, epoch_loss, epoch_acc)
                  )





rnn = LSTM(7)
batch_size = 64
net_type = 'ResNet18'
training_folder, validation_folder = setup_dataset_folders()
data_folders = training_folder + validation_folder
data_folders = ['1']
cholec = generate_dataset(data_folders)
train_loader = generate_dataloader(cholec, data_folders, batch_size)
data_sizes = get_dataset_sizes(cholec, data_folders)
device = set_device()
optimizer = optim.Adam(rnn.parameters(), 0.001)

resnet = models.resnet18()
resnet.fc = nn.Linear(3072, 7)
resnet.load_state_dict(torch.load(net_path))


print(data_sizes['1'])

train(1)

