import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.optim as optim
from torchvision import models
from train import *
import numpy as np
from bot import *
import os


cnn_path = '/media/TCO/TCO-Studenten/justaviju/results/resnet18/Valset1/model'
rnn_path = '/media/TCO/TCO-Studenten/justaviju/results/rnns'
parent_result_folder_rnn = '/media/data/ToolClassification/results/rnns/'
parent_network_folder_rnn = '/media/TCO/TCO-Studenten/justaviju/results/rnns'

resnet = None
net = None
device = set_device()


class LSTM(nn.Module):
    def __init__(self, num_classes=7, hidden_size=1):
        super(LSTM, self).__init__()
        resnet = models.resnet18()
        resnet.fc = nn.Linear(3072, 7)
        resnet.load_state_dict(torch.load(cnn_path))
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
        self.hidden = (self.hidden[0].to(device), self.hidden[1].to(device))
        output, hidden = self.lstm(inputs, self.hidden)
        output = output.view(output.size(1), output.size(2))
        output = self.fc(output)
        return output, hidden


    def init_hidden(self, batch_size):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))


def training(model, data_folders, learning_rate, optimizer, date, b_size, resnet_type, path, epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    result_path = get_result_path("", parent_folder=parent_result_folder_rnn)
    print(result_path)
    net_path = get_net_path(net_type.lower(), parent_folder=parent_network_folder_rnn)

    predictions_path = os.path.join(result_path, "{}_predictions.csv".format(date))
    validation_folder = data_folders[-1]
    print(data_folders)
    print("Validationfolder: ", validation_folder)
    batch_size = b_size
    best_acc = 0.0

    for ep in range(epoch):
        print("Epoch {}/{}".format(ep, epoch))

        with open(os.path.join(result_path, date), 'a') as result_file:
            result_file.write("Epoch {}:\n".format(ep))
            for folder in data_folders:
                running_loss = 0.0
                running_corrects = 0
                num_run = 0
                total = data_sizes[folder]

                if folder != validation_folder:
                    model.train()
                else:
                    model.eval()
                    with open(predictions_path, 'a') as file:
                        file.write("\nEpoch {}\n".format(ep))

                for inputs, labels in train_loader[folder]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    num_run += 1
                    current = num_run * batch_size
                    progress_out(current, total)
                    
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(folder != validation_folder):

                        output, hidden = rnn(inputs)
                        _, pred = output.max(1)

                        loss = criterion(output, labels)

                        if folder != validation_folder:
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(pred == labels)

                    if folder == validation_folder and epoch % 5 == 0:
                        write_epoch_predictions(predictions_path,
                                                pred.cpu().numpy(),
                                                labels.cpu().numpy())

                epoch_loss = running_loss / total
                epoch_acc = running_corrects.double() / total

                result_file.write("\nSet: {} Loss: {:.4f} Acc: {:.4f}".format(folder, epoch_loss, epoch_acc))
                logger = "Lstm Epoch: {} Set: {} Loss: {:.4f} Acc: {:.4f}".format(ep, folder, epoch_loss, epoch_acc)
                print(logger)
                try:
                    send_message(logger)
                except:
                    print("Sending results was not successful")

                if folder == validation_folder and epoch_acc > best_acc:
                    epoch_of_best_acc = ep
                    best_acc = epoch_acc
                    try:
                        print("Saving...")
                        torch.save(model.lstm.state_dict(), os.path.join(path, 'lstm'))
                        torch.save(model.fc.state_dict(), os.path.join(path, 'classifier'))
                        torch.save(optimizer.state_dict(), os.path.join(path, "{}_optimizer_test".format(net_type)))
                        #torch.save(scheduler.state_dict(), os.path.join(net_path, "{}_scheduler_test".format(net_type)))
                    except:
                        e = sys.exc_info()[0]
                        print(e)
                        print("Saving failed")
    with open(os.path.join(result_path, date), 'a') as result_file:
        result_file.write("{} Best val Acc: {:4f} in epoch: {} \
        Learning rate: {}\n".format(net_type, best_acc,
                                    epoch_of_best_acc, learning_rate))
        result_file.write("Optimizer: {}".format(optimizer_conv))


rnn = LSTM(7)
batch_size = 90
net_type = 'lstm-18'
resnet_type = 'resnet18'
data_folders = ['1', '2', '3', '4']
cholec = generate_dataset(data_folders)
train_loader = generate_dataloader(cholec, data_folders, batch_size, shuffling=False)
data_sizes = get_dataset_sizes(cholec, data_folders)
learning_rate = 0.0001
optimizer = optim.Adam(rnn.parameters(), learning_rate)
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

rnn.to("cuda:0")
training(rnn, data_folders, learning_rate, optimizer, date, batch_size, resnet_type, rnn_path, epoch=100)

