import os
import re
import copy
import torch
from collections import OrderedDict
from train import *

model_path = '/media/TCO/TCO-Studenten/justaviju/Pretraining_Sebastian.pkl'


class SimResNet(nn.Module):
    def __init__(self):
        super(SimResNet, self).__init__()
        self.resnet = torchvision.models.resnet152(pretrained=True)
        self.classifier = nn.Sequential(nn.Linear(32768, 4096),
                                        nn.Sigmoid(),
                                        nn.Linear(4096, 4096),
                                        nn.Sigmoid(),
                                        nn.Linear(4096, 2))

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)

        return x


model_p = torch.load(model_path)
net = SimResNet()
net.load_state_dict(model_p)
sim_net_weights = copy.deepcopy(net.resnet.state_dict())

model = torchvision.models.resnet152()
model.load_state_dict(sim_net_weights)
model.fc = nn.Linear(12288, 7)

for name, layer in model._modules.items():
    if 'layer4' not in name and 'fc' not in name:
        for param in layer.parameters():
            param.requires_grad = False

trainable_layers = list(model.layer4.parameters()) + list(model.fc.parameters())

batch_size = 128
net_type = 'pickle'
data_folders = ['1', '2', '3', '4']
validation_folder = 0
cholec = generate_dataset(data_folders)
dataloaders = generate_dataloader(cholec, data_folders, batch_size, shuffling=True)
data_sizes = get_dataset_sizes(cholec, data_folders)
device = set_device()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.000001

model.to(device)
adam = True
if adam:
    optimizer_conv = optim.Adam(trainable_layers, lr=learning_rate)
# optim SGD
else:
    optimizer_conv = optim.SGD(trainable_layers, lr=learning_rate, momentum=0.9)

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.95)

try:
    model_pickle = train(model, criterion, optimizer_conv, exp_lr_scheduler, batch_size, learning_rate, data_sizes, dataloaders, data_folders, validation_folder, date, net_type, device, epochs=50)
    send_message("Training Finished. (pickle)")
except Exception as e:
    print(e)
    print("pickle crashed...")
    # send_message("pickle crashed...")