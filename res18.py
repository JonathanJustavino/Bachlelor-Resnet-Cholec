from train import *


batch_size = 64
net_type = 'ResNet18'
training_folder, validation_folder = setup_dataset_folders()
data_folders = training_folder + validation_folder
cholec = generate_dataset(data_folders)
dataloaders = generate_dataloader(cholec, data_folders, batch_size)


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

model_conv = model_conv.to(set_device())
criterion = nn.CrossEntropyLoss()
trainable_layers = list(model_conv.layer4.parameters()) + list(model_conv.fc.parameters())

learning_rate = 0.0005
validation_set = validation_folder[0]

# optim Adam
adam = True
if adam:
    optimizer_conv = optim.Adam(trainable_layers, lr=learning_rate)
# optim SGD
else:
    optimizer_conv = optim.SGD(trainable_layers, lr=learning_rate, momentum=0.9)


exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
print("Optimizer", optimizer_conv)
# send_message("Training Started...({})".format(net_type))

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

try:
    model_conv = train(model_conv, criterion, optimizer_conv, exp_lr_scheduler, batch_size, learning_rate, validation_set, date, net_type, epochs=150)
    torch.save(model_conv.state_dict(), os.path.join(net_path, "{}_model".format(net_type)))
    # send_message("Training Finished. (ResNet18)")
except Exception as e:
    print(e)
    print("ResNet18 crashed...")
    # send_message("ResNet18 crashed...")