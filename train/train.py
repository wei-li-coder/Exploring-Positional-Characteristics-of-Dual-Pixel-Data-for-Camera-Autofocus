import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import MobileNetV2
import torchvision.models.mobilenet
from load_data import MyDataset
import torchvision
import torch.nn.functional as F
import csv
import random

def loss_ordinal_diaz(logits, labels, diaz_coef, num_classes):
    expand_labels = torch.tile(torch.unsqueeze(labels, 1), [1, num_classes])
    encoded_vector = torch.tile(torch.unsqueeze(torch.tensor(range(num_classes)), 0), [logits.shape[0], 1])
    criterion = -(encoded_vector - expand_labels) * (encoded_vector - expand_labels)
    criterion = criterion.type(torch.float32) / diaz_coef
    gt = F.softmax(criterion, dim=1)
    loss = torch.nn.CrossEntropyLoss()(logits, gt.to(device))
    return loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.ToTensor()
                                 ]),
    "test": transforms.Compose([transforms.ToTensor()
                               ])}

# rootpath for dataset
root = '/data/wl/autofocus/learn2focus/dataset/'
train_dataset=MyDataset(txt=root+'train_set.txt', transform=data_transform["train"])

train_num = len(train_dataset)
print('num_of_trainData:', train_num)

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=8)

num_classes = 49
net = MobileNetV2(num_classes)
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5, 0.999))
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75, eta_min=0)

seed = random.randint(1,10000)
path = "/data/wl/autofocus/codes/iccv23/"
file_dir = path + str(seed)
os.makedirs(file_dir)

for epoch in range(50):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        # for 1 step prediction, the coefficient is 1; for 2 step prediction, the coefficient is 0.5
        loss = loss_ordinal_diaz(logits, labels, 1, num_classes)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
    print('\nTotal step is:', step)
    torch.save(net, file_dir + '/MobileNetV2_{}.pt'.format(epoch + 1))

print('Finished Training')
