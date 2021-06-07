import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from vit_pytorch import ViT
import matplotlib.pyplot as plt
import pandas as pd
from torchsummary import summary


# Define the CNN architecture


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.pool = nn.MaxPool2d(5, 5)
        self.fc1 = nn.Linear(726, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = x.view(-1, 6*11*11)
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNModel():
    def __init__(self):
        print("init")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)

    # loss function
    criterion = torch.nn.CrossEntropyLoss()
# optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
# scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    def train(self, train, val, epoch):
        train_loss_record = []
        train_acc_record = []
        val_loss_record = []
        val_acc_record = []

        for epoch in range(1, epoch):

            train_loss = 0
            valid_loss = 0

    # training steps
            #self.model.train()
            for batch_index, (data, target) in enumerate(train):
              # moves tensors to GPU
                data, target = data.cuda(), target.cuda()
        # clears gradients
                self.optimizer.zero_grad()
        # forward pass
                output = self.model(data)
        # loss in batch
                loss = self.criterion(output, target)
        # backward pass for loss gradient
                loss.backward()
        # update paremeters
                self.optimizer.step()
        # update training loss
                train_loss += loss.item()*data.size(0)

    # validation steps
           # self.model.eval()
            for batch_index, (data, target) in enumerate(val):
              # moves tensors to GPU
                data, target = data.cuda(), target.cuda()
        # forward pass
                output = self.model(data)
        # loss in batch
                loss = self.criterion(output, target)
        # update validation loss
                valid_loss += loss.item()*data.size(0)

    # average loss calculations
            train_loss = train_loss/len(train.sampler)
            valid_loss = valid_loss/len(val.sampler)

    # Display loss statistics
        print(
            f'Current Epoch: {epoch}\nTraining Loss: {round(train_loss, 6)}\nValidation Loss: {round(valid_loss, 6)}')
