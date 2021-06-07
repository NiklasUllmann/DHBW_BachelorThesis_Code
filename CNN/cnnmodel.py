import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from vit_pytorch import ViT
import matplotlib.pyplot as plt
import pandas as pd
from CNN.cnn import CNN


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
        for epoch in range(epoch):
            epoch_loss = 0
            epoch_accuracy = 0

            for data, label in tqdm(train):
                data = data.to(self.device)
                label = label.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / len(train)
                epoch_loss += loss / len(train)

            train_acc_record.append(
                epoch_accuracy.detach().cpu().numpy().flat[0])
            train_loss_record.append(epoch_loss.detach().cpu().numpy().flat[0])

            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for data, label in val:
                    data = data.to(self.device)
                    label = label.to(self.device)

                    val_output = self.model(data)
                    val_loss = self.criterion(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(val)
                    epoch_val_loss += val_loss / len(val)

                val_acc_record.append(
                    epoch_val_accuracy.detach().cpu().numpy().flat[0])
                val_loss_record.append(
                    epoch_val_loss.detach().cpu().numpy().flat[0])

            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
            )

        result = {"train_loss_record": train_loss_record, "train_acc_record": train_acc_record,
                  "val_loss_record": val_loss_record, "val_acc_record": val_acc_record}
        return result
