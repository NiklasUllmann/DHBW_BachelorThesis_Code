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
import numpy as np
import shap


class CNNModel():
    def __init__(self, load=False, path=None):

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        if(not load):
            self.model = CNN().to(self.device)
            print("init CNN")
        else:
            self.model = torch.load(path)
            print("load CNN")

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-5)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.7)

    def train_and_val(self, train, val, epoch):
        train_loss_record = []
        train_acc_record = []
        val_loss_record = []
        val_acc_record = []
        self.model.train()
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

    def conv_matrix(self, val, class_count):
        confusion_matrix = np.zeros((class_count, class_count))
        with torch.no_grad():
            for i, (inputs, classes) in enumerate(val):
                inputs = inputs.to(self.device)
                classes = classes.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        return confusion_matrix

    def save_model(self, path):
        torch.save(self.model, path)

    def predict_and_explain(self, val):
        batch = next(iter(val))
        images, _ = batch

        images = images.to(self.device)
        background = images[:30]
        test_images = images[30:31]

        e = shap.DeepExplainer(self.model, background)
        shap_values = e.shap_values(test_images)
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2)
                      for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(
            test_images.detach().cpu().numpy(), 1, -1), 1, 2)

        shap.image_plot(shap_numpy, -test_numpy, show=True)

    def printnorm(self, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        print('Inside ' + self.__class__.__name__ + ' forward')
        print('')
        print('input: ', type(input))
        print('input[0]: ', type(input[0]))
        print('output: ', type(output))
        print('')
        print('input size:', input[0].size())
        print('output size:', output.data.size())
        print('output norm:', output.data.norm())
