import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


class CNN(pl.LightningModule):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 320, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(320, 10, kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.dropout1 = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(250, 18)
        self.dropout2 = torch.nn.Dropout(0.08)
        self.fc2 = torch.nn.Linear(18, 10)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.leaky_relu(self.dropout2(x))
        return F.softmax(self.fc2(x))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):

        # extracting input and output from the batch
        x, labels = batch

        # doing a forward pass
        pred = self.forward(x)

        # calculating the loss
        loss = F.nnl_loss(pred, labels)

        # logs
        logs = {"train_loss": loss}

        output = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            # optional for logging purposes
            "log": logs,
        }

        return output
