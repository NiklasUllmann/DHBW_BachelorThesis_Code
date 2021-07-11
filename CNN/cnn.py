import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_Layer = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 12, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(12, 24, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

        )

        self.fc_layer = nn.Sequential(
            nn.Linear(24*36*36, 8192),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)

        )


    def forward(self, x):
        x = self.conv_Layer(x)
        x = x.view(-1, 24*36*36)
        x = self.fc_layer(x)
        return x
