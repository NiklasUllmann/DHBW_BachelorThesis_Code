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

        )

        self.fc_layer = nn.Sequential(
            nn.Linear(12*77*77, 1024),
             nn.ReLU(), 
             nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)

        )


    def forward(self, x):
        x = self.conv_Layer(x)
        x = x.view(-1, 12*77*77)
        x = self.fc_layer(x)
        return x
