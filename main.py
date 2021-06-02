import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image

from dataset import ImagenetteDataset, load_data
from utils import imshow

import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from linformer import Linformer
from ViT.vit import ViT

from CNN.cnn import CNN


def main():
    """ Main program """

    torch.manual_seed(42)
    train, val = load_data()

    # it2 = iter(train)
    # images2, labels2 = next(it2)

    # imshow(torchvision.utils.make_grid(images2[:4]), labels2[:4])

    #using one gpu given to us by google colab for max 40 epochs
    myTrainer=pl.Trainer()

    model=CNN()
    myTrainer.fit(model, train, val)

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = ViT(
        dim=128,
        image_size=320,
        patch_size=20,
        num_classes=10,
        channels=3,
        depth=3,
        heads=6,
        mlp_dim=10
    ).to(device)

    from torch.optim.lr_scheduler import StepLR
    import torch.optim as optim
    from tqdm.notebook import tqdm



    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(5):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train):
            data = data.to(device)
            label = torch.tensor(label).to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train)
            epoch_loss += loss / len(train)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in val:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(val)
                epoch_val_loss += val_loss / len(val)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
    """
    return 0


if __name__ == "__main__":
    main()
