import torch
import pytorch_lightning as pl
import torchvision


from dataset import ImagenetteDataset, load_data
from utils import imshow, plot_metrics


from CNN.cnn import CNN
from CNN.cnn2 import CNNModel
from ViTModel.vitmodel import ViTModel
import torch.nn as nn


def main():
    """ Main program """

    torch.manual_seed(42)
    train, val = load_data()
    """
    dataiter = iter(train)
    images, labels = dataiter.next()
    conv1 = nn.Conv2d(3, 6, 5)
    conv2 = nn.Conv2d(6, 6, 5)
    pool = nn.MaxPool2d(5, 5)
    print(images.shape)

    x = conv1(images)
    print(x.shape)
    x = pool(x)
    print(x.shape)
    x = conv2(x)
    print(x.shape)
    x = pool(x)
    print(x.shape)
    """
    cnnModel = CNNModel()
    metrics = cnnModel.train(train, val, 5)

    #imshow(torchvision.utils.make_grid(images2[:4]), labels2[:4])

    #myTrainer = pl.Trainer(gpus=-1, max_epochs=10, benchmark=True)

    #model = CNN()
    #myTrainer.fit(model, train, val)

    #vitModel = ViTModel()
    #metrics = vitModel.train(train, val, 5)

    #stuff = {'train_loss_record': [1.9909438, 1.7835513, 1.6459688, 1.5096759, 1.4031097], 'train_acc_record': [0.29729727, 0.38798562, 0.43105996, 0.48279142, 0.52174836], 'val_loss_record': [    2.7187908, 2.501501, 2.4107258, 2.4315443, 2.0897255], 'val_acc_record': [0.011474609, 0.101347655, 0.18351562, 0.1968164, 0.30956054]}

    #plot_metrics(stuff, "ViT", True)

    return 0


if __name__ == "__main__":
    main()
