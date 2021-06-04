import torch
import pytorch_lightning as pl
import torchvision


from dataset import ImagenetteDataset, load_data
from utils import imshow


from CNN.cnn import CNN
from ViTModel.vitmodel import ViTModel


def main():
    """ Main program """

    torch.manual_seed(42)
    train, val = load_data()

    #imshow(torchvision.utils.make_grid(images2[:4]), labels2[:4])

    myTrainer = pl.Trainer(gpus=-1, max_epochs=10, benchmark=True)

    model = CNN()
    myTrainer.fit(model, train, val)

    #vitModel = ViTModel()
    #vitModel.train(train, val, 10)

    return 0


if __name__ == "__main__":
    main()
