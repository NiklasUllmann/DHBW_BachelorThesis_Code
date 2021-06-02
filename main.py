import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image

from dataset import ImagenetteDataset, load_data
from utils import imshow


def main():
    """ Main program """

    torch.manual_seed(42)




    train, val = load_data()

    it2 = iter(train)
    images2, labels2 = next(it2)

    imshow(torchvision.utils.make_grid(images2[:4]), labels2[:4])


    return 0


if __name__ == "__main__":
    main()
