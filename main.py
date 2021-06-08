import torch
from utils.dataset import ImagenetteDataset, load_data
from utils.utils import imshow, plot_metrics, show_distribution
from CNN.cnnmodel import CNNModel
from ViTModel.vitmodel import ViTModel


def main():
    """ Main program """

    torch.manual_seed(42)
    train, val = load_data()

    #cnnModel = CNNModel()
    #metrics = cnnModel.train(train, val, 25)
    #plot_metrics(metrics, "CNN", True)
    #torch.cuda.empty_cache()



    vitModel = ViTModel()
    metrics = vitModel.train(train, val, 25)
    torch.cuda.empty_cache()
    plot_metrics(metrics, "ViT", True)

    return 0


if __name__ == "__main__":
    main()
