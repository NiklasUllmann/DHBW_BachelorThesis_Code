import torch
from utils.dataset import ImagenetteDataset, load_data
from utils.utils import imshow, plot_metrics, show_distribution, plot_confusion_matrix
from CNN.cnnmodel import CNNModel
from ViTModel.vitmodel import ViTModel
import time


def main():
    """ Main program """

    torch.manual_seed(42)
    train, val = load_data()

    #cnnModel = CNNModel()
    #metrics = cnnModel.train(train, val, 25)
    #plot_metrics(metrics, "CNN", True)

    # cnnModel.save_model("./savedModels/cnn.pt")

    # torch.cuda.empty_cache()
    # time.sleep(100)

    #vitModel = ViTModel()
    #metrics = vitModel.train(train, val, 25)
    #plot_metrics(metrics, "ViT", True)
    # vitModel.save_model("./savedModels/vit.pt")

    cnnModel = CNNModel(load=True, path="./savedModels/cnn.pt")
    vitModel = ViTModel(load=True, path="./savedModels/cnn.pt")

    cnn_matrix = cnnModel.conv_matrix(val, 10)
    vit_matrix = vitModel.conv_matrix(val, 10)

    plot_confusion_matrix(cnn_matrix, "CNN", True)
    plot_confusion_matrix(vit_matrix, "ViT", True)

    return 0


if __name__ == "__main__":
    main()
