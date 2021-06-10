import torch
from utils.dataset import ImagenetteDataset, load_data, load_single_image
from utils.utils import imshow, plot_metrics, show_distribution, plot_confusion_matrix
from CNN.cnnmodel import CNNModel
from ViTModel.vitmodel import ViTModel
import time

import shap
import numpy as np


def main():
    """ Main program """

    torch.cuda.empty_cache()
    torch.manual_seed(42)
    train, val = load_data()

    #cnnModel = CNNModel()
    #metrics = cnnModel.train(train, val, 25)
    #plot_metrics(metrics, "CNN", True)

    # cnnModel.save_model("./savedModels/cnn.pt")

    torch.cuda.empty_cache()

    #vitModel = ViTModel()
    #metrics = vitModel.train(train, val, 1)
    #plot_metrics(metrics, "ViT", True)
    #vitModel.save_model("./savedModels/vit_V1.pt")

    cnnModel = CNNModel(load=True, path="./savedModels/cnn.pt")
    #vitModel = ViTModel(load=True, path="./savedModels/vit.pt")

    #single_image = load_single_image(    "./data/val/n01440764/n01440764_27451.JPEG")

    cnnModel.predict_and_explain(val)
    #vitModel.predict_and_explain(single_image)

        #cnn_matrix = cnnModel.conv_matrix(val, 10)
        #vit_matrix = vitModel.conv_matrix(val, 10)

        #plot_confusion_matrix(cnn_matrix, "CNN", True)
        #plot_confusion_matrix(vit_matrix, "ViT", True)

    return 0


if __name__ == "__main__":
    main()
