import torch
from utils.dataset import ImagenetteDataset, load_data, load_single_image
from utils.attentionmap import visualise_attention
from utils.utils import imshow, plot_metrics, show_distribution, plot_confusion_matrix
from CNN.cnnmodel import CNNModel
from ViTModel.vitmodel import ViTModel
import time

import shap
import matplotlib.pyplot as plt
from matplotlib import cm

from PIL import Image


def main():
    """ Main program """

    torch.cuda.empty_cache()
    torch.manual_seed(42)
    train, val = load_data(batch_size=16)

    # cnnModel = CNNModel()
    # metrics = cnnModel.train_and_val(train, val, 50)
    # plot_metrics(metrics, "CNN", True)

    # cnnModel.save_model("./savedModels/cnn.pt")

    torch.cuda.empty_cache()

    #vitModel = ViTModel()
    #metrics = vitModel.train(train, val, 16)
    #plot_metrics(metrics, "ViT", True)
    # vitModel.save_model("./savedModels/vit_smallerPatches.pt")

    cnnModel = CNNModel(load=True, path="./savedModels/cnn_newArch.pt")
    #vitModel = ViTModel(load=True, path="./savedModels/vit_smallerPatches.pt")

    cnnModel.predict_and_explain(val)

    #paths = ["./data/val/n01440764/n01440764_27451.JPEG",             "./data/train/n03417042/n03417042_29408.JPEG", "./data/train/n03445777/ILSVRC2012_val_00003793.JPEG", "./data/train/n03888257/n03888257_73606.JPEG",             "./data/val/n03000684/ILSVRC2012_val_00007460.JPEG", "./data/val/n03000684/n03000684_34440.JPEG", "./data/val/n03417042/ILSVRC2012_val_00027150.JPEG",             "./data/val/n03425413/ILSVRC2012_val_00004452.JPEG", "./data/val/n03425413/n03425413_1242.JPEG", "./data/train/n02102040\ILSVRC2012_val_00001968.JPEG", "./data/train/n02979186/n02979186_844.JPEG", "./data/train/n03000684/n03000684_1015.JPEG", "./data/train/n03888257/ILSVRC2012_val_00026575.JPEG"]

    #for p in paths:
    #single_image = load_single_image(p)
    #preds, attn = vitModel.predict_and_attents(single_image)
    #visualise_attention(attn, 16, 20, 320, p)


# cnn_matrix = cnnModel.conv_matrix(val, 10)
    #vit_matrix = vitModel.conv_matrix(val, 10)

# plot_confusion_matrix(cnn_matrix, "CNN", True)
    #plot_confusion_matrix(vit_matrix, "ViT", True)

    return 0


if __name__ == "__main__":
    main()
