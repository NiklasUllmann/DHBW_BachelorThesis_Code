import torch
from utils.dataset import ImagenetteDataset, just_load_resize_pil, load_data, load_single_image
from utils.attentionmap import visualise_attention
from utils.lime_vis import vis_and_save
from utils.utils import imshow, plot_aumentation, plot_class_images, plot_confusion_matrix, plot_data_preprocessing, plot_metrics, show_distribution
from CNN.cnnmodel import CNNModel
from ViTModel.vitmodel import ViTModel
import time

import shap
import matplotlib.pyplot as plt
from matplotlib import cm

from PIL import Image
import json
import numpy as np
import matplotlib.image as mpimg


def main():
    """ Main program """

    
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    train, val, test = load_data(batch_size=256)

    show_distribution(train, val, test, True)

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
    vitModel = ViTModel(load=True, path="./savedModels/vit_smallerPatches.pt")
    """
    with open('./utils/constants.json') as json_file:
        data = json.load(json_file)
        for p in data["img"]:
            tensor, pil = load_single_image(p["path"])
            preds, attn = vitModel.predict_and_attents(tensor)
            visualise_attention(attn, 16, 20, 320, p["path"])
            temp, mask = cnnModel.lime_and_explain(pil)
            vis_and_save(mask, temp, p["path"])
    

    """
    
    
    


# cnn_matrix = cnnModel.conv_matrix(val, 10)
    #vit_matrix = vitModel.conv_matrix(val, 10)

# plot_confusion_matrix(cnn_matrix, "CNN", True)
    #plot_confusion_matrix(vit_matrix, "ViT", True)

    return 0


if __name__ == "__main__":
    main()
