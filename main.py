import torch
from utils.dataset import ImagenetteDataset, just_load_resize_pil, load_data, load_single_image, load_image_and_mirror
from utils.attentionmap import visualise_attention
from utils.lime_vis import vis_and_save
from utils.plotting_utils import imshow, plot_aumentation, plot_class_images, plot_confusion_matrix, plot_data_preprocessing, plot_metrics, show_distribution
from CNN.cnnmodel import CNNModel
from ViTModel.vitmodel import ViTModel
import time

from utils.logic import XNOR

import shap
import matplotlib.pyplot as plt
from matplotlib import cm

from PIL import Image
import json
import numpy as np
import matplotlib.image as mpimg

from benchmarks.consistency import cnn_consitency


def main():
    """ Main program """

    # empty cache and load train, val test
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    train, val, test = load_data(batch_size=256)

    torch.cuda.empty_cache()

    # Load current models
    cnnModel = CNNModel(load=True, path="./savedModels/cnn_newArch.pt")
    vitModel = ViTModel(load=True, path="./savedModels/vit_smallerPatches.pt")


    path = []
    with open('./utils/constants.json') as json_file:
        data = json.load(json_file)
        for p in data["img"]:

            path.append(p["path"])
            """
            tensor, pil = load_single_image(p["path"])
            preds, attn = vitModel.predict_and_attents(tensor)
            visualise_attention(attn, 16, 20, 320, p["path"])
            temp, mask = cnnModel.lime_and_explain(pil)
            vis_and_save(mask, temp, p["path"])
            """
    

    

    

    print(cnn_consitency(cnnModel, path))
    
    


    return 0


def train_and_eval_models(train, test, val):

    # Train and save CNN Model
    cnnModel = CNNModel()
    cnn_metrics = cnnModel.train_and_val(train, val, 50)
    cnnModel.save_model("./savedModels/cnn.pt")

    # Train and save ViT Model
    vitModel = ViTModel()
    vit_metrics = vitModel.train(train, val, 16)
    vitModel.save_model("./savedModels/vit_smallerPatches.pt")


    # Eval Model with Test
    plot_metrics(cnn_metrics, "CNN", True)
    plot_metrics(vit_metrics, "ViT", True)

    cnn_matrix = cnnModel.conv_matrix(test, 10)
    vit_matrix = vitModel.conv_matrix(test, 10)

    plot_confusion_matrix(cnn_matrix, "CNN", True)
    plot_confusion_matrix(vit_matrix, "ViT", True)


if __name__ == "__main__":
    main()
