import torch
from utils.dataset import ImagenetteDataset, just_load_resize_pil, load_data, load_single_image, load_image_and_mirror
from utils.attentionmap import visualise_attention, generate_attention_map, sliding_window_method
from utils.lime_vis import vis_and_save
from utils.plotting_utils import imshow, plot_aumentation, plot_class_images, plot_confusion_matrix, plot_data_preprocessing, plot_metrics, show_distribution, plot_patches
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

from benchmarks.consistency import cnn_consitency, vit_consitency
import torch.nn as nn
from findpeaks import findpeaks
# Standard imports
import cv2

import numpy as np
import matplotlib.pyplot as plt
import random

from skimage import measure
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries


def main():
    """ Main program """

    # empty cache and seeding
    torch.cuda.empty_cache()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # load train val test set
    train, val, test = load_data(batch_size=16)

    # Load current models
    cnnModel = CNNModel(load=True, path="./savedModels/cnn_newArch_4.pt")
    vitModel = ViTModel(load=True, path="./savedModels/vit_smallerPatches.pt")

    # cnnModel.eval_metric(test)
    # vitModel.eval_metric(test)

    path = []
    with open('./utils/constants.json') as json_file:
        data = json.load(json_file)
        for p in data["img"]:
            path.append(p["path"])

            #x, y = load_single_image( p["path"])
            #temp, mask = cnnModel.lime_and_explain(y)
            #vis_and_save(mask, p["path"])

            #x, y = load_single_image(p["path"])
            #preds, attns = vitModel.predict_and_attents(x)
            #visualise_attention(attns, 16, 20, 320, p["path"])

    print("ViT Consitency: " + str(vit_consitency(vitModel, path)))
    #print("CNN Consitency: " + str(cnn_consitency(cnnModel, path)))

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
