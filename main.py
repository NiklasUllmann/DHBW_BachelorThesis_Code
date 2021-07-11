import torch
from utils.dataset import ImagenetteDataset, just_load_resize_pil, load_data, load_single_image, load_image_and_mirror
from utils.attentionmap import visualise_attention, generate_attention_map
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
    cnnModel = CNNModel(load=True, path="./savedModels/cnn_newArch.pt")
    vitModel = ViTModel(load=True, path="./savedModels/vit_smallerPatches.pt")

    cnnModel.eval_metric(test)
    #vitModel.eval_metric(test)

    sliding_window_method(
        vitModel, "./data/val/n03000684/ILSVRC2012_val_00029211.JPEG")

    """
    path = []
    with open('./utils/constants.json') as json_file:
        data = json.load(json_file)
        for p in data["img"]:

            path.append(p["path"])
            
            tensor, pil = load_single_image(p["path"])
            preds, attn = vitModel.predict_and_attents(tensor)
            visualise_attention(attn, 16, 20, 320, p["path"])
            temp, mask = cnnModel.lime_and_explain(pil)
            vis_and_save(mask, temp, p["path"])
            

    # for i in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025, 0.01]:
    # print(str(i) + str(vit_consitency(vitModel, path, i)))
    # print(cnn_consitency(cnnModel, path))
    """
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


def segmentation_method(vitModel):
    scale = 500
    sigma = 1.5
    min_size = 500

    x, x_mir, y, y_mir = load_image_and_mirror(
        "./data/val/n03417042/ILSVRC2012_val_00006922.JPEG")
    preds, attn = vitModel.predict_and_attents(x)
    a_map = generate_attention_map(attn, patches_per_row=20, patch_size=16)

    segments_fz = felzenszwalb(
        a_map, scale=scale, sigma=sigma, min_size=min_size)

    print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")

    fig, ax = plt.subplots(2, int(len(np.unique(segments_fz))/2),
                           figsize=(10, 10), sharex=True, sharey=True)

    for i in range(int(len(np.unique(segments_fz))/2)):
        ax[0, i].imshow(mark_boundaries(a_map, [segments_fz[:, :] == i][0]))

    for i in range(int(len(np.unique(segments_fz))/2), int(len(np.unique(segments_fz)))):
        ax[1, int(len(np.unique(segments_fz))/2) -
           i].imshow(mark_boundaries(a_map, [segments_fz[:, :] == i][0]))

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()


def contour_method(vitModel):
    x, x_mir, y, y_mir = load_image_and_mirror(
        "./data/val/n03417042/ILSVRC2012_val_00006922.JPEG")
    preds, attn = vitModel.predict_and_attents(x)
    a_map = generate_attention_map(attn, patches_per_row=20, patch_size=16)
    # Find contours at a constant value of 0.8
    contours = measure.find_contours(a_map, 0.5)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(a_map, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def sliding_window_method(vitModel, path):
    x, x_mir, y, y_mir = load_image_and_mirror(
        path)
    preds, attn = vitModel.predict_and_attents(x)
    a_map = generate_attention_map(attn, patches_per_row=20, patch_size=16)
    a_map_orig = a_map

    windows_size = 25
    avg_max = 0

    bit_mask = np.zeros((320, 320))

    i_max = 0
    i_end_max = 0
    j_max = 0
    j_end_max = 0

    fig, ax = plt.subplots(1, 20, figsize=(10, 10), sharex=True, sharey=True)

    for a in range(20):
        i_max = 0
        i_end_max = 0
        j_max = 0
        j_end_max = 0
        avg_max = 0
        for i in range(0, a_map.shape[0] - windows_size + 1):
            for j in range(0, a_map.shape[1]-windows_size+1):
                window = a_map[i: i + windows_size, j: j + windows_size]

                if (np.mean(window) >= avg_max):
                    i_max = i
                    i_end_max = i+windows_size
                    j_max = j
                    j_end_max = j + windows_size
                    avg_max = np.mean(window)

        bit_mask[i_max:i_end_max, j_max:j_end_max] = 1
        a_map[i_max:i_end_max, j_max:j_end_max] = 0

        ax[a].imshow(bit_mask)
    plt.show()

    plt.imshow(a_map_orig)
    plt.show()

    return None


if __name__ == "__main__":
    main()
