import torch
from utils.dataset import ImagenetteDataset, just_load_resize_pil, load_data, load_single_image, load_image_and_mirror
from utils.attentionmap import generate_multi_attention_map, generate_attention_map
from utils.lime_vis import vis_and_save
from utils.plotting_utils import imshow, plot_aumentation, plot_class_images, plot_confusion_matrix, plot_data_preprocessing, plot_metrics, show_distribution, plot_patches
from CNN.cnnmodel import CNNModel
from ViTModel.vitmodel import ViTModel


from tqdm.notebook import tqdm


import json
import numpy as np
import matplotlib.image as mpimg

from benchmarks.consistency import cnn_consitency, vit_consitency
import torch.nn as nn
from findpeaks import findpeaks
# Standard imports

import numpy as np
import matplotlib.pyplot as plt
import random

from skimage import measure
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries

from utils.masking_data import create_json
from benchmarks.correctness import cnn_correctness, vit_correctness
from benchmarks.confidence import cnn_confidence, vit_confidence
from datetime import datetime


def main():
    """ Main program """

    # empty cache and seeding
    torch.cuda.empty_cache()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # load train val test set
    train, val, test = load_data(batch_size=200)

    # Load current models
    cnnModel = CNNModel(load=True, path="./savedModels/cnn_resnet_3.pt")
    vitModel = ViTModel(load=True, path="./savedModels/vit_smallerPatches.pt")

    calc_benchmarks(test, num_cases=10, cnn=cnnModel, vit=vitModel)

    return 0


def train_and_eval_models(train, test, val):

    # Train and save CNN Model
    cnnModel = CNNModel(load=True, pretrained=True)
    cnn_metrics = cnnModel.train_and_val(train, val, 75)
    cnnModel.save_model("./savedModels/cnn_resnet_3.pt")

    # Train and save ViT Model
    vitModel = ViTModel()
    vit_metrics = vitModel.train(train, val, 16)
    vitModel.save_model("./savedModels/vit_smallerPatches.pt")

    # Eval Model with Test
    plot_metrics(cnn_metrics, "CNN_ResNet", True)
    plot_metrics(vit_metrics, "ViT", True)
    torch.cuda.empty_cache()

    cnn_matrix = cnnModel.conv_matrix(test, 10)
    vit_matrix = vitModel.conv_matrix(test, 10)

    plot_confusion_matrix(cnn_matrix, "CNN_ResNet", True)
    plot_confusion_matrix(vit_matrix, "ViT", True)


def calc_benchmarks(test, num_cases, cnn, vit):

    array = create_json(load_from_file=False)
    path = './output/benchmarks/values'+datetime.now().strftime('%Y_%m_%d_%H_%M') + \
        '_numCases_'+str(num_cases)+'.json'

    vals = {"num_cases": num_cases, "cnn": {}, "vit": {}}

    
    x, y = cnn.eval_metric(test)
    vals["cnn"]["F1 Score"] = x
    vals["cnn"]["Acc"] = y

    x, y = vit.eval_metric(test)
    vals["vit"]["F1 Score"] = x
    vals["vit"]["Acc"] = y
    print("Performance done")

    vals["cnn"]["Consitency"] = cnn_consitency(cnn, array, num_cases)
    vals["vit"]["Consitency"] = vit_consitency(cnn, array, num_cases)
    print("Consitency done")

    x, y = cnn_correctness(cnn, array, num_cases)
    vals["cnn"]["Correctness"] = {"acc low images": x, "acc masked images": y}

    x, y = vit_correctness(cnn, array, num_cases)
    vals["vit"]["Correctness"] = {"acc low images": x, "acc masked images": y}
    print("Consitency done")

    vals["cnn"]["Confidence"] = cnn_confidence(cnn, array, num_cases)
    vals["vit"]["Confidence"] = vit_confidence(vit, array, num_cases)
    print("Confidence done")
    
    with open(path, 'w') as fp:
        json.dump(vals, fp,  indent=4)
        print("saved json: " + path)


if __name__ == "__main__":
    main()
