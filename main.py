import json
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from numpy.core.fromnumeric import squeeze

from benchmarks.confidence import (cnn_confidence, cnn_mask_image,
                                   vit_confidence, vit_mask_image)
from benchmarks.consistency import cnn_consitency, vit_consitency
from benchmarks.correctness import cnn_correctness, vit_correctness
from CNN.cnnmodel import CNNModel
from utils.attentionmap import (generate_attention_map,
                                generate_multi_attention_map,
                                sliding_window_method, visualise_attention)
from utils.dataset import (ImagenetteDataset, just_load_resize_pil, load_data,
                           load_image_and_mirror, load_single_image)
from utils.lime_vis import vis_and_save
from utils.masking_data import create_json
from utils.plotting_utils import (imshow, plot_aumentation, plot_class_images,
                                  plot_confusion_matrix,
                                  plot_data_preprocessing, plot_metrics,
                                  plot_patches, show_distribution)
from ViTModel.vitmodel import ViTModel

# Standard imports





def main():
    """ Main program """

    # empty cache and seeding
    torch.cuda.empty_cache()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # load train val test set
    train, val, test = load_data(batch_size=8)

    # Load current models
    #cnnModel = CNNModel(load=True, path="./savedModels/cnn_resnet_3.pt")
    #vitModel = ViTModel(load=True, path="./savedModels/vit_smallerPatches.pt")
    plot_patches("ele.jpg")
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
    """
    x, y = cnn.eval_metric(test)
    vals["cnn"]["F1 Score"] = x
    vals["cnn"]["Acc"] = y

    x, y = vit.eval_metric(test)
    vals["vit"]["F1 Score"] = x
    vals["vit"]["Acc"] = y
    print("Performance done")

    vals["cnn"]["Consitency"] = cnn_consitency(cnn, array, num_cases)
    vals["vit"]["Consitency"] = vit_consitency(vit, array, num_cases)
    print("Consitency done")

    x, y = cnn_correctness(cnn, array, num_cases)
    vals["cnn"]["Correctness"] = {"acc low images": x, "acc masked images": y}

    x, y = vit_correctness(vit, array, num_cases)
    vals["vit"]["Correctness"] = {"acc low images": x, "acc masked images": y}
    print("Correctness done")

    x, y, z = cnn_confidence(cnn, array, num_cases)
    vals["cnn"]["Confidence"] = {"prob low images": x,
                                 "prob masked images": y, "mean prob dif": z}
    """
    x, y, z = vit_confidence(vit, array, num_cases)
    vals["vit"]["Confidence"] = {"prob low images": x,
                                 "prob masked images": y, "mean prob dif": z}
    print("Confidence done")

    with open(path, 'w') as fp:
        json.dump(vals, fp,  indent=4)
        print("saved json: " + path)


if __name__ == "__main__":
    main()
