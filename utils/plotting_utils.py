import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.notebook import tqdm
import torch
import json
from utils.dataset import just_load_resize_pil, load_single_image, pil_augmentation
from PIL import Image

OUTPUT_PATH = "./output/"


def imshow(img, label):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(str(label))
    plt.show()


def plot_metrics(metric_map, title, save):

    loss_train = metric_map.get('train_loss_record')
    loss_val = metric_map.get('val_loss_record')

    acc_train = metric_map.get('train_acc_record')
    acc_val = metric_map.get('val_acc_record')
    epochs = range(1, len(metric_map.get('val_loss_record'))+1)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle(title+datetime.today().strftime('%Y_%m_%d'))
    ax1.plot(epochs, loss_train, 'g', label='Training Loss')
    ax1.plot(epochs, loss_val, 'b', label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    ax2.plot(epochs, acc_train, 'g', label='Training Acc')
    ax2.plot(epochs, acc_val, 'b', label='Validation Acc')
    ax2.set_xlabel('Epochs')
    ax2.legend()

    ax1.set_title("Loss")
    ax2.set_title("Acc")
    if(save):
        plt.savefig(OUTPUT_PATH+title +
                    datetime.today().strftime('%Y_%m_%d_%H_%M')+'.png')
    #plt.show()


def show_distribution(train, val, test, save):

    train_counted = torch.tensor(10)
    for data, label in tqdm(train):
        train_counted = train_counted + torch.bincount(label, minlength=10)

    val_counted = torch.tensor(10)
    for data, label in tqdm(val):
        val_counted = val_counted + torch.bincount(label, minlength=10)

    test_counted = torch.tensor(10)
    for data, label in tqdm(test):
        test_counted = test_counted + torch.bincount(label, minlength=10)

    r = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    barWidth = 1
    plt.bar(r, train_counted.numpy(), color='#264653',
            edgecolor='white', width=barWidth, label="Train")
    plt.bar(r, val_counted.numpy(), bottom=train_counted.numpy(),
            color='#E9C46A',           edgecolor='white', width=barWidth, label="Val")
    plt.bar(r, test_counted.numpy(), bottom=val_counted.numpy() + train_counted.numpy(),
            color='#E76F51', edgecolor='white', width=barWidth, label="Test")

    plt.legend()
    plt.xticks(r)
    plt.xlabel("Classes")
    plt.ylabel("Image Count")
    if(save):
        plt.savefig(OUTPUT_PATH+"data_Dist" +
                    datetime.today().strftime('%Y_%m_%d_%H_%M')+'.png')


def plot_confusion_matrix(conf_mat, title, save):

    plt.figure(figsize=(15, 10))

    class_names = ["tench", "springer",
                   "casette_player", "chain_saw", "church", "French_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute"]
    df_cm = pd.DataFrame(conf_mat, index=class_names,
                         columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if(save):
        plt.savefig(OUTPUT_PATH+"Conf_Matr_" + title +
                    datetime.today().strftime('%Y_%m_%d_%H_%M')+'.png')
    #plt.show()


def plot_class_images():

    fig, axes = plt.subplots(nrows=2, ncols=5, squeeze=False)

    with open('./utils/constants.json') as json_file:
        data = json.load(json_file)
        for p in data["img"]:

            pil = just_load_resize_pil(p["path"])
            a = np.asarray(pil)

            axes[p["row"], p["col"]].imshow(a)
            axes[p["row"], p["col"]].set_title(p["name"], fontsize=10)

            axes[p["row"], p["col"]].axis('off')

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH+'Class_Images.png')


def plot_aumentation():
    x, y, z = pil_augmentation(
        "./data/val/n03417042/ILSVRC2012_val_00006922.JPEG")

    fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=False)

    axes[0, 0].imshow(np.asarray(x))
    axes[0, 0].set_title("Original", fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.asarray(y))
    axes[0, 1].set_title("Flipped", fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.asarray(z))
    axes[0, 2].set_title("Mirrored", fontsize=10)
    axes[0, 2].axis('off')

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH+'Class_Images_Augmentation.png')


def plot_data_preprocessing():
    path = "./data/val/n03425413/ILSVRC2012_val_00004452.JPEG"

    o_image = Image.open(path)

    resized_img = just_load_resize_pil(path)
    x, normalized = load_single_image(path)

    fig, axes = plt.subplots(
        nrows=1, ncols=3, squeeze=False, constrained_layout=True)

    axes[0, 0].imshow(np.asarray(o_image))
    axes[0, 0].set_title("Original", fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.asarray(resized_img))
    axes[0, 1].set_title("Resized", fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.asarray(normalized))
    axes[0, 2].set_title("Normalized", fontsize=10)
    axes[0, 2].axis('off')

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH+'Class_Images_PreProcessing.png')
