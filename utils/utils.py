import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.notebook import tqdm
import torch


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
        plt.savefig("./output/"+title +
                    datetime.today().strftime('%Y_%m_%d_%H_%M')+'.png')
    #plt.show()


def show_distribution(train, val, save):

    train_counted = torch.tensor(10)
    for data, label in tqdm(train):
        train_counted = train_counted + torch.bincount(label)

    val_counted = torch.tensor(10)
    for data, label in tqdm(val):
        val_counted = val_counted + torch.bincount(label)

    r = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    barWidth = 1
    plt.bar(r, train_counted.numpy(), color='#7f6d5f',
            edgecolor='white', width=barWidth)
    plt.bar(r, val_counted.numpy(), bottom=train_counted.numpy(), color='#557f2d',
            edgecolor='white', width=barWidth)
    plt.show()


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
        plt.savefig("./output/"+"Conf_Matr_" + title +
                    datetime.today().strftime('%Y_%m_%d_%H_%M')+'.png')
    plt.show()
