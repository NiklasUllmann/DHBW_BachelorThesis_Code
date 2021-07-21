from utils.dataset import load_image_and_mirror
from utils.logic import XNOR
from utils.attentionmap import generate_attention_map, sliding_window_method
import numpy as np


def cnn_consitency(model, list_of_paths):
    avgs = []
    for i in list_of_paths:
        for key, value in i.items():
            x, x_mir, y, y_mir = load_image_and_mirror("./data/"+value["path"])
            temp, mask = model.lime_and_explain(y, value["class"])
            temp_mir, mask_mir = model.lime_and_explain(y_mir, value["class"])

            avgs.append(compare_bitmasks(mask, np.fliplr(mask_mir)))

    return sum(avgs)/len(avgs)


def vit_consitency(model, list_of_paths):
    avgs = []
    for i in list_of_paths:
        for key, value in i.items():
            x, x_mir, y, y_mir = load_image_and_mirror("./data/"+value["path"])
        preds, attn = model.predict_and_attents(x)
        preds_mir, attn_mir = model.predict_and_attents(x_mir)

        bit_mask = sliding_window_method(generate_attention_map(attn, 20, 16))
        bit_mask_mir = np.fliplr(sliding_window_method(
            generate_attention_map(attn_mir, 20, 16)))

        avgs.append(compare_bitmasks(bit_mask, bit_mask_mir))

    return sum(avgs)/len(avgs)


def compare_bitmasks(mask, mask_mir):

    whole_array = np.logical_or(mask, mask_mir)
    whole = count_ones(whole_array)

    same_array = np.logical_and(mask, mask_mir)
    same = count_ones(same_array)

    return same / whole


def count_ones(a):
    return sum(np.count_nonzero(x == 1) for x in a)
