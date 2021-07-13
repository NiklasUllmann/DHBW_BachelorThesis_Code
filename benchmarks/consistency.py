from utils.dataset import load_image_and_mirror
from utils.logic import XNOR
from utils.attentionmap import generate_attention_map, sliding_window_method
import numpy as np


def cnn_consitency(model, list_of_paths):

    avgs = []

    for p in list_of_paths:
        x, x_mir, y, y_mir = load_image_and_mirror(p)
        temp, mask = model.lime_and_explain(y)
        temp_mir, mask_mir = model.lime_and_explain(y_mir)

        avgs.append(compare_bitmasks(mask, np.fliplr(mask_mir)))

    return sum(avgs)/len(avgs)


def vit_consitency(model, list_of_paths):
    avgs = []

    for p in list_of_paths:
        x, x_mir, y, y_mir = load_image_and_mirror(p)
        preds, attn = model.predict_and_attents(x)
        preds_mir, attn_mir = model.predict_and_attents(x_mir)

        bit_mask = sliding_window_method(generate_attention_map(attn, 20, 16))
        bit_mask_mir = np.fliplr(sliding_window_method(generate_attention_map(attn_mir, 20, 16)))

        avgs.append(compare_bitmasks(bit_mask, bit_mask_mir))

    return sum(avgs)/len(avgs)


def compare_bitmasks(mask, mask_mir):

    h, w = mask.shape
    ones = 0

    for x in range(0, h):
        for y in range(0, w):
            ones += XNOR(mask[x, y], mask_mir[x, y])
    return ones / (h*w)




