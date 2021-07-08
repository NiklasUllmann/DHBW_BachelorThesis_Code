from utils.dataset import load_image_and_mirror
from utils.logic import XNOR
from utils.attentionmap import generate_attention_map
import numpy as np


def cnn_consitency(model, list_of_paths):

    avgs = []

    for p in list_of_paths:
        x, x_mir, y, y_mir = load_image_and_mirror(p)
        temp, mask = model.lime_and_explain(y)
        temp_mir, mask_mir = model.lime_and_explain(y_mir)

        avgs.append(compare_bitmasks(mask, np.fliplr(mask_mir)))

    return sum(avgs)/len(avgs)


def vit_consitency(model, list_of_paths, ep):
    avgs = []

    for p in list_of_paths:
        x, x_mir, y, y_mir = load_image_and_mirror(p)
        preds, attn = model.predict_and_attents(x)
        preds_mir, attn_mir = model.predict_and_attents(x_mir)

        avgs.append(compare_attn_maps(generate_attention_map(attn, 20, 16), np.fliplr(generate_attention_map(attn_mir, 20, 16)), ep))

    return sum(avgs)/len(avgs)


def compare_bitmasks(mask, mask_mir):

    h, w = mask.shape
    ones = 0

    for x in range(0, h):
        for y in range(0, w):
            ones += XNOR(mask[x, y], mask_mir[x, y])
    return ones / (h*w)


def compare_attn_maps(attn_map, attn_map_mir, ep):
    h, w = attn_map.shape
    ones = 0

    for x in range(0, h):
        for y in range(0, w):
            if abs(attn_map[x, y]- attn_map_mir[x, y]) <= ep:
                ones+=1
    return ones / (h*w)


