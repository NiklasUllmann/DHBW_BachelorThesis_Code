from utils.dataset import load_image_and_mirror
from utils.logic import XNOR


def cnn_consitency(model, list_of_paths):

    avgs = []

    for p in list_of_paths:
        x, x_mir, y, y_mir = load_image_and_mirror(p)
        temp, mask = model.lime_and_explain(y)
        temp_mir, mask_mir = model.lime_and_explain(y)

        avgs.append(compare_bitmasks(mask, mask_mir))

    return sum(avgs)/len(avgs)


def vit_consitency():
    return None


def compare_bitmasks(mask, mask_mir):

    h, w = mask.shape
    ones = 0

    for x in range(0, h):
        for y in range(0, w):
            ones += XNOR(mask[x, y], mask_mir[x, y])
    return ones / (h*w)
