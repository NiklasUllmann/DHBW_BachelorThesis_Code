from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
from utils.dataset import just_load_resize_pil, load_single_image
from torchvision import transforms
from sklearn.metrics import accuracy_score
from utils.attentionmap import generate_attention_map, sliding_window_method


def cnn_correctness(model, array):

    trans = transforms.ToTensor()
    classes = np.empty(0)
    low_classes = np.empty(0)
    masked_classes = np.empty(0)

    for i in array:
        for key, value in i.items():
            x, y = load_single_image("./data/"+value["path"])
            value["probab"] = model.predict(x)[0][value["class"]]

    for i in array:
        abc = list(dict(
            sorted(i.items(), key=lambda item: item[1]["probab"], reverse=True)).values())
        for x in range(0, int(len(abc)/2)):

            high_path = "./data/"+abc[x]["path"]
            low_path = "./data/"+abc[(len(abc)-1)-x]["path"]

            low_img, pil = load_single_image(low_path)
            masked_img = torch.unsqueeze(
                trans(cnn_mask_image(model, high_path, low_path, abc[x]["class"])), 0)

            classes = np.append(classes, abc[x]["class"])
            low_classes = np.append(
                low_classes, np.argmax(model.predict(low_img)[0]))
            masked_classes = np.append(
                masked_classes, np.argmax(model.predict(masked_img)[0]))

    print("CNN Acc with low images: " +
          str(accuracy_score(classes, low_classes)))
    print("CNN Acc with masked images: " +
          str(accuracy_score(classes, masked_classes)))
    return 0


def vit_corecctness(model, array):
    trans = transforms.ToTensor()
    classes = np.empty(0)
    low_classes = np.empty(0)
    masked_classes = np.empty(0)

    for i in array:
        for key, value in i.items():
            x, y = load_single_image("./data/"+value["path"])
            value["probab"] = model.predict(x)[0][value["class"]]

    for i in array:
        abc = list(dict(
            sorted(i.items(), key=lambda item: item[1]["probab"], reverse=True)).values())
        for x in range(0, int(len(abc)/2)):

            high_path = "./data/"+abc[x]["path"]
            low_path = "./data/"+abc[(len(abc)-1)-x]["path"]

            low_img, pil = load_single_image(low_path)
            masked_img = torch.unsqueeze(
                trans(vit_mask_image(model, high_path, low_path)), 0)

            classes = np.append(classes, abc[x]["class"])
            low_classes = np.append(
                low_classes, np.argmax(model.predict(low_img)[0]))
            masked_classes = np.append(
                masked_classes, np.argmax(model.predict(masked_img)[0]))

    print("ViT Acc with low images: " +
          str(accuracy_score(classes, low_classes)))
    print("Vit Acc with masked images: " +
          str(accuracy_score(classes, masked_classes)))
    return 0


def cnn_mask_image(model, high, low, class_num):

    x_high, y_high = load_single_image(high)
    temp, mask = model.lime_and_explain(y_high, class_num)
    mask = np.where(mask == 1, 255, mask)

    im2 = Image.composite(just_load_resize_pil(high), just_load_resize_pil(
        low), Image.fromarray(np.uint8(mask)).convert('L'))

    return im2


def vit_mask_image(model, high, low):
    x_high, y_high = load_single_image(high)

    preds, attns = model.predict_and_attents(x_high)

    a_map = generate_attention_map(attns, 20, 16)
    mask = sliding_window_method(a_map)

    mask = np.where(mask == 1, 255, mask)

    im2 = Image.composite(just_load_resize_pil(high), just_load_resize_pil(
        low), Image.fromarray(np.uint8(mask)).convert('L'))
    return im2
