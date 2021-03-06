from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import uuid
import numpy as np

from utils.dataset import just_load_resize_pil
import matplotlib.patches as mpatches



def create_uuid():
    return str(uuid.uuid4().hex[:10])


def vis_and_save(mask, path):

    pil_img = np.asarray(just_load_resize_pil(path))
    img_boundry = mark_boundaries(pil_img / 255.0, mask)
    plt.imshow(img_boundry)
    plt.axis("off")
    plt.savefig("./output/lime/test/"+create_uuid()+".jpg")


