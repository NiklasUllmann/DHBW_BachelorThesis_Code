from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import uuid


def create_uuid():
    return str(uuid.uuid4().hex[:10])


def vis_and_save(mask, img):
    img_boundry = mark_boundaries(img / 255.0, mask)
    plt.imshow(img_boundry)
    plt.savefig("./output/lime/"+create_uuid()+".jpg")


