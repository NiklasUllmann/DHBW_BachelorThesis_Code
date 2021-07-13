from matplotlib import pyplot as plt
from skimage import measure
from utils.dataset import load_image_and_mirror
from utils.attentionmap import generate_attention_map
from skimage.segmentation import felzenszwalb, mark_boundaries
import numpy as np


def contour_method(vitModel):
    x, x_mir, y, y_mir = load_image_and_mirror(
        "./data/val/n03417042/ILSVRC2012_val_00006922.JPEG")
    preds, attn = vitModel.predict_and_attents(x)
    a_map = generate_attention_map(attn, patches_per_row=20, patch_size=16)
    # Find contours at a constant value of 0.8
    contours = measure.find_contours(a_map, 0.5)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(a_map, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def segmentation_method(vitModel):
    scale = 500
    sigma = 1.5
    min_size = 500

    x, x_mir, y, y_mir = load_image_and_mirror(
        "./data/val/n03417042/ILSVRC2012_val_00006922.JPEG")
    preds, attn = vitModel.predict_and_attents(x)
    a_map = generate_attention_map(attn, patches_per_row=20, patch_size=16)

    segments_fz = felzenszwalb(
        a_map, scale=scale, sigma=sigma, min_size=min_size)

    print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")

    fig, ax = plt.subplots(2, int(len(np.unique(segments_fz))/2),
                           figsize=(10, 10), sharex=True, sharey=True)

    for i in range(int(len(np.unique(segments_fz))/2)):
        ax[0, i].imshow(mark_boundaries(a_map, [segments_fz[:, :] == i][0]))

    for i in range(int(len(np.unique(segments_fz))/2), int(len(np.unique(segments_fz)))):
        ax[1, int(len(np.unique(segments_fz))/2) -
           i].imshow(mark_boundaries(a_map, [segments_fz[:, :] == i][0]))

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()
