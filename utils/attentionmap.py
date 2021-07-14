import torch
import numpy as np
from PIL import Image
import uuid
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches



def create_uuid():
    return str(uuid.uuid4().hex[:10])


def generate_attention_map(att_tensor, patches_per_row, patch_size):

    attn = torch.squeeze(input=att_tensor, dim=0)
    attn = torch.mean(attn, dim=[0, 1])
    attn = torch.sum(attn, dim=0)
    attn = attn.detach().cpu().numpy()
    attn = attn[1:].copy()
    attn = np.interp(attn, (attn.min(), attn.max()), (0, +1))

    b = np.reshape(attn, (patches_per_row, patches_per_row))
    b = np.kron(b, np.ones((patch_size, patch_size), dtype=b.dtype))

    return b


def visualise_attention(att_tensor, patch_size, patches_per_row, img_size, orig_path):

    b= generate_attention_map(att_tensor=att_tensor, patches_per_row=patches_per_row, patch_size=patch_size)

    
    im1 = Image.fromarray(b*255)

    rgbimg = Image.new("RGB", im1.size)
    rgbimg.paste(im1)

    im2 = Image.open(orig_path)
    im2 = im2.resize((img_size, img_size))

    out = Image.blend(rgbimg, im2, 0.2)

    out.save("./output/atm/" + create_uuid() + ".jpg")
    """
    plt.subplot(1, 2, 1)
    plt.imshow(b, cmap='gray')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(out)
    plt.axis("off")

    plt.savefig("./output/atm/" + create_uuid() + ".jpg")
    """
    return 0


def sliding_window_method(a_map):

    windows_size = 25
    avg_max = 0

    bit_mask = np.zeros((320, 320), dtype=int)

    i_max = 0
    i_end_max = 0
    j_max = 0
    j_end_max = 0

    for a in range(25):
        i_max = 0
        i_end_max = 0
        j_max = 0
        j_end_max = 0
        avg_max = 0
        for i in range(0, a_map.shape[0] - windows_size + 1):
            for j in range(0, a_map.shape[1]-windows_size+1):
                window = a_map[i: i + windows_size, j: j + windows_size]

                if (np.mean(window) >= avg_max):
                    i_max = i
                    i_end_max = i+windows_size
                    j_max = j
                    j_end_max = j + windows_size
                    avg_max = np.mean(window)

        bit_mask[i_max:i_end_max, j_max:j_end_max] = 1
        a_map[i_max:i_end_max, j_max:j_end_max] = 0

    return bit_mask
