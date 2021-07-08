import torch
import numpy as np
from PIL import Image
import uuid


def create_uuid():
    return str(uuid.uuid4().hex[:10])


def generate_attention_map(att_tensor, patches_per_row, patch_size):

    attn = torch.squeeze(input=att_tensor, dim=0)
    attn = torch.mean(attn, dim=[0, 1])
    attn = torch.sum(attn, dim=0)
    attn = attn.detach().cpu().numpy()
    attn = attn[:-1].copy()
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

    out.save("./output/atm/"+create_uuid()+".jpg")

    return 0
