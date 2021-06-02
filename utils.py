import matplotlib.pyplot as plt
import numpy as np

def imshow(img, label):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(str(label))
    plt.show()