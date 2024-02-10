# %% import libraries
import numpy as np
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import argmax, unique

# %% get color for mask
def colors(numb):
    numb = numb.item()
    return {0: (233, 233, 233), 1: (254, 0, 255)}.get(numb)
    
# %% color mask
def color_mask(tensor_mask):
    colored = np.zeros((3, tensor_mask.shape[0], tensor_mask.shape[1]), dtype='uint8')
    for i in unique(tensor_mask):
        x = tensor_mask == i
        color = colors(i)
        colored[0, x] = color[2]
        colored[1, x] = color[1]
        colored[2, x] = color[0]
    return np.transpose(colored, (1, 2, 0))

# %% show image and predicted masks
def show_mask(tensor_image, tensor_mask, logits=True):
    """
    Receives a ground truth tensor and logits prediction
    :param tensor_image: b-scan image
    :param tensor_mask: output of the model, size CxWxH if logits, WxH if not logits
    :param logits: true if sending the logits prediction
    """
    if logits:
        tensor_mask = F.softmax(tensor_mask, dim=0)
        tensor_mask = argmax(tensor_mask, dim=0)
    mask = color_mask(tensor_mask)

    img = tensor_image.data.numpy()
    img = img.transpose((1, 2, 0))

    fig = plt.figure(figsize=(8., 8.5))
    rows = 2
    columns = 1
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("B-scan")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(mask)
    plt.axis('off')
    plt.title("Mask")
    fig.show()
