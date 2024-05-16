import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np


def show_tensor_img(img, seg):
    print(img.shape)
    img = img.permute(1, 2, 0)
    x_size = img.shape[1]
    y_size = img.shape[0]
    seg_padded = np.pad(seg, 1)
    seg_padded = seg_padded.astype('uint8')
    f = plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    plt.imshow(np.ma.masked_where(seg == 0, seg), alpha=0.25)
    plt.show()
