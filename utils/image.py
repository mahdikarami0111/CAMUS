import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np


def show_tensor_img(img, seg):
    img = img.permute(1, 2, 0)
    x_size = img.shape[1]
    y_size = img.shape[0]
    seg = seg.squeeze(0)
    seg_padded = np.pad(seg, 1)
    seg_padded = seg_padded.astype('uint8')
    f = plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    plt.imshow(np.ma.masked_where(seg == 0, seg), alpha=0.4)
    plt.show()


def compare_masks(img, pred, mask):
    img = img.permute(1, 2, 0)
    fig = plt.figure(figsize=(10, 20))
    fig.add_subplot(1, 2, 1)
    mask = mask.squeeze(0)
    mask = np.array(mask)
    mask = mask.astype('uint8')
    pred = np.array(pred)
    pred = pred.astype('uint8')
    plt.imshow(img, cmap='gray')
    plt.imshow(np.ma.masked_where(mask == 0, mask), alpha=0.2)
    plt.title("Mask")
    fig.add_subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.imshow(np.ma.masked_where(pred == 0, pred), alpha=0.2)
    plt.title("Prediction")
    plt.show()



