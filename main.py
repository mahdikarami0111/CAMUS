# Based
import pywt
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import skimage
from tqdm import tqdm
import torch.utils.data
from torch.utils.data import DataLoader
from utils.image import *
from models.Unet import Unet
from preprocess.preprocessor import *
import yaml
from train import select_transform
from torch.optim.lr_scheduler import ExponentialLR
from eval.evaluation import *
from utils import save
from dataset import CAMUS
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from utils.debug import *
from train import train
from dataset import Wrapper
from train import train_K_fold
from dotmap import DotMap
from skimage.metrics import hausdorff_distance
from models.TransUnet.TransUnet import VisionTransformer
from config import TransUnet_cfg
from train import train
from config.Unet_cfg import get_config
from torch.utils.data import random_split
from torch.utils.data import Subset
from models.BCUnet.Convnext import LayerNorm
from models.BCUnet.BCUnet import BCUnet
from config.BCUnet_cfg import get_train_cfg
from models.BCUnet.train import train
from  models.DUCKnet.train import train as train_ducknet
from config.DuckUnet_cfg import get_DuckNet_train_config
from models.FCT.FCT import FCT
from models.FCT.train import train as train_fct
from config.FCT_cfg import get_FCT_config
from models.TransAttUnet.transformer_parts import PositionEmbeddingLearned
from models.TransAttUnet.TransAttUnet import TransAttUnet
from models.TransAttUnet.train import train as train_trans_att_unet
from config.TransAttUnet_cfg import get_TransAttUnet_train_config
from utils.wavelet import IDWT_2D
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from models.WSegNet.WSegnet import WSegNetVGG, make_w_layers



if __name__ == '__main__':
    # model = save.load_model("TransUnet", "transunetV2").to('cuda')
    # dataset = CAMUS({
    #     "root": "data/database_expanded",
    #     "device": "cuda",
    #     "type": "N 4CH",
    # })
    # indices = save.load_indices()
    # test = Subset(dataset, indices["test"])
    # test_set = Wrapper(test, transform=select_transform('basic'))
    #
    # test_loader = DataLoader(test_set, batch_size=8, shuffle=True, pin_memory=True)
    # print(calculate_dice_metric(model, test_loader, 'cuda'))
    # train(get_config(), save.load_indices())
    # _____________________________________________

    # dataset = CAMUS({
    #     "root": "data/database_expanded",
    #     "device": "cuda",
    #     "type": "N 4CH",
    # })
    # dataset = Wrapper(dataset, transform=select_transform("basic"))
    # model = BCUnet(1, 2).to('cuda')
    # loader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True)
    # for data in loader:
    #     X, Y = data
    #     out1, out2, out = model(X.to('cuda'))
    #     print(out.shape)
    #     break
    # __________________________________________________

    # model = save.load_model("BCUnet", "BCUnet").cuda()
    # indices = save.load_indices()
    # dataset = CAMUS({
    #     "root": "data/database_expanded",
    #     "device": "cuda",
    #     "type": "N 4CH",
    # })
    # test_set = Subset(dataset, indices["test"])
    # test_set = Wrapper(test_set, select_transform("basic"))
    # test_loader = DataLoader(test_set, batch_size=2, shuffle=True, pin_memory=True)
    # print(BCUnet_dice(model, test_loader, "cuda"))
    # __________________________________________________

    # model = save.load_model("ducknet", "DuckNet").to('cuda')
    # dataset = CAMUS({
    #     "root": "data/database_expanded",
    #     "device": "cuda",
    #     "type": "N 4CH",
    # })
    # indices = save.load_indices()
    # test = Subset(dataset, indices["test"])
    # test_set = Wrapper(test, transform=select_transform('basic'))
    #
    # test_loader = DataLoader(test_set, batch_size=8, shuffle=True, pin_memory=True)
    # print(calculate_dice_metric(model, test_loader, 'cuda'))
    # ___________________________________________________

    # train_trans_att_unet(get_TransAttUnet_train_config(), save.load_indices("data"))

    # dataset = CAMUS({
    #     "root": "data/database_expanded",
    #     "device": "cuda",
    #     "type": "N 4CH",
    # })
    # img, mask = dataset[3526]
    # show_tensor_img(img, mask)
    # img = np.stack((img,)*3, axis=-1).astype(np.uint8)
    # noisy = img_as_float(img)
    #
    # fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 5), sharex=True, sharey=True)
    # plt.gray()
    #
    # sigma_est = estimate_sigma(noisy, channel_axis=-1, average_sigmas=True)
    #
    # im_bayes = denoise_wavelet(
    #     noisy,
    #     channel_axis=-1,
    #     convert2ycbcr=True,
    #     method='BayesShrink',
    #     mode='soft',
    #     rescale_sigma=True,
    # )
    # print(np.unique(im_bayes))
    # im_visushrink = denoise_wavelet(
    #     noisy,
    #     channel_axis=-1,
    #     convert2ycbcr=True,
    #     method='VisuShrink',
    #     mode='soft',
    #     sigma=sigma_est * 8,
    #     rescale_sigma=True,
    # )
    # im_visushrink2 = denoise_wavelet(
    #     noisy,
    #     channel_axis=-1,
    #     convert2ycbcr=True,
    #     method='VisuShrink',
    #     mode='soft',
    #     sigma=sigma_est / 2,
    #     rescale_sigma=True,
    # )
    # im_visushrink4 = denoise_wavelet(
    #     noisy,
    #     channel_axis=-1,
    #     convert2ycbcr=True,
    #     method='VisuShrink',
    #     mode='soft',
    #     sigma=sigma_est / 4,
    #     rescale_sigma=True,
    # )
    #
    # ax[0, 0].imshow(noisy)
    # ax[0, 0].axis('off')
    # ax[0, 0].set_title('noisy')
    #
    # ax[0, 1].imshow(im_bayes)
    # ax[0, 1].axis('off')
    # ax[0, 1].set_title('bayes')
    #
    # ax[1, 1].imshow(im_visushrink2)
    # ax[1, 1].axis('off')
    # ax[1, 1].set_title("v2")
    #
    # ax[1, 0].imshow(im_visushrink4)
    # ax[1, 0].axis('off')
    # ax[1, 0].set_title("v4")
    #
    # ax[1, 2].imshow(im_visushrink)
    # ax[1, 2].axis('off')
    # ax[1, 2].set_title("v")
    #
    # ax[0, 2].imshow(mask)
    # ax[0, 2].axis('off')
    # ax[0, 2].set_title("mask")
    # plt.show()
    # expand_series("data/database", "data/database_wavelet_VisuShrink_2", transform_method="VisuShrink", param=2)
    #______________________________________________________
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    model = WSegNetVGG(make_w_layers(cfg, batch_norm=True))




