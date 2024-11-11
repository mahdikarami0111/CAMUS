import torch
import torch.nn as nn
import os
from train import select_transform
from dataset import CAMUSP, Wrapper
from models.WSegNet.WUnetV2 import WUNet
from config.TransUnet_cfg import get_TransUnet_config
from torch.utils.data import random_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.nn.modules.loss import BCEWithLogitsLoss
from utils.save import save_model
from torch.nn.functional import sigmoid
import torch.nn.functional as F
from tqdm import tqdm
from utils.losses import DiceLoss
from eval.evaluation import calculate_dice_metric
from models.UneXt.UneXt import UneXt


def train(cfg, preset_indices=None):
    batch_size = cfg.batch_size
    num_classes = cfg.num_classes
    image_size = cfg.image_size
    epochs = cfg.max_epoch
    base_lr = cfg.lr

    model = UneXt(in_chans=1).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    bce_loss = BCEWithLogitsLoss()
    dice_loss = DiceLoss(num_classes)

    if preset_indices is None:
        pass
    else:
        train_ = CAMUSP(cfg.data, indices=preset_indices["train"])
        test = CAMUSP(cfg.data, indices=preset_indices["test"])
        val = CAMUSP(cfg.data, indices=preset_indices["val"])

    train_ = Wrapper(train_, select_transform(cfg.transform))
    val = Wrapper(val, select_transform('basic'))
    test = Wrapper(test, select_transform('basic'))

    trainloader = DataLoader(train_, batch_size=batch_size, shuffle=True, pin_memory=True)
    valloader = DataLoader(val, batch_size=batch_size, shuffle=True, pin_memory=True)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=True, pin_memory=True)

    min_loss = 999999
    best_model = None

    for e in tqdm(range(epochs)):
        model.train()
        for i, sampled_batch in enumerate(trainloader):
            image_batch, mask_batch = sampled_batch[0].to('cuda'), sampled_batch[1].to('cuda')
            outputs = model(image_batch)
            loss_ce = bce_loss(outputs, mask_batch)
            # loss_dice = dice_loss(outputs, mask_batch)
            loss = loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 25 == 0 or i == len(trainloader) - 1:
                print(f"iteration {i}/{len(trainloader) - 1} total loss: {loss} | bce: {loss_ce}")
        scheduler.step()
        print(scheduler.get_lr())

        total_loss = 0
        with torch.no_grad():
            model.eval()
            for i, sampled_batch in enumerate(valloader):
                image_batch, mask_batch = sampled_batch[0].to('cuda'), sampled_batch[1].to('cuda')
                outputs = model(image_batch)
                loss_ce = bce_loss(outputs, mask_batch)
                # loss_dice = dice_loss(outputs, mask_batch, softmax=True)
                loss = loss_ce
                total_loss += loss
            total_loss /= len(val)
            print(f"epoch {e}/{epochs} total loss: {total_loss}")
            print(calculate_dice_metric(model, testloader, "cuda", sigmoid=True))


