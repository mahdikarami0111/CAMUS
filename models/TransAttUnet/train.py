import torch
import torch.nn as nn
import os
from train import select_transform
from dataset import CAMUS, Wrapper
from models.TransAttUnet.TransAttUnet import TransAttUnet
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


def init_weights(m):
    """
    Initialize the weights
    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def train(cfg, preset_indices=None):
    dataset = CAMUS(cfg.data)
    batch_size = cfg.batch_size
    num_classes = cfg.num_classes
    image_size = cfg.image_size
    epochs = cfg.max_epoch
    base_lr = cfg.lr
    model = TransAttUnet(in_channels=1, n_classes=num_classes).cuda()
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    bce_loss = torch.nn.modules.loss.BCEWithLogitsLoss()
    dice_loss = DiceLoss(num_classes)

    if preset_indices is None:
        train_, test, val = random_split(dataset, [cfg.train_split, cfg.test_split, cfg.val_split])
    else:
        train_ = Subset(dataset, preset_indices["train"])
        test = Subset(dataset, preset_indices["test"])
        val = Subset(dataset, preset_indices["val"])
    train_ = Wrapper(train_, select_transform(cfg.transform))
    val = Wrapper(val, select_transform('basic'))
    test = Wrapper(test, select_transform('basic'))

    trainloader = DataLoader(train_, batch_size=batch_size, shuffle=True, pin_memory=True)
    valloader = DataLoader(val, batch_size=batch_size, shuffle=True, pin_memory=True)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=True, pin_memory=True)

    for e in tqdm(range(epochs)):
        model.train()
        for i, sampled_batch in enumerate(valloader):
            x, y = sampled_batch[0].to('cuda'), sampled_batch[1].to('cuda')
            outputs = model(x)
            loss_bce = bce_loss(outputs, y)
            loss_dice = dice_loss(outputs, y, sigmoid=True)
            loss = 0.5 * loss_bce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 25 == 0:
                print(f"iteration {i}/{len(valloader)} total loss: {loss} | dice: {loss_dice} | bce: {loss_bce}")

        total_loss = 0
        with torch.no_grad():
            model.eval()
            for i, sampled_batch in enumerate(valloader):
                x, y = sampled_batch[0].to('cuda'), sampled_batch[1].to('cuda')
                outputs = model(x)
                loss_bce = bce_loss(outputs, y)
                loss_dice = dice_loss(outputs, y, sigmoid=True)
                loss = 0.5 * loss_bce + 0.5 * loss_dice
                total_loss += loss
            total_loss /= len(valloader)
            print(f"epoch {e}/{epochs} total loss: {total_loss}")
            print(calculate_dice_metric(model, testloader, "cuda", sigmoid=True))


