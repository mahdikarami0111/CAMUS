import torch.nn as nn
from models.FCT.FCT import FCT
from dataset import CAMUS, Wrapper
import torch
from torch.utils.data import random_split, Subset, DataLoader
from config.DuckUnet_cfg import get_DuckNet_config
from train import select_transform
import torch.nn.functional as F
from tqdm import tqdm
import os
from utils.save import save_model
from torch.nn.modules.loss import BCEWithLogitsLoss
from utils.losses import DiceLoss
from eval.evaluation import calculate_dice_metric


def train(cfg, preset_indices=None):
    dataset = CAMUS(cfg.data)
    batch_size = cfg.batch_size
    num_classes = cfg.num_classes
    img_size = cfg.img_size
    epochs = cfg.max_epoch
    base_lr = cfg.lr
    model = FCT(img_size).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    bce_loss = BCEWithLogitsLoss()
    dice_loss = DiceLoss(num_classes)
    if preset_indices is None:
        train_, test, val = random_split(dataset, [cfg.train_split, cfg.test_split, cfg.val_split])
    else:
        train_ = Subset(dataset, preset_indices["train"])
        test = Subset(dataset, preset_indices["test"])
        val = Subset(dataset, preset_indices["val"])
    train_ = Wrapper(train_, select_transform(cfg.transform))
    val = Wrapper(val, select_transform('basic'))

    trainloader = DataLoader(train_, batch_size=batch_size, shuffle=True, pin_memory=True)
    valloader = DataLoader(val, batch_size=batch_size, shuffle=True, pin_memory=True)

    min_loss = 999999
    best_model = None

    for e in tqdm(range(epochs)):
        model.train()
        for i, sampled_batch in enumerate(trainloader):
            x, y = sampled_batch[0].to('cuda'), sampled_batch[1].to('cuda')
            pred_y = model(x)
            down1 = F.interpolate(y, model.img_size // 2)
            down2 = F.interpolate(y, model.img_size // 4)
            loss = (bce_loss(pred_y[2], y) * 0.57 + bce_loss(pred_y[1], down1) * 0.29 +
                    bce_loss(pred_y[0], down2) * 0.14)
            loss_ = dice_loss(pred_y[2], y)
            loss = 0.5 * loss + 0.5 * loss_
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"iteration {i}/{len(trainloader)} total loss: {loss}")


        total_loss = 0
        with torch.no_grad():
            model.eval()
            for i, sampled_batch in enumerate(valloader):
                x, y = sampled_batch[0].to('cuda'), sampled_batch[1].to('cuda')
                pred_y = model(x)
                down1 = F.interpolate(y, model.img_size // 2)
                down2 = F.interpolate(y, model.img_size // 4)
                loss = (bce_loss(pred_y[2], y) * 0.57 + bce_loss(pred_y[1], down1) * 0.29 +
                        bce_loss(pred_y[0], down2) * 0.14)
                loss_ = dice_loss(pred_y[2], y)
                loss = 0.5 * loss + 0.5 * loss_
                total_loss += loss
                print(f"epoch {e}/{epochs} total loss: {total_loss}")
                print(calculate_dice_metric(model, trainloader, "cuda"))



