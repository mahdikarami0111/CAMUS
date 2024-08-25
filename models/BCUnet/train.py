from models.BCUnet.BCUnet import BCUnet
from dataset import CAMUS, Wrapper
import torch
import torch.nn as nn
from torch.utils.data import random_split, Subset, DataLoader
from models.BCUnet.Unet_parts import RecallCrossEntropy
from train import select_transform
from tqdm import tqdm
import sklearn.metrics as metrics
import numpy as np


def train(cfg, preset_indices=None):
    dataset = CAMUS(cfg.data)
    batch_size = cfg.batch_size
    num_classes = cfg.num_classes
    image_size = cfg.image_size
    epochs = cfg.max_epoch
    base_lr = cfg.lr
    bilinear = cfg.bilinear

    model = BCUnet(n_channels=1, n_classes=num_classes, bilinear=bilinear).cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr)
    criterion = RecallCrossEntropy()
    softmax2d = nn.Softmax2d()
    EPS = 1e-12

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
            image_batch, gt = sampled_batch[0].to('cuda'), sampled_batch[1].to('cuda')
            out1, out2, out = model(image_batch)
            out = torch.log(softmax2d(out) + EPS)
            loss = criterion(out, gt)
            loss += criterion(torch.log(softmax2d(out1) + EPS), gt)
            loss += criterion(torch.log(softmax2d(out2) + EPS), gt)

            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print(f"epoch: {e}, iteration: {i}, loss: {loss}")






