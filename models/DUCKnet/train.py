from models.DUCKnet.Ducknet import DuckNet
from dataset import CAMUS, Wrapper
import torch
import torch.nn as nn
from torch.utils.data import random_split, Subset, DataLoader
from config.DuckUnet_cfg import get_DuckNet_config
from train import select_transform
from tqdm import tqdm
import os
from utils.save import save_model
from torch.nn.modules.loss import BCELoss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False, weight=None):
        if self.n_classes == 1:
            inputs = inputs.unsqueeze(dim=1)
        elif softmax == True:
            inputs = torch.softmax(inputs, dim=1)
        target = target.unsqueeze(dim=1)
        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def train(cfg, preset_indices=None):
    dataset = CAMUS(cfg.data)
    batch_size = cfg.batch_size
    num_classes = cfg.num_classes
    image_size = cfg.image_size
    epochs = cfg.max_epoch
    base_lr = cfg.lr

    model = DuckNet(in_channels=1, out_channels=1, depth=5, init_features=32).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    loss_fn = DiceLoss(num_classes)

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
            image_batch, mask_batch = sampled_batch[0].to('cuda'), sampled_batch[1].to('cuda')
            output = model(image_batch)
            loss = loss_fn(output, mask_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 25 == 0:
                print(f"iteration {i}/{len(trainloader)} total loss: {loss}")
        total_loss = 0
        with torch.no_grad():
            model.eval()
            for i, sampled_batch in enumerate(valloader):
                image_batch, mask_batch = sampled_batch[0].to('cuda'), sampled_batch[1].to('cuda')
                output = model(image_batch)
                loss = loss_fn(output, mask_batch)
                total_loss += loss
            total_loss /= len(val)
            print(f"epoch {e}/{epochs} total loss: {total_loss}")
            if total_loss < min_loss:
                min_loss = total_loss
                save_model(model.state_dict(), f"{e}")
                if best_model is not None:
                    os.remove("models/trained_models/" + best_model + ".pth")
                best_model = f"{e}"
