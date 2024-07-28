import torch
import torch.nn as nn
from train import select_transform
from dataset import CAMUS, Wrapper
from ..TransUnet.TransUnet import VisionTransformer
from config.TransUnet_cfg import get_TransUnet_config
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.nn.functional import sigmoid
import torch.nn.functional as F
from tqdm import tqdm


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == 1  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if self.n_classes == 1:
            inputs = inputs.unsqueeze(dim=1)
            softmax = False
            inputs = torch.sigmoid(inputs)
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = target.unsqueeze(dim=1)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def train(cfg):
    dataset = CAMUS(cfg.data)
    batch_size = cfg.batch_size
    num_classes = cfg.num_classes
    image_size = cfg.image_size
    epochs = cfg.max_epoch
    base_lr = cfg.lr
    model = VisionTransformer(get_TransUnet_config(), img_size=image_size, num_classes=num_classes).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=0.0001)
    bce_loss = BCEWithLogitsLoss()
    dice_loss = DiceLoss(num_classes)
    train_, test = random_split(dataset, [cfg.train_split, cfg.test_split])
    train_ = Wrapper(train_, select_transform(cfg.transform))
    test = Wrapper(test, select_transform('basic'))

    trainloader = DataLoader(train_, batch_size=batch_size, shuffle=True, pin_memory=True)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=True, pin_memory=True)

    for e in tqdm(range(epochs)):
        model.train()
        for i, sampled_batch in enumerate(trainloader):
            image_batch, mask_batch = sampled_batch[0].to('cuda'), sampled_batch[1].to('cuda')
            outputs = model(image_batch)
            loss_ce = bce_loss(outputs, mask_batch)
            loss_dice = dice_loss(outputs, mask_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 25 == 0:
                print(f"epoch {e} total average loss: {loss} | dice: {loss_dice} | bce: {loss_ce}")

        total_loss = 0
        with torch.no_grad():
            model.eval()
            for i, sampled_batch in enumerate(testloader):
                image_batch, mask_batch = sampled_batch[0].to('cuda'), sampled_batch[1].to('cuda')
                outputs = model(image_batch)
                loss_ce = bce_loss(outputs, mask_batch)
                loss_dice = dice_loss(outputs, mask_batch, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                total_loss += loss
            total_loss /= len(test)
            print(f"epoch {e} total average loss: {total_loss}")



