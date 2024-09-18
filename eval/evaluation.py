import torch
from skimage.metrics import hausdorff_distance
import numpy as np
from surface_distance import compute_surface_distances
from surface_distance import compute_average_surface_distance
from medpy.metric.binary import dc
from torch import nn
import sklearn.metrics as metrics


def predict(img, model):
    with torch.no_grad():
        model.eval()
        pred = torch.sigmoid(model(img.unsqueeze(0)))
        return (pred > 0.5).int()


def calculate_dice_metric(model, test_loader, device, single_class=True, sigmoid=True):
    with torch.no_grad():
        model.eval()
        model.to(device)
        avg_score = 0
        softmax2d = nn.Softmax2d()

        for (X, Y) in test_loader:
            (X, Y) = (X.to(device), Y.to(device))

            if single_class:
                if sigmoid:
                    out = torch.sigmoid(model(X)[2])
                else:
                    out = model(X)[2]
                preds = (out > 0.5).int()
            else:
                out = model(X)[2]
                preds = torch.log(softmax2d(out) + 1e-12)
                preds = torch.log(softmax2d(preds) + 1e-12)
                preds = preds.argmax(dim=1, keepdim=True)

            scores = avg_dice_metric_batch(preds, Y)
            avg_score += scores.sum()

        return avg_score / len(test_loader.dataset)


def calculate_Accuracy(confusion):
    confusion = np.asarray(confusion)
    pos = np.sum(confusion, 1).astype(np.float32) # 1 for row
    res = np.sum(confusion, 0).astype(np.float32) # 0 for coloum
    tp = np.diag(confusion).astype(np.float32)
    f1 = 2 * confusion[1][1] / (2 * confusion[1][1] + confusion[1][0] + confusion[0][1])
    IU = tp / (pos + res - tp)
    dice = 2 * tp / (pos+res)
    meanDice = np.mean(dice)
    meanIU = np.mean(IU)
    Acc = np.sum(tp) / np.sum(confusion)
    Se = confusion[1][1] / (confusion[1][1]+confusion[0][1])
    Sp = confusion[0][0] / (confusion[0][0]+confusion[1][0])
    return  IU[1],dice[1],Acc,Se,Sp,IU,f1


def BCUnet_dice(model, test_loader, device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        avg_score = 0
        softmax2d = nn.Softmax2d()
        total = 0

        for (X, Y) in test_loader:
            (X, Y) = (X.to(device)), Y
            out = torch.log(softmax2d(model(X)[2]) + 1e-12)
            ppi = np.argmax(out.cpu().data.numpy(), 1)
            tmp_out = ppi.reshape([-1])
            tmp_gt = Y.reshape([-1])
            my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
            miou, mdice, Acc, Se, Sp, IU, f1 = calculate_Accuracy(my_confusion)
            total += mdice
        return total / len(test_loader)


def avg_dice_metric_batch(preds, masks):
    intersection = (preds * masks).sum(dim=(1, 2, 3)).float()
    return (2 * intersection) / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)))


def new_avg_dice_metric_batch(preds, masks):
    batch_size = preds.shape[0]
    batch_dc = 0
    for i in range(batch_size):
        batch_dc += dc(np.asarray(preds[i, :].unsqueeze(0).cpu()), np.asarray(masks[i, :].unsqueeze(0).cpu()))
    return batch_dc


def calculate_hausdorff_metric(model, test_loader, device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        avg_score = 0

        for (X, Y) in test_loader:
            (X, Y) = (X.to(device), Y.to(device))
            out = torch.sigmoid(model(X))
            preds = (out > 0.5).int()
            preds = preds.to('cpu')
            Y = Y.to('cpu')
            for i in range(preds.shape[0]):
                avg_score += hausdorff_distance(np.asarray(preds[i].squeeze()), np.asarray(Y[i].squeeze()))

        return avg_score / len(test_loader.dataset)


def calculate_mean_distance(model, test_loader, device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        avg_score = 0

        for (X, Y) in test_loader:
            (X, Y) = (X.to(device), Y.to(device))
            out = torch.sigmoid(model(X))
            preds = (out > 0.5).int()
            preds = preds.to('cpu')
            Y = Y.to('cpu')

            for i in range(preds.shape[0]):
                distances = compute_surface_distances(np.asarray(preds[i].squeeze(), dtype=bool), np.asarray(Y[i].squeeze(), dtype=bool), (2, 1))
                avg_dist = compute_average_surface_distance(distances)
                avg_score += (avg_dist[0] + avg_dist[1])/2

        return avg_score / len(test_loader.dataset)


def evaluate_model(model, test_loader, device):
    eval_dict = {
        "Dice": calculate_dice_metric(model, test_loader, device),
        "Hausdorff": calculate_hausdorff_metric(model, test_loader, device),
        "Mean distance": calculate_mean_distance(model, test_loader, device),
    }
    return eval_dict

