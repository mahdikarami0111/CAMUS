import torch
from skimage.metrics import hausdorff_distance
import numpy as np
from surface_distance import compute_surface_distances
from surface_distance import compute_average_surface_distance


def predict(img, model):
    with torch.no_grad():
        model.eval()
        pred = torch.sigmoid(model(img.unsqueeze(0)))
        return (pred > 0.5).int()


def calculate_dice_metric(model, test_loader, device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        avg_score = 0

        for (X, Y) in test_loader:
            (X, Y) = (X.to(device), Y.to(device))
            out = torch.sigmoid(model(X))
            preds = (out > 0.5).int()
            scores = avg_dice_metric_batch(preds, Y)
            avg_score += scores.sum()

        return avg_score / len(test_loader.dataset)


def avg_dice_metric_batch(preds, masks):
    intersection = (preds * masks).sum(dim=(1, 2, 3)).float()
    return (2 * intersection) / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)))


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

