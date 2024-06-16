import torch


def predict(img, model):
    with torch.no_grad():
        model.eval()
        pred = torch.sigmoid(model(img.unsqueeze(0)))
        return (pred > 0.5).int()


def calculate_dice_metric(model, test_loader, device):
    with torch.no_grad():
        model.eval()
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
