import torch


def sslLoss(output, target, dists, lambd=1, sigma=0.01):
    # compute sup_loss only on non-zero targets indices
    sup_ids = (target != -1).nonzero()[:, 0]
    sup_loss = criterion(output[sup_ids], target[sup_ids].float())  # scalar
    ssl_loss = (
        torch.zeros_like(target).to(device).type_as(sup_loss)
    )

    dists = dists.type_as(sup_loss)
    target = target.type_as(sup_loss)

    if lambd == 0:
        return sup_loss

    # compute binary outputs
    preds = torch.round(torch.sigmoid(output))
    preds = torch.tensor(torch.where(preds == 0, -1, 1))

    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            ssl_loss[i] += (
                lambd
                * torch.exp(-dists[i, j] ** 2 / sigma)
                * (preds[i, 0] - preds[j, 0]) ** 2
            )

    print("compare losses: ssl: {}, sup: {}".format(ssl_loss.float().mean(), sup_loss))
    return ssl_loss.float().mean() + sup_loss
